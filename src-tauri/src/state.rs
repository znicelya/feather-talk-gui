//! Desktop ownership of projects, resource preparation and native jobs.
mod jobs;
#[cfg(test)]
mod tests;

use crate::{
    projects::{ProjectStore, regular_file},
    resources::ResourceManager,
};
use feathertalk_core::*;
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Condvar, Mutex, MutexGuard},
    thread,
};

pub const TASK_EVENT: &str = "feathertalk://task";

#[derive(Clone)]
pub struct AppState(Arc<Inner>);

struct Inner {
    projects: ProjectStore,
    resources: ResourceManager,
    tasks: Mutex<Tasks>,
    idle: Condvar,
    devices: Mutex<Vec<DeviceDescriptor>>,
    /// Called in event order while publishing the snapshot. Must not reenter AppState.
    emit: Box<dyn Fn(WorkerEvent) + Send + Sync>,
}

#[derive(Default)]
struct Tasks {
    snapshot: Option<TaskSnapshot>,
    cancel: Option<Cancellation>,
    discovery: Option<Cancellation>,
    closing: bool,
}

#[derive(Debug, PartialEq, Eq)]
pub enum CloseState {
    Ready,
    Waiting,
    AlreadyWaiting,
}

impl AppState {
    pub fn new(
        projects: ProjectStore,
        resources: ResourceManager,
        emit: impl Fn(WorkerEvent) + Send + Sync + 'static,
    ) -> Self {
        Self(Arc::new(Inner {
            projects,
            resources,
            emit: Box::new(emit),
            tasks: Mutex::default(),
            idle: Condvar::new(),
            devices: Mutex::default(),
        }))
    }

    pub fn list_projects(&self) -> Result<Vec<ProjectSummary>> {
        self.0.projects.list()
    }
    pub fn create_project(&self, project: NewProject) -> Result<ProjectSummary> {
        self.0.projects.create(project)
    }
    pub fn get_project(&self, id: ProjectId) -> Result<ProjectManifest> {
        self.0.projects.get(id)
    }
    pub fn project_directory(&self, id: ProjectId) -> Result<PathBuf> {
        self.0.projects.directory(id)
    }

    pub fn open_project(&self, directory: &Path) -> Result<ProjectSummary> {
        let (root, manifest) = self.0.projects.inspect(directory)?;
        let guard = self.tasks()?;
        if let Some(active) = &guard.snapshot
            && active.status.is_active()
            && active.project_id == Some(manifest.id)
            && self.project_directory(manifest.id).ok().as_ref() != Some(&root)
        {
            return Err(busy(
                "This project cannot be moved while its task is active",
            ));
        }
        self.0.projects.register_open(&root, &manifest)
    }

    pub fn resource_selections(&self) -> Vec<ResourceSelection> {
        self.0.resources.local_selections()
    }
    pub fn get_resources(&self, target: &TargetPlatform) -> Result<ResourceInventory> {
        self.0.resources.inventory(target)
    }

    pub fn prepare_resources(&self, selection: ResourceSelection) -> Result<TaskId> {
        validate_local(&selection.target)?;
        if selection.resources.is_empty()
            || !self.resource_selections().iter().any(|configured| {
                configured.target == selection.target
                    && selection
                        .resources
                        .iter()
                        .all(|pin| configured.resources.contains(pin))
            })
            || selection.resources.iter().enumerate().any(|(i, pin)| {
                selection.resources[..i]
                    .iter()
                    .any(|other| other.id == pin.id)
            })
        {
            return Err(invalid("Select configured resource pins for this backend"));
        }
        self.submit(
            JobKind::PrepareResources,
            None,
            None,
            move |app, id, cancel| {
                let observer = ResourceObserver(app.clone(), id);
                app.payload(
                    id,
                    EventPayload::Started {
                        job: JobKind::PrepareResources,
                    },
                )?;
                let result = app.0.resources.prepare(&selection, &observer, &cancel)?;
                if !result.ready {
                    return Err(AppError::new(
                        ErrorCode::ResourceMissing,
                        "resources",
                        "Some requested resources are unavailable",
                    ));
                }
                app.reset_devices()?;
                Ok(Conclusion::Local(EventPayload::Completed {
                    result: ArtifactResult::Resources(result),
                }))
            },
        )
    }

    pub fn import_resource(&self, target: TargetPlatform, manifest: PathBuf) -> Result<TaskId> {
        validate_local(&target)?;
        if !manifest.is_absolute() {
            return Err(invalid("Select the manifest of a configured local package"));
        }
        self.submit(
            JobKind::PrepareResources,
            None,
            None,
            move |app, id, cancel| {
                let observer = ResourceObserver(app.clone(), id);
                app.payload(
                    id,
                    EventPayload::Started {
                        job: JobKind::PrepareResources,
                    },
                )?;
                app.0
                    .resources
                    .import_local_observed(&manifest, &observer, &cancel)?;
                app.reset_devices()?;
                let result = app.get_resources(&target)?;
                Ok(Conclusion::Local(EventPayload::Completed {
                    result: ArtifactResult::Resources(result),
                }))
            },
        )
    }

    pub fn start_training(&self, request: TrainRequest) -> Result<TaskId> {
        request.config.validate()?;
        let project = request.project_id;
        let run = request.resume.as_ref().map(|model| model.run_id);
        if request
            .resume
            .as_ref()
            .is_some_and(|model| model.project_id != project)
        {
            return Err(invalid("Select a checkpoint committed by this project"));
        }
        self.submit(
            JobKind::Train,
            Some(project),
            run,
            move |app, id, cancel| jobs::run(app, id, JobSpec::Train(request), cancel),
        )
    }

    pub fn start_generation(&self, request: GenerateRequest) -> Result<TaskId> {
        let project = request.project_id;
        if request.model_ref.project_id != project
            || !request.audio_path.is_absolute()
            || !request.output_path.is_absolute()
            || !is_mp4(&request.output_path)
        {
            return Err(invalid(
                "Select this project's checkpoint, an audio file, and an absolute MP4 output path",
            ));
        }
        self.submit(
            JobKind::Generate,
            Some(project),
            None,
            move |app, id, cancel| jobs::run(app, id, JobSpec::Generate(request), cancel),
        )
    }

    pub fn read_metrics(&self, request: &MetricsPageRequest) -> Result<MetricsPage> {
        let root = self.0.projects.directory(request.project_id)?;
        history::metrics_page(&root, request)
    }

    pub fn list_devices(&self) -> Result<Vec<DeviceDescriptor>> {
        jobs::list_devices(self)
    }

    pub fn snapshot(&self) -> Result<Option<TaskSnapshot>> {
        Ok(self.tasks()?.snapshot.clone())
    }

    pub fn cancel_task(&self, id: TaskId) -> Result<()> {
        let mut tasks = self.tasks()?;
        if tasks.snapshot.as_ref().is_none_or(|s| s.task_id != id) {
            return Err(invalid("This task is no longer the current task"));
        }
        tasks.cancel();
        Ok(())
    }

    pub fn request_close(&self) -> Result<CloseState> {
        let mut tasks = self.tasks()?;
        if !tasks.is_busy() {
            tasks.closing = true;
            return Ok(CloseState::Ready);
        }
        if tasks.closing {
            return Ok(CloseState::AlreadyWaiting);
        }
        tasks.closing = true;
        tasks.cancel();
        if let Some(flag) = &tasks.discovery {
            flag.cancel();
        }
        Ok(CloseState::Waiting)
    }

    pub fn wait_until_idle(&self) -> Result<()> {
        drop(
            self.0
                .idle
                .wait_while(self.tasks()?, |tasks| tasks.is_busy())
                .map_err(state_error)?,
        );
        Ok(())
    }

    pub fn completed_video(&self, project: ProjectId, task: TaskId) -> Result<PathBuf> {
        let snapshot = self.snapshot()?.filter(|s| {
            s.project_id == Some(project) && s.task_id == task && s.status == TaskStatus::Completed
        });
        let Some(ArtifactResult::Video(result)) = snapshot.and_then(|s| s.result) else {
            return Err(invalid("No completed video for this project and task"));
        };
        let path = regular_file(&result.output_path)?;
        if path != result.output_path || !is_mp4(&path) {
            return Err(invalid(
                "The completed video has moved or been replaced by a link",
            ));
        }
        Ok(path)
    }

    fn tasks(&self) -> Result<MutexGuard<'_, Tasks>> {
        self.0.tasks.lock().map_err(state_error)
    }

    fn reset_devices(&self) -> Result<()> {
        self.0.devices.lock().map_err(state_error)?.clear();
        Ok(())
    }

    fn submit(
        &self,
        job: JobKind,
        project_id: Option<ProjectId>,
        run_id: Option<TrainingRunId>,
        run: impl FnOnce(&Self, TaskId, Cancellation) -> Result<Conclusion> + Send + 'static,
    ) -> Result<TaskId> {
        let id = TaskId::new();
        let cancel = Cancellation::new();
        {
            let mut tasks = self.tasks()?;
            if tasks.closing || tasks.cancel.is_some() {
                return Err(busy("Another task is active or the application is closing"));
            }
            if let Some(flag) = &tasks.discovery {
                flag.cancel();
            }
            tasks.snapshot = Some(TaskSnapshot {
                task_id: id,
                project_id,
                run_id,
                job,
                status: TaskStatus::Queued,
                last_seq: 0,
                latest_event: None,
                result: None,
                error: None,
                progress: None,
                metrics: None,
                checkpoint: None,
            });
            tasks.cancel = Some(cancel.clone());
        }
        let app = self.clone();
        if let Err(e) = thread::Builder::new()
            .name(format!("feathertalk-{id}"))
            .spawn(move || {
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    drop(
                        app.0
                            .idle
                            .wait_while(app.tasks()?, |tasks| tasks.discovery.is_some())
                            .map_err(state_error)?,
                    );
                    run(&app, id, cancel)
                }))
                .unwrap_or_else(|_| Err(state_error("Background task panicked")));
                let terminal = match result {
                    Ok(Conclusion::Worker(event)) => app.record(event),
                    Ok(Conclusion::Local(payload)) => app.payload(id, payload),
                    Err(e) if e.code == ErrorCode::Cancelled => {
                        app.payload(id, EventPayload::Cancelled { checkpoint: None })
                    }
                    Err(error) => app.payload(id, EventPayload::Failed { error }),
                };
                if let Err(error) = terminal {
                    let _ = app.payload(id, EventPayload::Failed { error });
                }
            })
        {
            let error = state_error(e);
            self.payload(
                id,
                EventPayload::Failed {
                    error: error.clone(),
                },
            )?;
            return Err(error);
        }
        Ok(id)
    }

    fn payload(&self, id: TaskId, payload: EventPayload) -> Result<()> {
        let mut tasks = self.tasks()?;
        let last = tasks.owner(id)?.last_seq;
        let event = WorkerEvent::new(
            Some(id),
            last.checked_add(1)
                .ok_or_else(|| protocol_error("Task sequence exhausted"))?,
            payload,
        )?;
        self.publish(&mut tasks, event)
    }

    fn record(&self, event: WorkerEvent) -> Result<()> {
        let mut tasks = self.tasks()?;
        self.publish(&mut tasks, event)
    }

    fn publish(&self, tasks: &mut Tasks, event: WorkerEvent) -> Result<()> {
        event.validate()?;
        let id = event
            .task_id
            .ok_or_else(|| protocol_error("Control event in desktop task stream"))?;
        let snapshot = tasks.owner(id)?;
        update_snapshot(snapshot, &event)?;
        if !snapshot.status.is_active() {
            tasks.cancel.take();
            self.0.idle.notify_all();
        }
        (self.0.emit)(event);
        Ok(())
    }
}

enum Conclusion {
    Local(EventPayload),
    Worker(WorkerEvent),
}

struct ResourceObserver(AppState, TaskId);

impl TaskObserver for ResourceObserver {
    fn emit(&self, payload: EventPayload) -> Result<()> {
        if !matches!(&payload, EventPayload::Progress { .. }) {
            return Err(protocol_error("Resource callbacks only report progress"));
        }
        self.0.payload(self.1, payload)
    }
}

impl Tasks {
    fn is_busy(&self) -> bool {
        self.cancel.is_some() || self.discovery.is_some()
    }

    fn owner(&mut self, id: TaskId) -> Result<&mut TaskSnapshot> {
        self.snapshot
            .as_mut()
            .filter(|s| s.task_id == id && s.status.is_active())
            .ok_or_else(|| protocol_error("Event does not own the current desktop task"))
    }
    fn cancel(&mut self) {
        if let Some(flag) = &self.cancel {
            flag.cancel();
            if let Some(snapshot) = &mut self.snapshot {
                snapshot.status = TaskStatus::Cancelling;
            }
        }
    }
}

fn update_snapshot(snapshot: &mut TaskSnapshot, event: &WorkerEvent) -> Result<()> {
    if event.seq <= snapshot.last_seq && snapshot.latest_event.is_some() {
        return Err(protocol_error(
            "Out-of-order or repeated desktop task event",
        ));
    }
    if snapshot.latest_event.is_none()
        && !matches!(
            event.payload,
            EventPayload::Started { .. }
                | EventPayload::Failed { .. }
                | EventPayload::Cancelled { .. }
        )
    {
        return Err(protocol_error("Desktop job has not started"));
    }
    let bind_run = |s: &mut TaskSnapshot, project, run| {
        if s.job != JobKind::Train
            || s.project_id != Some(project)
            || s.run_id.is_some_and(|known| known != run)
        {
            return Err(protocol_error(
                "Training event belongs to another project or run",
            ));
        }
        s.run_id = Some(run);
        Ok(())
    };
    match &event.payload {
        EventPayload::Started { job }
            if snapshot.job == *job && snapshot.latest_event.is_none() =>
        {
            if snapshot.status != TaskStatus::Cancelling {
                snapshot.status = TaskStatus::Running;
            }
        }
        EventPayload::Progress {
            stage,
            completed,
            total,
        } if total.is_none_or(|t| *completed <= t) => {
            snapshot.progress = Some(TaskProgress {
                stage: stage.clone(),
                completed: *completed,
                total: *total,
            });
        }
        EventPayload::Metrics(metrics) => {
            bind_run(snapshot, metrics.project_id, metrics.run_id)?;
            snapshot.metrics = Some(metrics.clone());
        }
        EventPayload::CheckpointCommitted { model_ref } => {
            bind_run(snapshot, model_ref.project_id, model_ref.run_id)?;
            snapshot.checkpoint = Some(model_ref.clone());
        }
        EventPayload::Completed { result } => {
            match result {
                ArtifactResult::Model(model) => {
                    bind_run(snapshot, model.project_id, model.run_id)?;
                    snapshot.checkpoint = Some(model.clone());
                }
                ArtifactResult::Video(_) if snapshot.job == JobKind::Generate => {}
                ArtifactResult::Resources(_) if snapshot.job == JobKind::PrepareResources => {}
                _ => return Err(protocol_error("Unexpected artifact for desktop task")),
            }
            snapshot.result = Some(result.clone());
            snapshot.status = TaskStatus::Completed;
        }
        EventPayload::Cancelled { checkpoint } => {
            if let Some(model) = checkpoint {
                bind_run(snapshot, model.project_id, model.run_id)?;
                snapshot.checkpoint = Some(model.clone());
            }
            snapshot.status = TaskStatus::Cancelled;
        }
        EventPayload::Failed { error } => {
            snapshot.status = TaskStatus::Failed;
            snapshot.error = Some(error.clone());
        }
        _ => return Err(protocol_error("Invalid event for the active desktop task")),
    }
    snapshot.last_seq = event.seq;
    snapshot.latest_event = Some(event.clone());
    Ok(())
}

fn is_mp4(path: &Path) -> bool {
    path.extension()
        .is_some_and(|s| s.eq_ignore_ascii_case("mp4"))
}

fn validate_local(target: &TargetPlatform) -> Result<()> {
    if target.os == std::env::consts::OS && target.architecture == std::env::consts::ARCH {
        return crate::resources::catalog::validate_target(target);
    }
    Err(invalid("Choose a resource target for this computer"))
}
fn busy(message: &str) -> AppError {
    AppError::new(ErrorCode::Busy, "task", message)
}
fn invalid(message: &str) -> AppError {
    AppError::new(ErrorCode::InvalidConfig, "desktop", message)
}
fn state_error(e: impl std::fmt::Display) -> AppError {
    AppError::new(ErrorCode::Internal, "desktop", e.to_string())
}
