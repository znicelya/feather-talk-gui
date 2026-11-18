use crate::context::validate_context;
use feathertalk_core::*;
use feathertalk_engine::{
    self as engine,
    media::MediaTools,
    training::{
        batch_probe::TrainingRuntime,
        lifecycle::{StoppingReason, train},
    },
};
use std::{
    collections::HashSet,
    fs,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc::SyncSender,
    },
    thread::JoinHandle,
    time::{Duration, Instant},
};
struct State {
    seq: u64,
    active: Option<TaskId>,
    used: HashSet<TaskId>,
    output: SyncSender<WorkerEvent>,
    metrics: Option<Instant>,
}
#[derive(Clone)]
pub struct EventOwner(Arc<Mutex<State>>, Arc<AtomicBool>);
fn terminal(p: &EventPayload) -> bool {
    matches!(
        p,
        EventPayload::Completed { .. }
            | EventPayload::Cancelled { .. }
            | EventPayload::Failed { .. }
    )
}
impl State {
    fn send(&mut self, task: Option<TaskId>, payload: EventPayload) -> Result<()> {
        let event = WorkerEvent::new(task, self.seq, payload)?;
        self.seq = self
            .seq
            .checked_add(1)
            .ok_or_else(|| protocol_error("Sequence exhausted"))?;
        self.output.send(event).map_err(protocol_error)
    }
}
impl EventOwner {
    pub fn new(output: SyncSender<WorkerEvent>) -> Self {
        Self(
            Arc::new(Mutex::new(State {
                seq: 0,
                active: None,
                used: HashSet::new(),
                output,
                metrics: None,
            })),
            Arc::new(AtomicBool::new(false)),
        )
    }
    pub fn start(&self, task: TaskId, kind: JobKind) -> Result<()> {
        let mut s = self.0.lock().map_err(protocol_error)?;
        if s.active.is_some() {
            return Err(AppError::new(
                ErrorCode::Busy,
                "worker",
                "A job is already running",
            ));
        }
        if !s.used.insert(task) {
            return Err(protocol_error("Task ID reused"));
        }
        s.active = Some(task);
        self.1.store(true, Ordering::Release);
        s.metrics = None;
        s.send(Some(task), EventPayload::Started { job: kind })
    }
    pub fn emit(&self, task: TaskId, payload: EventPayload) -> Result<()> {
        let mut s = self.0.lock().map_err(protocol_error)?;
        if s.active != Some(task) {
            return Err(protocol_error("Late or foreign task event"));
        }
        if matches!(
            payload,
            EventPayload::Ready { .. }
                | EventPayload::RequestRejected { .. }
                | EventPayload::Started { .. }
        ) {
            return Err(protocol_error("Invalid task callback"));
        }
        if matches!(payload, EventPayload::Metrics(_)) {
            if s.metrics
                .is_some_and(|t| t.elapsed() < Duration::from_millis(500))
            {
                return Ok(());
            }
            s.metrics = Some(Instant::now());
        }
        let final_event = terminal(&payload);
        s.send(Some(task), payload)?;
        if final_event {
            s.active = None;
            self.1.store(false, Ordering::Release);
        }
        Ok(())
    }
    fn active(&self) -> bool {
        self.1.load(Ordering::Acquire)
    }
    fn control(&self, p: EventPayload) -> Result<()> {
        self.0.lock().map_err(protocol_error)?.send(None, p)
    }
}
struct Observe {
    owner: EventOwner,
    task: TaskId,
}
impl TaskObserver for Observe {
    fn emit(&self, p: EventPayload) -> Result<()> {
        if terminal(&p) {
            return Err(protocol_error("Engine must not emit terminal events"));
        }
        self.owner.emit(self.task, p)
    }
}
pub struct WorkerRuntime {
    context: WorkerContext,
    owner: EventOwner,
    hello: bool,
    current: Option<(TaskId, Cancellation, JoinHandle<()>)>,
    stopping: bool,
}
impl WorkerRuntime {
    pub fn new(context: WorkerContext, output: SyncSender<WorkerEvent>) -> Result<Self> {
        validate_context(&context)?;
        Ok(Self {
            context,
            owner: EventOwner::new(output),
            hello: false,
            current: None,
            stopping: false,
        })
    }
    pub fn handle(&mut self, request: WorkerRequest) -> Result<()> {
        if self
            .current
            .as_ref()
            .is_some_and(|(_, _, h)| h.is_finished() || !self.owner.active())
        {
            let (_, _, h) = self.current.take().unwrap();
            let _ = h.join();
        }
        let (kind, id) = match &request {
            WorkerRequest::Hello { .. } => (RequestKind::Hello, None),
            WorkerRequest::Start { task_id, .. } => (RequestKind::Start, Some(*task_id)),
            WorkerRequest::Cancel { task_id } => (RequestKind::Cancel, Some(*task_id)),
            WorkerRequest::Shutdown => (RequestKind::Shutdown, None),
        };
        let result = self.accept(request);
        if let Err(error) = &result {
            self.owner.control(EventPayload::RequestRejected {
                request: kind,
                requested_task_id: id,
                error: error.clone(),
            })?;
        }
        result
    }
    fn accept(&mut self, request: WorkerRequest) -> Result<()> {
        if self.stopping {
            return Err(protocol_error("Worker is shutting down"));
        }
        match request {
            WorkerRequest::Hello { protocol_version } => {
                if self.hello || protocol_version != PROTOCOL_VERSION {
                    return Err(protocol_error("Invalid or duplicate Hello"));
                }
                let devices = crate::context::devices(&self.context)?;
                self.owner.control(EventPayload::Ready {
                    capabilities: WorkerCapabilities {
                        engine_version: env!("CARGO_PKG_VERSION").into(),
                        components: self
                            .context
                            .components
                            .iter()
                            .map(|c| ComponentCapability {
                                name: c.pin.id.clone(),
                                version: c.pin.version.clone(),
                                available: true,
                            })
                            .collect(),
                        training: true,
                        generation: true,
                    },
                    devices,
                })?;
                self.hello = true;
                Ok(())
            }
            WorkerRequest::Start { task_id, job } => {
                if !self.hello {
                    return Err(protocol_error("Hello required before Start"));
                }
                if self.current.is_some() {
                    return Err(AppError::new(
                        ErrorCode::Busy,
                        "worker",
                        "A job is still finishing",
                    ));
                }
                let kind = match &job {
                    JobSpec::Train(_) => JobKind::Train,
                    JobSpec::Generate(_) => JobKind::Generate,
                };
                self.owner.start(task_id, kind)?;
                let cancel = Cancellation::new();
                let flag = cancel.clone();
                let context = self.context.clone();
                let owner = self.owner.clone();
                let handle = std::thread::spawn(move || {
                    let observer = Observe {
                        owner: owner.clone(),
                        task: task_id,
                    };
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        execute(&context, &job, &observer, &flag)
                    }));
                    let payload = match result {
                        Ok(Ok(p)) => p,
                        Ok(Err(e)) if e.code == ErrorCode::Cancelled => {
                            EventPayload::Cancelled { checkpoint: None }
                        }
                        Ok(Err(error)) => EventPayload::Failed { error },
                        Err(_) => EventPayload::Failed {
                            error: AppError::new(
                                ErrorCode::Internal,
                                "compute",
                                "Worker computation panicked; see stderr",
                            ),
                        },
                    };
                    if let Err(e) = owner.emit(task_id, payload) {
                        eprintln!("{e}");
                    }
                });
                self.current = Some((task_id, cancel, handle));
                Ok(())
            }
            WorkerRequest::Cancel { task_id } => match &self.current {
                Some((id, cancel, _)) if *id == task_id && self.owner.active() => {
                    cancel.cancel();
                    Ok(())
                }
                _ => Err(protocol_error("Cancel does not own the active task")),
            },
            WorkerRequest::Shutdown => {
                self.stopping = true;
                if let Some((_, c, _)) = &self.current {
                    c.cancel();
                }
                Ok(())
            }
        }
    }
    pub fn stopping(&self) -> bool {
        self.stopping
    }
    pub fn finish(&mut self) {
        if let Some((_, cancel, handle)) = self.current.take() {
            cancel.cancel();
            let _ = handle.join();
        }
    }
}
impl Drop for WorkerRuntime {
    fn drop(&mut self) {
        self.finish();
    }
}
fn execute(
    c: &WorkerContext,
    job: &JobSpec,
    observer: &dyn TaskObserver,
    cancel: &Cancellation,
) -> Result<EventPayload> {
    let (id, policy) = match job {
        JobSpec::Train(r) => (r.project_id, &r.device),
        JobSpec::Generate(r) => (r.project_id, &r.device),
    };
    let binding = c
        .projects
        .iter()
        .find(|b| b.project_id == id)
        .ok_or_else(|| protocol_error("Project not bound to worker"))?;
    let project: ProjectManifest = serde_json::from_slice(
        &fs::read(binding.directory.join("project.json")).map_err(protocol_error)?,
    )
    .map_err(protocol_error)?;
    if project.id != id {
        return Err(protocol_error("Project binding mismatch"));
    }
    let device = engine::open_device(&engine::resolve_device(
        policy,
        &crate::context::devices(c)?,
    )?)?;
    let tools = MediaTools {
        ffmpeg: c.ffmpeg.clone(),
        ffprobe: c.ffprobe.clone(),
    };
    let reference = match job {
        JobSpec::Train(r) => r.resume.as_ref(),
        JobSpec::Generate(r) => Some(&r.model_ref),
    };
    let prepared = if let Some(reference) = reference {
        let state =
            engine::training::checkpoint::load_checkpoint(&binding.directory, reference, &device)?;
        for identity in state
            .metadata
            .preprocessing
            .component_identities
            .iter()
            .chain(state.metadata.vgg_identity.iter())
        {
            let manifest: AssetManifest = serde_json::from_slice(
                &fs::read(c.assets.join(&identity.kind).join("manifest.json"))
                    .map_err(protocol_error)?,
            )
            .map_err(protocol_error)?;
            if manifest.identity != *identity {
                return Err(AppError::new(
                    ErrorCode::ModelMismatch,
                    "context",
                    "Resolve the checkpoint's original resource bundle before resuming or generating",
                ));
            }
        }
        engine::data::prepared::reopen_generation(
            &binding.directory,
            &project,
            &state.metadata.preprocessing.cache_directory,
            &state.metadata.prepared_manifest_sha256,
        )?
    } else {
        engine::data::prepare_project(
            &binding.directory,
            &project,
            &c.assets,
            &tools,
            &device,
            observer,
            cancel,
        )?
    };
    match job {
        JobSpec::Train(request) => {
            let result = train(
                request,
                &prepared,
                &c.assets.join("vgg19"),
                &device,
                &TrainingRuntime {
                    probe_runner: c.probe_runner.clone(),
                },
                observer,
                cancel,
            )?;
            Ok(match result.stopping_reason {
                StoppingReason::Completed => EventPayload::Completed {
                    result: ArtifactResult::Model(result.committed_model),
                },
                StoppingReason::Cancelled => EventPayload::Cancelled {
                    checkpoint: Some(result.committed_model),
                },
            })
        }
        JobSpec::Generate(request) => engine::inference::generate(
            request, &prepared, &c.assets, &tools, &device, observer, cancel,
        )
        .map(|r| EventPayload::Completed {
            result: ArtifactResult::Video(r),
        }),
    }
}

/// Bounded worker output queue; shared by the binary and focused scheduling tests.
pub fn event_channel() -> (
    std::sync::mpsc::SyncSender<WorkerEvent>,
    std::sync::mpsc::Receiver<WorkerEvent>,
) {
    std::sync::mpsc::sync_channel(64)
}

#[cfg(test)]
mod backpressure_tests {
    use super::*;
    #[test]
    fn stalled_output_keeps_cancel_responsive_and_terminal_lossless() {
        let (tx, rx) = event_channel();
        let owner = EventOwner::new(tx);
        let task = TaskId::new();
        owner.start(task, JobKind::Train).unwrap();
        let cancel = Cancellation::new();
        let flag = cancel.clone();
        let producer = owner.clone();
        let (done, completed) = std::sync::mpsc::channel();
        let handle = std::thread::spawn(move || {
            for i in 0..128 {
                producer
                    .emit(
                        task,
                        EventPayload::Progress {
                            stage: "blocked".into(),
                            completed: i,
                            total: None,
                        },
                    )
                    .unwrap();
            }
            assert!(flag.is_cancelled());
            producer
                .emit(task, EventPayload::Cancelled { checkpoint: None })
                .unwrap();
            done.send(()).unwrap();
        });
        let context = WorkerContext {
            protocol_version: 1,
            projects: vec![],
            components: vec![],
            target: TargetPlatform {
                os: "test".into(),
                architecture: "test".into(),
                backend: "cpu".into(),
            },
            assets: ".".into(),
            ffmpeg: ".".into(),
            ffprobe: ".".into(),
            probe_runner: ".".into(),
            cuda_root: None,
        };
        let mut runtime = WorkerRuntime {
            context,
            owner,
            hello: true,
            current: Some((task, cancel, handle)),
            stopping: false,
        };
        // Give the producer a full queue, with no stdout consumer.
        assert!(completed.recv_timeout(Duration::from_millis(100)).is_err());
        runtime
            .handle(WorkerRequest::Cancel { task_id: task })
            .unwrap();
        let events = (0..130)
            .map(|_| rx.recv_timeout(Duration::from_secs(2)).unwrap())
            .collect::<Vec<_>>();
        completed.recv_timeout(Duration::from_secs(2)).unwrap();
        assert!(events.windows(2).all(|w| w[0].seq < w[1].seq));
        assert_eq!(events.iter().filter(|e| terminal(&e.payload)).count(), 1);
        assert!(matches!(
            events.last().unwrap().payload,
            EventPayload::Cancelled { .. }
        ));
        runtime.finish();
    }
}
