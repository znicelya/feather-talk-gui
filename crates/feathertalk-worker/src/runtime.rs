use crate::context::validate_context;
use crate::outbox::terminal;
pub use crate::outbox::{EventOwner, event_channel};
use feathertalk_core::*;
use feathertalk_engine::{
    self as engine,
    media::MediaTools,
    training::{
        batch_probe::TrainingRuntime,
        lifecycle::{StoppingReason, TrainingProject, train_project},
    },
};
use std::{fs, thread::JoinHandle};
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
    prelude: WorkerPrelude,
    owner: EventOwner,
    hello: bool,
    current: Option<(TaskId, Cancellation, JoinHandle<()>)>,
    stopping: bool,
}
impl WorkerRuntime {
    pub fn new(context: WorkerContext, owner: EventOwner) -> Result<Self> {
        Self::from_prelude(
            WorkerPrelude::Job(WorkerBootstrap {
                protocol_version: PROTOCOL_VERSION,
                context,
            }),
            owner,
        )
    }

    pub fn from_prelude(prelude: WorkerPrelude, owner: EventOwner) -> Result<Self> {
        match &prelude {
            WorkerPrelude::Job(bootstrap) if bootstrap.protocol_version == PROTOCOL_VERSION => {
                validate_context(&bootstrap.context)?
            }
            WorkerPrelude::Job(_) => return Err(protocol_error("Bootstrap version mismatch")),
            WorkerPrelude::Discovery(discovery) => crate::context::validate_discovery(discovery)?,
        }
        Ok(Self {
            prelude,
            owner,
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
            // A published terminal means heavy computation has ended. Do not wait for
            // the old thread to be scheduled just to return from its final emit.
            if h.is_finished() {
                let _ = h.join();
            }
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
                let (target, components, runs_jobs) = match &self.prelude {
                    WorkerPrelude::Job(bootstrap) => (
                        &bootstrap.context.target,
                        &bootstrap.context.components,
                        true,
                    ),
                    WorkerPrelude::Discovery(discovery) => {
                        (&discovery.target, &discovery.components, false)
                    }
                };
                let devices = crate::context::devices(target)?;
                self.owner.control(EventPayload::Ready {
                    capabilities: WorkerCapabilities {
                        engine_version: env!("CARGO_PKG_VERSION").into(),
                        components: components
                            .iter()
                            .map(|c| ComponentCapability {
                                name: c.pin.id.clone(),
                                version: c.pin.version.clone(),
                                available: true,
                            })
                            .collect(),
                        training: runs_jobs,
                        generation: runs_jobs,
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
                let WorkerPrelude::Job(bootstrap) = &self.prelude else {
                    return Err(protocol_error("A discovery session does not run jobs"));
                };
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
                let context = bootstrap.context.clone();
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
        self.owner.close();
        self.finish();
    }
}
fn execute(
    c: &WorkerContext,
    job: &JobSpec,
    observer: &dyn TaskObserver,
    cancel: &Cancellation,
) -> Result<EventPayload> {
    let mut normalized = job.clone();
    if let JobSpec::Generate(request) = &mut normalized {
        crate::context::normalize_generation(request)?;
    }
    let job = &normalized;
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
        &crate::context::devices(&c.target)?,
    )?)?;
    let tools = MediaTools {
        ffmpeg: c.ffmpeg.clone(),
        ffprobe: c.ffprobe.clone(),
    };
    let reference = match job {
        JobSpec::Train(r) => r.resume.as_ref(),
        JobSpec::Generate(r) => Some(&r.model_ref),
    };
    let checkpoint_metadata = if let Some(reference) = reference {
        let metadata =
            engine::training::checkpoint::read_committed_metadata(&binding.directory, reference)?;
        for identity in metadata
            .preprocessing
            .component_identities
            .iter()
            .chain(metadata.vgg_identity.iter())
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
        Some(metadata)
    } else {
        None
    };
    match job {
        JobSpec::Train(request) => {
            let result = train_project(
                request,
                &TrainingProject {
                    directory: &binding.directory,
                    assets: &c.assets,
                    tools: &tools,
                },
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
        JobSpec::Generate(request) => {
            let metadata = checkpoint_metadata.expect("Generation always selects a checkpoint");
            let prepared = engine::data::prepared::reopen_generation(
                &binding.directory,
                &project,
                &metadata.preprocessing.cache_directory,
                &metadata.prepared_manifest_sha256,
            )?;
            engine::inference::generate(
                request, &prepared, &c.assets, &tools, &device, observer, cancel,
            )
            .map(|r| EventPayload::Completed {
                result: ArtifactResult::Video(r),
            })
        }
    }
}

#[cfg(test)]
mod backpressure_tests {
    use super::*;
    use std::time::Duration;
    fn fixture_context() -> WorkerPrelude {
        WorkerPrelude::Job(WorkerBootstrap {
            protocol_version: 1,
            context: WorkerContext {
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
            },
        })
    }
    #[test]
    fn stalled_output_keeps_cancel_responsive_and_terminal_lossless() {
        let (owner, rx) = event_channel();
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
        let context = fixture_context();
        let mut runtime = WorkerRuntime {
            prelude: context,
            owner,
            hello: true,
            current: Some((task, cancel, handle)),
            stopping: false,
        };
        // Give the producer a full queue, with no stdout consumer.
        assert!(completed.recv_timeout(Duration::from_millis(100)).is_err());
        let (control_done, control_rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            assert!(
                runtime
                    .handle(WorkerRequest::Cancel {
                        task_id: TaskId::new()
                    })
                    .is_err()
            );
            runtime
                .handle(WorkerRequest::Cancel { task_id: task })
                .unwrap();
            runtime.handle(WorkerRequest::Shutdown).unwrap();
            control_done.send(runtime).unwrap();
        });
        let mut runtime = control_rx
            .recv_timeout(Duration::from_secs(1))
            .expect("invalid request must not strand subsequent cancel/shutdown");
        let events = (0..131)
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
    #[test]
    fn terminal_allows_next_start_before_old_thread_returns() {
        let (owner, rx) = event_channel();
        let first = TaskId::new();
        owner.start(first, JobKind::Train).unwrap();
        let producer = owner.clone();
        let (release, gate) = std::sync::mpsc::channel();
        let thread = std::thread::spawn(move || {
            producer
                .emit(first, EventPayload::Cancelled { checkpoint: None })
                .unwrap();
            gate.recv().unwrap(); // Deliberately descheduled after terminal publication.
        });
        let mut runtime = WorkerRuntime {
            prelude: fixture_context(),
            owner,
            hello: true,
            current: Some((first, Cancellation::new(), thread)),
            stopping: false,
        };
        rx.recv_timeout(Duration::from_secs(1)).unwrap();
        assert!(matches!(
            rx.recv_timeout(Duration::from_secs(1)).unwrap().payload,
            EventPayload::Cancelled { .. }
        ));
        let (done, completed) = std::sync::mpsc::channel();
        let second = TaskId::new();
        std::thread::spawn(move || {
            let result = runtime.handle(WorkerRequest::Start {
                task_id: second,
                job: JobSpec::Train(TrainRequest {
                    project_id: ProjectId::new(),
                    batch_mode: BatchMode::Fixed,
                    config: TrainConfig::default(),
                    device: DevicePolicy::Auto,
                    resume: None,
                }),
            });
            done.send((runtime, result)).unwrap();
        });
        let result = completed.recv_timeout(Duration::from_secs(1));
        release.send(()).unwrap();
        let (mut runtime, accepted) =
            result.expect("next Start must not wait for old thread teardown");
        accepted.unwrap();
        assert_eq!(
            rx.recv_timeout(Duration::from_secs(1)).unwrap().task_id,
            Some(second)
        );
        runtime.finish();
    }
    #[test]
    fn forced_drop_wakes_blocked_producer_with_receiver_alive() {
        let (owner, _receiver) = event_channel();
        let task = TaskId::new();
        owner.start(task, JobKind::Train).unwrap();
        let producer = owner.clone();
        let thread = std::thread::spawn(move || {
            while producer
                .emit(
                    task,
                    EventPayload::Progress {
                        stage: "blocked".into(),
                        completed: 0,
                        total: None,
                    },
                )
                .is_ok()
            {}
        });
        let runtime = WorkerRuntime {
            prelude: fixture_context(),
            owner,
            hello: true,
            current: Some((task, Cancellation::new(), thread)),
            stopping: false,
        };
        let (done, completed) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            drop(runtime);
            done.send(()).unwrap();
        });
        completed
            .recv_timeout(Duration::from_secs(1))
            .expect("forced Drop must wake senders");
    }
}
