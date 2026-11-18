use crate::worker_process::Containment;
use feathertalk_core::*;
use std::{
    collections::HashSet,
    io::{BufReader, Read, Write},
    path::Path,
    process::{Child, ChildStdin, Command, Stdio},
    sync::{Arc, Mutex, mpsc},
    thread::JoinHandle,
};
enum Incoming {
    Event(WorkerEvent),
    End(Option<AppError>),
}
pub struct WorkerHost {
    child: Child,
    input: Option<ChildStdin>,
    containment: Option<Containment>,
    incoming: mpsc::Receiver<Incoming>,
    readers: Vec<JoinHandle<()>>,
    stderr: Arc<Mutex<Vec<u8>>>,
    active: Option<TaskId>,
    used: HashSet<TaskId>,
    seq: Option<u64>,
    ready: bool,
    ended: bool,
    stage: String,
    checkpoint: Option<ModelRef>,
    started: bool,
    job: Option<JobKind>,
    project: Option<ProjectId>,
}
impl WorkerHost {
    pub fn spawn(component: &Path, context: WorkerContext) -> Result<Self> {
        if !component.is_absolute() {
            return Err(protocol_error("Worker component must be absolute"));
        }
        let mut command = Command::new(component);
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        // CUDA_PATH is scoped only to this owned child; never mutate the GUI environment.
        command.env_remove("CUDA_PATH");
        if let Some(root) = &context.cuda_root {
            let lib = root.join("nvvm/libdevice/libdevice.10.bc");
            let canonical = std::fs::canonicalize(&lib).map_err(protocol_error)?;
            if !context.components.iter().any(|c| {
                c.files
                    .iter()
                    .any(|f| std::fs::canonicalize(&f.path).ok().as_ref() == Some(&canonical))
            }) {
                return Err(protocol_error(
                    "CUDA libdevice is not in explicit inventory",
                ));
            }
            use sha2::{Digest, Sha256};
            let record = context
                .components
                .iter()
                .flat_map(|c| &c.files)
                .find(|f| std::fs::canonicalize(&f.path).ok().as_ref() == Some(&canonical))
                .unwrap();
            let actual = format!(
                "{:x}",
                Sha256::digest(std::fs::read(&canonical).map_err(protocol_error)?)
            );
            if actual != record.sha256 {
                return Err(AppError::new(
                    ErrorCode::ResourceCorrupt,
                    "worker_host",
                    "CUDA libdevice hash mismatch",
                ));
            }
            command.env("CUDA_PATH", root);
        }
        Containment::configure(&mut command);
        let mut child = command.spawn().map_err(protocol_error)?;
        let containment = Containment::assign(&mut child)?;
        let input = child.stdin.take();
        let stdout = child.stdout.take().unwrap();
        let mut stderr = child.stderr.take().unwrap();
        let (tx, rx) = mpsc::sync_channel(64);
        let log = Arc::new(Mutex::new(Vec::new()));
        let log_copy = log.clone();
        let output_reader = std::thread::spawn(move || {
            let mut reader = BufReader::new(stdout);
            loop {
                match read_frame(&mut reader) {
                    Ok(Some(bytes)) => match serde_json::from_slice(&bytes) {
                        Ok(e) => {
                            if tx.send(Incoming::Event(e)).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            let _ = tx.send(Incoming::End(Some(protocol_error(e))));
                            break;
                        }
                    },
                    Ok(None) => {
                        let _ = tx.send(Incoming::End(None));
                        break;
                    }
                    Err(e) => {
                        let _ = tx.send(Incoming::End(Some(e)));
                        break;
                    }
                }
            }
        });
        let log_reader = std::thread::spawn(move || {
            let mut buf = [0; 4096];
            while let Ok(n) = stderr.read(&mut buf) {
                if n == 0 {
                    break;
                }
                let mut tail = log_copy.lock().unwrap();
                tail.extend_from_slice(&buf[..n]);
                if tail.len() > 65536 {
                    let extra = tail.len() - 65536;
                    tail.drain(..extra);
                }
            }
        });
        let mut host = Self {
            child,
            input,
            containment: Some(containment),
            incoming: rx,
            readers: vec![output_reader, log_reader],
            stderr: log,
            active: None,
            used: HashSet::new(),
            seq: None,
            ready: false,
            ended: false,
            stage: "startup".into(),
            checkpoint: None,
            started: false,
            job: None,
            project: None,
        };
        host.send(&WorkerBootstrap {
            protocol_version: PROTOCOL_VERSION,
            context,
        })?;
        host.send(&WorkerRequest::Hello {
            protocol_version: PROTOCOL_VERSION,
        })?;
        Ok(host)
    }
    fn send(&mut self, value: &impl serde::Serialize) -> Result<()> {
        let bytes = serde_json::to_vec(value).map_err(protocol_error)?;
        if bytes.len() > MAX_MESSAGE_BYTES {
            return Err(protocol_error("Outbound message exceeds 1 MiB"));
        }
        let input = self
            .input
            .as_mut()
            .ok_or_else(|| protocol_error("Worker input closed"))?;
        input
            .write_all(&bytes)
            .and_then(|_| input.write_all(b"\n"))
            .and_then(|_| input.flush())
            .map_err(protocol_error)
    }
    pub fn start(&mut self, task_id: TaskId, job: JobSpec) -> Result<()> {
        if self.active.is_some() {
            return Err(AppError::new(
                ErrorCode::Busy,
                "worker_host",
                "A job is already running",
            ));
        }
        if !self.ready || self.ended {
            return Err(protocol_error("Worker is not ready"));
        }
        if self.used.contains(&task_id) {
            return Err(protocol_error("Task ID reused"));
        }
        let (kind, project) = match &job {
            JobSpec::Train(r) => (JobKind::Train, r.project_id),
            JobSpec::Generate(r) => (JobKind::Generate, r.project_id),
        };
        self.send(&WorkerRequest::Start { task_id, job })?;
        self.started = false;
        self.job = Some(kind);
        self.project = Some(project);
        self.active = Some(task_id);
        self.used.insert(task_id);
        self.stage = "starting".into();
        self.checkpoint = None;
        Ok(())
    }
    pub fn cancel(&mut self, task_id: TaskId) -> Result<()> {
        if self.active != Some(task_id) {
            return Err(protocol_error("Cancel does not own active task"));
        }
        self.send(&WorkerRequest::Cancel { task_id })
    }
    pub fn next_event(&mut self) -> Result<Option<WorkerEvent>> {
        if self.ended {
            return Ok(None);
        }
        let incoming = if self.ready {
            self.incoming.recv().map_err(|e| e.to_string())
        } else {
            self.incoming
                .recv_timeout(std::time::Duration::from_secs(300))
                .map_err(|e| e.to_string())
        };
        match incoming {
            Ok(Incoming::Event(event)) => {
                if event.seq == u64::MAX || self.seq.is_some_and(|s| event.seq <= s) {
                    return self.fail(Some(protocol_error("Nonmonotonic event sequence")));
                }
                self.seq = Some(event.seq);
                match &event.payload {
                    EventPayload::Ready { .. } => {
                        if self.ready || self.active.is_some() {
                            return self.fail(Some(protocol_error("Unexpected Ready")));
                        }
                        self.ready = true;
                    }
                    EventPayload::RequestRejected { error, .. } => {
                        // Control rejection never releases the task owner, even before Started.
                        return Err(error.clone());
                    }
                    _ => {
                        if event.task_id != self.active || self.active.is_none() {
                            return self.fail(Some(protocol_error("Foreign or late task event")));
                        }
                        if !self.started && !matches!(event.payload, EventPayload::Started { .. }) {
                            return self
                                .fail(Some(protocol_error("Task event arrived before Started")));
                        }
                        match &event.payload {
                            EventPayload::Started { job } => {
                                if self.started || self.job != Some(*job) {
                                    return self
                                        .fail(Some(protocol_error("Invalid Started event")));
                                }
                                self.started = true;
                            }
                            EventPayload::Progress {
                                completed, total, ..
                            } if total.is_some_and(|n| *completed > n) => {
                                return self.fail(Some(protocol_error("Invalid progress counts")));
                            }
                            EventPayload::Metrics(m) if Some(m.project_id) != self.project => {
                                return self.fail(Some(protocol_error("Foreign project metrics")));
                            }
                            EventPayload::CheckpointCommitted { model_ref }
                                if Some(model_ref.project_id) != self.project =>
                            {
                                return self
                                    .fail(Some(protocol_error("Foreign project checkpoint")));
                            }
                            EventPayload::Cancelled {
                                checkpoint: Some(r),
                            } if Some(r.project_id) != self.project => {
                                return self
                                    .fail(Some(protocol_error("Foreign cancelled checkpoint")));
                            }
                            EventPayload::Completed {
                                result: ArtifactResult::Model(r),
                            } if self.job != Some(JobKind::Train)
                                || Some(r.project_id) != self.project =>
                            {
                                return self
                                    .fail(Some(protocol_error("Unexpected completed model")));
                            }
                            EventPayload::Completed {
                                result: ArtifactResult::Video(_),
                            } if self.job != Some(JobKind::Generate) => {
                                return self
                                    .fail(Some(protocol_error("Unexpected completed video")));
                            }
                            EventPayload::Progress { stage, .. } => self.stage = stage.clone(),
                            EventPayload::CheckpointCommitted { model_ref } => {
                                self.checkpoint = Some(model_ref.clone())
                            }
                            EventPayload::Completed { .. }
                            | EventPayload::Cancelled { .. }
                            | EventPayload::Failed { .. } => self.active = None,
                            _ => {}
                        }
                    }
                }
                Ok(Some(event))
            }
            Ok(Incoming::End(error)) => self.fail(error),
            Err(e) => self.fail(Some(protocol_error(e))),
        }
    }
    fn fail(&mut self, cause: Option<AppError>) -> Result<Option<WorkerEvent>> {
        self.input.take();
        self.containment.take();
        let _ = self.child.kill();
        let status = self.child.wait().map_err(protocol_error)?;
        // Disconnect before joining: a reader may be blocked on a full queue.
        drop(std::mem::replace(&mut self.incoming, mpsc::channel().1));
        for reader in self.readers.drain(..) {
            let _ = reader.join();
        }
        self.ended = true;
        let protocol_failure = cause.is_some();
        let mut error = cause.unwrap_or_else(|| {
            AppError::new(
                ErrorCode::Internal,
                self.stage.clone(),
                "Worker exited before completing its job",
            )
        });
        error.evidence.push(format!(
            "exit={status}; last_stage={}; committed_checkpoint={:?}",
            self.stage, self.checkpoint
        ));
        error
            .evidence
            .push(String::from_utf8_lossy(&self.stderr.lock().unwrap()).into_owned());
        if let Some(task) = self.active.take() {
            let seq = self.seq.unwrap_or(0) + 1;
            self.seq = Some(seq);
            return WorkerEvent::new(Some(task), seq, EventPayload::Failed { error }).map(Some);
        }
        if protocol_failure || !self.ready || !status.success() {
            Err(error)
        } else {
            Ok(None)
        }
    }
    /// Forced termination reports only previously committed references via the subsequent failure event.
    pub fn terminate(&mut self) -> Result<()> {
        self.containment.take();
        self.child.kill().map_err(protocol_error)
    }
    pub fn shutdown(&mut self) -> Result<()> {
        self.send(&WorkerRequest::Shutdown)
    }
}
impl Drop for WorkerHost {
    fn drop(&mut self) {
        self.input.take();
        self.containment.take();
        let _ = self.child.kill();
        let _ = self.child.wait();
        drop(std::mem::replace(&mut self.incoming, mpsc::channel().1));
        for h in self.readers.drain(..) {
            let _ = h.join();
        }
    }
}
