use feathertalk_core::*;
use serde::Serialize;
use std::sync::mpsc;
#[test]
fn second_heavy_job_rejected() {
    let (owner, rx) = feathertalk_worker::runtime::event_channel();
    let a = TaskId::new();
    owner.start(a, JobKind::Train).unwrap();
    assert_eq!(
        owner
            .start(TaskId::new(), JobKind::Generate)
            .unwrap_err()
            .code,
        ErrorCode::Busy
    );
    owner
        .emit(a, EventPayload::Cancelled { checkpoint: None })
        .unwrap();
    assert_eq!(rx.try_iter().count(), 2);
}
#[test]
fn stale_or_wrong_task_event_rejected() {
    let (owner, rx) = feathertalk_worker::runtime::event_channel();
    let a = TaskId::new();
    let b = TaskId::new();
    owner.start(a, JobKind::Train).unwrap();
    assert!(
        owner
            .emit(b, EventPayload::Cancelled { checkpoint: None })
            .is_err()
    );
    owner
        .emit(a, EventPayload::Cancelled { checkpoint: None })
        .unwrap();
    assert!(
        owner
            .emit(a, EventPayload::Cancelled { checkpoint: None })
            .is_err()
    );
    owner.start(b, JobKind::Generate).unwrap();
    assert!(
        owner
            .emit(a, EventPayload::Cancelled { checkpoint: None })
            .is_err()
    );
    let events = rx.try_iter().collect::<Vec<_>>();
    assert_eq!(events.len(), 3);
    assert!(events.windows(2).all(|w| w[0].seq < w[1].seq));
}

struct Process {
    child: std::process::Child,
    input: std::process::ChildStdin,
    events: std::sync::mpsc::Receiver<WorkerEvent>,
    _root: tempfile::TempDir,
    project: ProjectId,
    _paused_output: Option<std::process::ChildStdout>,
}
impl Drop for Process {
    fn drop(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
impl Process {
    fn new() -> Self {
        Self::with_context(|_| {})
    }
    fn with_context(change: impl FnOnce(&mut WorkerContext)) -> Self {
        Self::with_output(change, true)
    }
    fn with_output(change: impl FnOnce(&mut WorkerContext), drain: bool) -> Self {
        use sha2::{Digest, Sha256};
        use std::fs;
        let root = tempfile::tempdir().unwrap();
        let directory = root.path().canonicalize().unwrap();
        // A real source-hash stage provides a bounded window to exercise concurrent control.
        // The dummy resources are never neural-compute acceptance fixtures.
        let source = vec![0u8; 32 * 1024 * 1024];
        fs::write(directory.join("source.bin"), &source).unwrap();
        let project = ProjectId::new();
        let manifest = ProjectManifest {
            schema_version: 1,
            id: project,
            name: "control fixture".into(),
            source_video: SourceVideo {
                path: "source.bin".into(),
                sha256: format!("{:x}", Sha256::digest(&source)),
                size_bytes: source.len() as u64,
            },
            preprocessing: None,
            model_identity: None,
            checkpoints: vec![],
            metrics_path: "metrics.jsonl".into(),
        };
        fs::write(
            directory.join("project.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let file = directory.join("manifest.json");
        fs::write(&file, b"{}").unwrap();
        let hash = format!("{:x}", Sha256::digest(b"{}"));
        let mut context = WorkerContext {
            protocol_version: PROTOCOL_VERSION,
            projects: vec![ProjectBinding {
                project_id: project,
                directory: directory.clone(),
            }],
            target: TargetPlatform {
                os: std::env::consts::OS.into(),
                architecture: std::env::consts::ARCH.into(),
                backend: "cpu".into(),
            },
            components: vec![RuntimeComponent {
                target: Some(TargetPlatform {
                    os: std::env::consts::OS.into(),
                    architecture: std::env::consts::ARCH.into(),
                    backend: "cpu".into(),
                }),
                pin: ResourcePin {
                    id: "control-fixture".into(),
                    version: "1".into(),
                },
                root: directory.clone(),
                files: vec![RuntimeFile {
                    path: file.clone(),
                    sha256: hash,
                }],
            }],
            assets: directory,
            ffmpeg: file.clone(),
            ffprobe: file.clone(),
            probe_runner: file,
            cuda_root: None,
        };
        change(&mut context);
        Self::spawn(
            root,
            project,
            WorkerPrelude::Job(WorkerBootstrap {
                protocol_version: PROTOCOL_VERSION,
                context,
            }),
            drain,
        )
    }

    fn spawn(
        root: tempfile::TempDir,
        project: ProjectId,
        prelude: WorkerPrelude,
        drain: bool,
    ) -> Self {
        use std::{
            io::BufReader,
            process::{Command, Stdio},
        };
        let mut child = Command::new(env!("CARGO_BIN_EXE_feathertalk-worker"))
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap();
        let input = child.stdin.take().unwrap();
        let output = child.stdout.take().unwrap();
        let (tx, rx) = mpsc::channel();
        let paused = if drain {
            std::thread::spawn(move || {
                let mut reader = BufReader::new(output);
                while let Ok(Some(bytes)) = read_frame(&mut reader) {
                    let event = serde_json::from_slice(&bytes).unwrap();
                    if tx.send(event).is_err() {
                        break;
                    }
                }
            });
            None
        } else {
            Some(output)
        };
        let mut p = Self {
            child,
            input,
            events: rx,
            _root: root,
            project,
            _paused_output: paused,
        };
        p.write(&prelude);
        p
    }
    fn write(&mut self, value: &impl Serialize) {
        use std::io::Write;
        self.input
            .write_all(&serde_json::to_vec(value).unwrap())
            .unwrap();
        self.input.write_all(b"\n").unwrap();
        self.input.flush().unwrap();
    }
    fn next(&self) -> WorkerEvent {
        self.events
            .recv_timeout(std::time::Duration::from_secs(30))
            .expect("bounded protocol response")
    }
    fn train(&self) -> JobSpec {
        JobSpec::Train(TrainRequest {
            project_id: self.project,
            batch_mode: BatchMode::Fixed,
            config: TrainConfig {
                batch_size: 1,
                ..Default::default()
            },
            device: DevicePolicy::Auto,
            resume: None,
        })
    }
}

#[test]
fn discovery_without_project_reports_devices_but_rejects_start() {
    use sha2::{Digest, Sha256};
    use std::{fs, path::Path};

    let scratch = tempfile::tempdir().unwrap();
    let worker = Path::new(env!("CARGO_BIN_EXE_feathertalk-worker"))
        .canonicalize()
        .unwrap();
    let mut file = fs::File::open(&worker).unwrap();
    let mut hash = Sha256::new();
    std::io::copy(&mut file, &mut hash).unwrap();
    let target = TargetPlatform {
        os: std::env::consts::OS.into(),
        architecture: std::env::consts::ARCH.into(),
        backend: "cpu".into(),
    };
    let discovery = WorkerPrelude::Discovery(WorkerDiscovery {
        protocol_version: PROTOCOL_VERSION,
        target: target.clone(),
        cuda_root: None,
        components: vec![RuntimeComponent {
            target: Some(target),
            pin: ResourcePin {
                id: "worker".into(),
                version: "fixture".into(),
            },
            root: worker.parent().unwrap().into(),
            files: vec![RuntimeFile {
                path: worker,
                sha256: format!("{:x}", hash.finalize()),
            }],
        }],
    });
    let mut process = Process::spawn(scratch, ProjectId::new(), discovery, true);
    process.write(&WorkerRequest::Hello {
        protocol_version: PROTOCOL_VERSION,
    });
    let ready = process.next();
    assert_eq!(ready.task_id, None);
    let EventPayload::Ready {
        capabilities,
        devices,
    } = ready.payload
    else {
        panic!("Expected discovery Ready: {ready:?}")
    };
    assert!(!capabilities.training && !capabilities.generation);
    assert!(
        devices
            .iter()
            .any(|device| device.backend == BackendKind::Flex)
    );

    let task = TaskId::new();
    process.write(&WorkerRequest::Start {
        task_id: task,
        job: process.train(),
    });
    let event = process.next();
    assert_eq!(event.task_id, None);
    assert!(matches!(event.payload, EventPayload::RequestRejected {
        request: RequestKind::Start, requested_task_id: Some(id), error,
    } if id == task && error.code == ErrorCode::Protocol));
    process.write(&WorkerRequest::Shutdown);
}

#[test]
fn raw_hello_start_cancel_roundtrip() {
    let mut p = Process::new();
    p.write(&WorkerRequest::Hello {
        protocol_version: 99,
    });
    let bad = p.next();
    assert!(matches!(
        bad.payload,
        EventPayload::RequestRejected {
            request: RequestKind::Hello,
            ..
        }
    ));
    assert!(bad.task_id.is_none());
    p.write(&WorkerRequest::Hello {
        protocol_version: 1,
    });
    let ready = p.next();
    assert!(matches!(ready.payload, EventPayload::Ready { .. }));
    p.write(&WorkerRequest::Hello {
        protocol_version: 1,
    });
    assert!(matches!(
        p.next().payload,
        EventPayload::RequestRejected {
            request: RequestKind::Hello,
            ..
        }
    ));
    let first = TaskId::new();
    p.write(&WorkerRequest::Start {
        task_id: first,
        job: p.train(),
    });
    assert!(matches!(p.next().payload, EventPayload::Started { .. }));
    let second = TaskId::new();
    p.write(&WorkerRequest::Start {
        task_id: second,
        job: p.train(),
    });
    p.write(&WorkerRequest::Cancel { task_id: second });
    p.write(&WorkerRequest::Cancel { task_id: first });
    let (mut busy, mut stale, mut terminal, mut seq) = (false, false, 0, ready.seq);
    loop {
        let e = p.next();
        assert!(e.seq > seq);
        seq = e.seq;
        match e.payload {
            EventPayload::RequestRejected {
                request: RequestKind::Start,
                requested_task_id,
                error,
            } => {
                assert_eq!(error.code, ErrorCode::Busy);
                assert_eq!(requested_task_id, Some(second));
                busy = true;
            }
            EventPayload::RequestRejected {
                request: RequestKind::Cancel,
                requested_task_id,
                error,
            } => {
                assert_eq!(error.code, ErrorCode::Protocol);
                assert_eq!(requested_task_id, Some(second));
                stale = true;
            }
            EventPayload::Cancelled { .. } => {
                assert_eq!(e.task_id, Some(first));
                terminal += 1;
                break;
            }
            EventPayload::Failed { error } => panic!("Unexpected computation outcome: {error:?}"),
            _ => {}
        }
    }
    assert!(busy && stale);
    assert_eq!(terminal, 1);
    p.write(&WorkerRequest::Cancel { task_id: first });
    assert!(matches!(
        p.next().payload,
        EventPayload::RequestRejected {
            request: RequestKind::Cancel,
            ..
        }
    ));
    p.write(&WorkerRequest::Shutdown);
    let started = std::time::Instant::now();
    loop {
        if let Some(status) = p.child.try_wait().unwrap() {
            assert!(status.success());
            break;
        }
        assert!(started.elapsed().as_secs() < 10);
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(p.events.try_iter().all(|e| !matches!(
        e.payload,
        EventPayload::Completed { .. }
            | EventPayload::Cancelled { .. }
            | EventPayload::Failed { .. }
    )));
}

#[test]
fn fragmented_input_and_oversized_unterminated_record() {
    use std::io::Write;
    let mut p = Process::new();
    let request = serde_json::to_vec(&WorkerRequest::Hello {
        protocol_version: 1,
    })
    .unwrap();
    for chunk in request.chunks(2) {
        p.input.write_all(chunk).unwrap();
        p.input.flush().unwrap();
    }
    p.input.write_all(b"\n").unwrap();
    p.input.flush().unwrap();
    assert!(matches!(p.next().payload, EventPayload::Ready { .. }));
    let oversized = vec![b'x'; MAX_MESSAGE_BYTES + 1];
    let _ = p.input.write_all(&oversized);
    let _ = p.input.flush();
    let start = std::time::Instant::now();
    loop {
        if let Some(status) = p.child.try_wait().unwrap() {
            assert!(!status.success());
            break;
        }
        assert!(start.elapsed().as_secs() < 10);
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    assert!(
        p.events
            .recv_timeout(std::time::Duration::from_secs(1))
            .is_err()
    );
}

#[test]
fn bootstrap_rejects_wrong_identity_target_hash_and_unbound_probe() {
    for case in 0..4 {
        let mut p = Process::with_context(|c| match case {
            0 => c.projects[0].project_id = ProjectId::new(),
            1 => c.target.architecture = "wrong".into(),
            2 => c.components[0].files[0].sha256 = "0".repeat(64),
            _ => c.components[0].target = None,
        });
        assert!(
            p.events
                .recv_timeout(std::time::Duration::from_secs(10))
                .is_err()
        );
        assert!(!p.child.wait().unwrap().success());
    }
}

#[test]
fn relative_generation_paths_are_owned_failures() {
    for relative_audio in [true, false] {
        let mut p = Process::new();
        p.write(&WorkerRequest::Hello {
            protocol_version: 1,
        });
        p.next();
        let task = TaskId::new();
        let job = JobSpec::Generate(GenerateRequest {
            project_id: p.project,
            audio_path: if relative_audio { "relative.wav".into() } else { p._root.path().join("source.bin") },
            output_path: if relative_audio { p._root.path().join("out.mp4") } else { "relative.mp4".into() },
            model_ref: serde_json::from_value(serde_json::json!({"project_id":p.project,"run_id":TrainingRunId::new(),"checkpoint_id":"fixture"})).unwrap(),
            device: DevicePolicy::Auto,
        });
        p.write(&WorkerRequest::Start { task_id: task, job });
        assert!(matches!(p.next().payload, EventPayload::Started { .. }));
        let EventPayload::Failed { error } = p.next().payload else {
            panic!("expected path failure")
        };
        assert!(error.message.contains("absolute"), "{error:?}");
    }
}

#[test]
fn stalled_output_control_overload_exits_with_idle_stdin() {
    use std::io::Write;
    let mut p = Process::with_output(|_| {}, false);
    // Never drain stdout. Repeated rejected controls exhaust the bounded reserve.
    let records = b"{\"type\":\"hello\",\"protocol_version\":99}\n".repeat(1024);
    let _ = p.input.write_all(&records);
    let _ = p.input.flush();
    let start = std::time::Instant::now();
    loop {
        if let Some(status) = p.child.try_wait().unwrap() {
            assert!(!status.success());
            break;
        }
        assert!(
            start.elapsed() < std::time::Duration::from_secs(10),
            "control overload must exit without another record"
        );
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
}
