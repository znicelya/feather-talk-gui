use feather_talk_gui_lib::worker_host::{HostPoll, WorkerHost};
use std::time::Duration;
fn next(host: &mut WorkerHost) -> Result<Option<WorkerEvent>> {
    match host.next_event(Duration::from_secs(10))? {
        HostPoll::Event(event) => Ok(Some(event)),
        HostPoll::Ended => Ok(None),
        HostPoll::Timeout => panic!("unexpected fixture timeout"),
    }
}
use feathertalk_core::*;
/// Actual worker/context supplied by the recorded local resource integration run.
#[test]
#[ignore = "requires explicitly installed local context"]
fn hello_and_crash_preserves_gui_error_context() {
    let Some(context) = std::env::var_os("FT_WORKER_CONTEXT") else {
        eprintln!("SKIP: explicit local resource context not supplied");
        return;
    };
    let context: WorkerContext = serde_json::from_slice(&std::fs::read(context).unwrap()).unwrap();
    let binary =
        std::path::PathBuf::from(std::env::var_os("FT_WORKER_BINARY").expect("worker binary"));
    let project = context.projects[0].project_id;
    let mut host = WorkerHost::spawn(&binary, context).unwrap();
    assert!(matches!(
        next(&mut host).unwrap().unwrap().payload,
        EventPayload::Ready { .. }
    ));
    let task = TaskId::new();
    host.start(
        task,
        JobSpec::Train(TrainRequest {
            batch_mode: BatchMode::Fixed,
            project_id: project,
            config: TrainConfig::default(),
            device: DevicePolicy::Auto,
            resume: None,
        }),
    )
    .unwrap();
    assert_eq!(
        host.start(
            TaskId::new(),
            JobSpec::Train(TrainRequest {
                batch_mode: BatchMode::Fixed,
                project_id: project,
                config: TrainConfig::default(),
                device: DevicePolicy::Auto,
                resume: None
            })
        )
        .unwrap_err()
        .code,
        ErrorCode::Busy
    );
    assert!(matches!(
        next(&mut host).unwrap().unwrap().payload,
        EventPayload::Started { .. }
    ));
    let _ = host.terminate();
    let mut terminal = 0;
    while let Some(event) = next(&mut host).unwrap() {
        if let EventPayload::Failed { error } = event.payload {
            terminal += 1;
            assert_eq!(event.task_id, Some(task));
            assert!(
                error
                    .evidence
                    .iter()
                    .any(|x| x.contains("exit=") && x.contains("last_stage="))
            );
            assert_ne!(error.code, ErrorCode::OutOfMemory);
        }
    }
    assert_eq!(terminal, 1);
}

#[test]
fn crash_stderr_control_error_and_late_events() {
    fn assert_send<T: Send>() {}
    assert_send::<WorkerHost>();
    use std::{fs, path::PathBuf, process::Command};
    let root = tempfile::tempdir().unwrap();
    let binary = root.path().join(if cfg!(windows) {
        "fixture.exe"
    } else {
        "fixture"
    });
    let rustc = std::env::var_os("RUSTC").unwrap_or_else(|| {
        if cfg!(windows) {
            "C:/Users/Administrator/.cargo/bin/rustc.exe".into()
        } else {
            "rustc".into()
        }
    });
    let result = Command::new(rustc)
        .arg(
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/worker_host_fixture.rs"),
        )
        .arg("-o")
        .arg(&binary)
        .output()
        .unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    for mode in [
        "crash",
        "malformed",
        "oversized",
        "wrong-seq",
        "foreign",
        "control-error",
        "pre-start-control",
        "flood-drop",
        "quiet-cancel",
        "gen-metrics",
        "gen-checkpoint",
        "gen-cancel-checkpoint",
        "gen-wrong-output",
        "gen-valid-output",
    ] {
        let project = ProjectId::new();
        let path = root.path().join(mode);
        fs::create_dir(&path).unwrap();
        let context = WorkerContext {
            protocol_version: 1,
            projects: vec![ProjectBinding {
                project_id: project,
                directory: path,
            }],
            target: TargetPlatform {
                os: std::env::consts::OS.into(),
                architecture: std::env::consts::ARCH.into(),
                backend: "cpu".into(),
            },
            components: vec![],
            assets: root.path().into(),
            ffmpeg: binary.clone(),
            ffprobe: binary.clone(),
            probe_runner: binary.clone(),
            cuda_root: None,
        };
        let model = ModelRef {
            project_id: project,
            run_id: TrainingRunId::new(),
            checkpoint_id: "fixture".into(),
        };
        let output = root.path().canonicalize().unwrap().join("chosen.mp4");
        let device = DeviceDescriptor {
            id: "cpu".into(),
            name: "fixture".into(),
            backend: BackendKind::Cpu,
            vendor: None,
            driver: None,
            memory_bytes: None,
        };
        let payload = match mode {
            "gen-metrics" => Some(EventPayload::Metrics(TrainingMetrics {
                project_id: project,
                run_id: model.run_id,
                step: 1,
                epoch: 1,
                elapsed_ms: 1,
                lr: 0.001,
                losses: LossScalars {
                    full: 0.,
                    mouth: 0.,
                    temporal: 0.,
                    temporal_mouth: 0.,
                    perceptual: 0.,
                    mouth_gradient: 0.,
                    total: 0.,
                },
                device: device.clone(),
                precision: Precision::F32,
                batch_size: 1,
            })),
            "gen-checkpoint" => Some(EventPayload::CheckpointCommitted {
                model_ref: model.clone(),
            }),
            "gen-cancel-checkpoint" => Some(EventPayload::Cancelled {
                checkpoint: Some(model.clone()),
            }),
            "gen-wrong-output" | "gen-valid-output" => Some(EventPayload::Completed {
                result: ArtifactResult::Video(ExportResult {
                    output_path: if mode == "gen-valid-output" {
                        output.clone()
                    } else {
                        root.path().join("unrelated.mp4")
                    },
                    video_frames: 1,
                    elapsed_ms: 1,
                    device,
                }),
            }),
            _ => None,
        };
        if let Some(payload) = payload {
            fs::write(
                root.path().join(format!("{mode}.json")),
                serde_json::to_vec(&payload).unwrap(),
            )
            .unwrap();
        }
        let mut host = WorkerHost::spawn(&binary.canonicalize().unwrap(), context).unwrap();
        assert!(matches!(
            next(&mut host).unwrap().unwrap().payload,
            EventPayload::Ready { .. }
        ));
        let task = TaskId::new();
        let job = if mode.starts_with("gen-") {
            JobSpec::Generate(GenerateRequest {
                project_id: project,
                audio_path: binary.clone(),
                output_path: output,
                model_ref: model,
                device: DevicePolicy::Auto,
            })
        } else {
            JobSpec::Train(TrainRequest {
                project_id: project,
                batch_mode: BatchMode::Fixed,
                config: TrainConfig::default(),
                device: DevicePolicy::Auto,
                resume: None,
            })
        };
        host.start(task, job).unwrap();
        if mode == "pre-start-control" {
            assert_eq!(next(&mut host).unwrap_err().code, ErrorCode::Busy);
        }
        assert!(matches!(
            next(&mut host).unwrap().unwrap().payload,
            EventPayload::Started { .. }
        ));
        assert!(matches!(
            next(&mut host).unwrap().unwrap().payload,
            EventPayload::Progress { .. }
        ));
        if mode.starts_with("gen-") {
            let event = next(&mut host).unwrap().unwrap();
            if mode == "gen-valid-output" {
                assert!(matches!(event.payload, EventPayload::Completed { .. }));
            } else {
                let EventPayload::Failed { error } = event.payload else {
                    panic!("generation accepted incompatible payload")
                };
                assert_eq!(error.code, ErrorCode::Protocol);
                assert!(
                    error
                        .evidence
                        .iter()
                        .any(|e| e.contains("committed_checkpoint=None"))
                );
            }
            continue;
        }
        if mode == "quiet-cancel" {
            assert!(matches!(
                host.next_event(Duration::from_millis(30)).unwrap(),
                HostPoll::Timeout
            ));
            host.cancel(task).unwrap();
            assert!(matches!(
                next(&mut host).unwrap().unwrap().payload,
                EventPayload::Cancelled { .. }
            ));
            continue;
        }
        if mode == "flood-drop" {
            std::thread::sleep(std::time::Duration::from_millis(100));
            let (done, finished) = std::sync::mpsc::channel();
            std::thread::spawn(move || {
                drop(host);
                done.send(()).unwrap();
            });
            finished
                .recv_timeout(std::time::Duration::from_secs(5))
                .expect("full-queue teardown must not deadlock");
            continue;
        }
        if mode == "control-error" {
            assert_eq!(next(&mut host).unwrap_err().code, ErrorCode::Protocol);
            assert!(matches!(
                next(&mut host).unwrap().unwrap().payload,
                EventPayload::Cancelled { .. }
            ));
            assert_eq!(next(&mut host).unwrap_err().code, ErrorCode::Protocol);
            assert!(next(&mut host).unwrap().is_none());
        } else {
            let e = next(&mut host).unwrap().unwrap();
            assert_eq!(e.task_id, Some(task));
            let EventPayload::Failed { error } = e.payload else {
                panic!("Expected owned failure")
            };
            assert!(
                error
                    .evidence
                    .iter()
                    .any(|e| e.contains("last_stage=fixture-compute"))
            );
            assert!(
                error
                    .evidence
                    .iter()
                    .any(|e| e.contains("actual diagnostic tail"))
            );
            assert!(error.evidence.iter().all(|e| e.len() < 66000));
            assert_ne!(error.code, ErrorCode::OutOfMemory);
            assert!(next(&mut host).unwrap().is_none());
        }
    }
}
