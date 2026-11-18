use feather_talk_gui_lib::worker_host::WorkerHost;
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
        host.next_event().unwrap().unwrap().payload,
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
        host.next_event().unwrap().unwrap().payload,
        EventPayload::Started { .. }
    ));
    let _ = host.terminate();
    let mut terminal = 0;
    while let Some(event) = host.next_event().unwrap() {
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
        let mut host = WorkerHost::spawn(&binary.canonicalize().unwrap(), context).unwrap();
        assert!(matches!(
            host.next_event().unwrap().unwrap().payload,
            EventPayload::Ready { .. }
        ));
        let task = TaskId::new();
        host.start(
            task,
            JobSpec::Train(TrainRequest {
                project_id: project,
                batch_mode: BatchMode::Fixed,
                config: TrainConfig::default(),
                device: DevicePolicy::Auto,
                resume: None,
            }),
        )
        .unwrap();
        if mode == "pre-start-control" {
            assert_eq!(host.next_event().unwrap_err().code, ErrorCode::Busy);
        }
        assert!(matches!(
            host.next_event().unwrap().unwrap().payload,
            EventPayload::Started { .. }
        ));
        assert!(matches!(
            host.next_event().unwrap().unwrap().payload,
            EventPayload::Progress { .. }
        ));
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
            assert_eq!(host.next_event().unwrap_err().code, ErrorCode::Protocol);
            assert!(matches!(
                host.next_event().unwrap().unwrap().payload,
                EventPayload::Cancelled { .. }
            ));
            assert_eq!(host.next_event().unwrap_err().code, ErrorCode::Protocol);
            assert!(host.next_event().unwrap().is_none());
        } else {
            let e = host.next_event().unwrap().unwrap();
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
            assert!(host.next_event().unwrap().is_none());
        }
    }
}
