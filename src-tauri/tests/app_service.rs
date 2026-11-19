use feather_talk_gui_lib::{
    projects::ProjectStore,
    resources::{Catalog, Component, FileSpec, ResourceManager},
    state::AppState,
};
use feathertalk_core::*;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    net::TcpListener,
    path::{Path, PathBuf},
    sync::mpsc::{self, Receiver, Sender},
    time::Duration,
};

fn make_project(root: &Path, name: &str) -> NewProject {
    let directory = root.join(name);
    fs::create_dir(&directory).unwrap();
    let source_video = directory.join("原始 video.mkv");
    fs::write(&source_video, b"single source for filesystem tests").unwrap();
    NewProject {
        name: name.into(),
        directory,
        source_video,
    }
}

#[test]
fn unicode_project_roundtrip() -> Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let index = dir.path().join("人物清单.json");
    let input = make_project(dir.path(), "人物 一");
    let old_directory = input.directory.clone();
    let project = ProjectStore::new(index.clone())?.create(input)?;
    let moved = dir.path().join("移动后 的目录");
    fs::rename(&old_directory, &moved).unwrap();
    let before = fs::read(moved.join("project.json")).unwrap();
    let store = ProjectStore::new(index.clone())?;
    assert!(store.get(project.id).is_err());
    assert_eq!(store.open(&moved)?.id, project.id);
    assert_eq!(fs::read(moved.join("project.json")).unwrap(), before);
    let reopened = ProjectStore::new(index)?;
    assert_eq!(reopened.list()?[0].directory, moved.canonicalize().unwrap());
    assert_eq!(
        reopened.get(project.id)?.source_video.path,
        Path::new("原始 video.mkv")
    );
    Ok(())
}

#[test]
fn damaged_project_reports_path_and_recovery() -> Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let store = ProjectStore::new(dir.path().join("projects.json"))?;
    let project = store.create(make_project(dir.path(), "测试"))?;
    let original = fs::read(&project.source_video).unwrap();
    fs::write(&project.source_video, vec![b'x'; original.len()]).unwrap();
    let bad_source = store.verify_source(project.id).unwrap_err();
    assert!(
        bad_source
            .evidence
            .iter()
            .any(|e| e.contains("原始 video.mkv"))
    );
    assert!(!bad_source.actions.is_empty());
    fs::write(&project.source_video, original).unwrap();
    fs::write(project.directory.join("project.json"), b"partial JSON").unwrap();
    let corrupt = store.get(project.id).unwrap_err();
    assert_eq!(corrupt.code, ErrorCode::ResourceCorrupt);
    assert!(corrupt.evidence.iter().any(|e| e.contains("project.json")));
    assert!(!corrupt.actions.is_empty());
    Ok(())
}

struct DownloadFixture {
    _dir: tempfile::TempDir,
    app: AppState,
    selection: ResourceSelection,
    connected: Receiver<()>,
    release: Sender<()>,
    events: Receiver<WorkerEvent>,
}

fn hash(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn prepare_fixture() -> Result<DownloadFixture> {
    let dir = tempfile::tempdir().unwrap();
    let (events_out, events) = mpsc::channel();
    let (connected_out, connected) = mpsc::channel();
    let (release, released) = mpsc::channel();
    let server = TcpListener::bind("127.0.0.1:0").unwrap();
    let url = format!("http://{}/manifest.json", server.local_addr().unwrap());
    let pin = ResourcePin {
        id: "test-model".into(),
        version: "1".into(),
    };
    let manifest = AssetManifest {
        identity: ModelIdentity {
            kind: "test".into(),
            architecture_version: "1".into(),
            source_sha256: hash(b"source"),
            weights_sha256: hash(b"source"),
        },
        relative_files: vec![],
        sizes: BTreeMap::new(),
        hashes: BTreeMap::new(),
        urls: BTreeMap::new(),
        min_engine_version: "0.1.0".into(),
        provenance: "loopback service test".into(),
    };
    let body = serde_json::to_vec(&manifest).unwrap();
    let component = Component {
        kind: "model".into(),
        target: None,
        pin: pin.clone(),
        manifest,
        manifest_sha256: hash(&body),
        manifest_size: body.len() as u64,
        manifest_urls: vec![url],
        supplemental: BTreeMap::<PathBuf, FileSpec>::new(),
        executables: vec![],
        notices: vec![],
        redistribution_approved: false,
        license: "NOASSERTION".into(),
    };
    std::thread::spawn(move || {
        let (mut stream, _) = server.accept().unwrap();
        stream.read(&mut [0; 4096]).unwrap();
        connected_out.send(()).unwrap();
        if released.recv_timeout(Duration::from_secs(10)).is_ok() {
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .unwrap();
            stream.write_all(&body).unwrap();
        }
    });
    let selection = ResourceSelection {
        target: TargetPlatform {
            os: std::env::consts::OS.into(),
            architecture: std::env::consts::ARCH.into(),
            backend: "cpu".into(),
        },
        resources: vec![pin],
    };
    let manager = ResourceManager::new(
        dir.path().join("resources"),
        Catalog {
            schema_version: 1,
            components: vec![component],
            defaults: vec![selection.clone()],
        },
    )?;
    let app = AppState::new(
        ProjectStore::new(dir.path().join("index.json"))?,
        manager,
        move |event| {
            let _ = events_out.send(event);
        },
    );
    Ok(DownloadFixture {
        _dir: dir,
        app,
        selection,
        connected,
        release,
        events,
    })
}

fn terminal(events: &Receiver<WorkerEvent>, id: TaskId) -> EventPayload {
    loop {
        let event = events.recv_timeout(Duration::from_secs(10)).unwrap();
        assert_eq!(event.task_id, Some(id));
        if matches!(
            event.payload,
            EventPayload::Completed { .. }
                | EventPayload::Cancelled { .. }
                | EventPayload::Failed { .. }
        ) {
            return event.payload;
        }
    }
}

#[test]
fn double_start_has_one_owner() -> Result<()> {
    let fixture = prepare_fixture()?;
    let app = &fixture.app;
    let project = app.create_project(make_project(fixture._dir.path(), "人物"))?;
    let first_id = app.prepare_resources(fixture.selection.clone())?;
    fixture
        .connected
        .recv_timeout(Duration::from_secs(10))
        .unwrap();
    let second = app.start_training(TrainRequest {
        batch_mode: BatchMode::Auto,
        project_id: project.id,
        config: TrainConfig::default(),
        device: DevicePolicy::Auto,
        resume: None,
    });
    assert_eq!(second.unwrap_err().code, ErrorCode::Busy);
    assert_eq!(app.snapshot()?.unwrap().task_id, first_id);
    fixture.release.send(()).unwrap();
    assert!(
        matches!(terminal(&fixture.events, first_id), EventPayload::Completed { result: ArtifactResult::Resources(resources) } if resources.ready)
    );
    Ok(())
}

#[test]
fn reconnect_returns_snapshot() -> Result<()> {
    let fixture = prepare_fixture()?;
    let id = fixture.app.prepare_resources(fixture.selection.clone())?;
    fixture
        .connected
        .recv_timeout(Duration::from_secs(10))
        .unwrap();
    let running = fixture.app.snapshot()?.unwrap();
    assert_eq!(running.task_id, id);
    assert_eq!(running.project_id, None);
    fixture.app.cancel_task(id)?;
    assert_eq!(
        fixture.app.snapshot()?.unwrap().status,
        TaskStatus::Cancelling
    );
    fixture.release.send(()).unwrap();
    assert!(matches!(
        terminal(&fixture.events, id),
        EventPayload::Cancelled { checkpoint: None }
    ));
    let ended = fixture.app.snapshot()?.unwrap();
    assert_eq!(ended.status, TaskStatus::Cancelled);
    assert!(ended.last_seq > running.last_seq);
    Ok(())
}

#[test]
fn history_uses_run_record_offsets_and_rejects_unsafe_paths() -> Result<()> {
    let dir = tempfile::tempdir().unwrap();
    let catalog = Catalog {
        schema_version: 1,
        components: vec![],
        defaults: vec![],
    };
    let app = AppState::new(
        ProjectStore::new(dir.path().join("projects.json"))?,
        ResourceManager::new(dir.path().join("cache"), catalog)?,
        |_| {},
    );
    let project = app.create_project(make_project(dir.path(), "历史 项目"))?;
    let mut manifest = app.get_project(project.id)?;
    let manifest_path = project.directory.join("project.json");
    manifest.metrics_path = "curves/run-log.jsonl".into();
    fs::create_dir(project.directory.join("curves")).unwrap();
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../contracts/desktop-fixture.json")).unwrap();
    let event: WorkerEvent = serde_json::from_value(fixture["events"][1].clone()).unwrap();
    let EventPayload::Metrics(mut metrics) = event.payload else {
        unreachable!()
    };
    metrics.project_id = project.id;
    let wanted = metrics.run_id;
    let q = MetricsPageRequest {
        project_id: project.id,
        run_id: wanted,
        offset: 0,
        limit: 1,
    };
    assert!(app.read_metrics(&q)?.items.is_empty());
    let mut history = fs::File::create(project.directory.join(&manifest.metrics_path)).unwrap();
    for run in [wanted, TrainingRunId::new(), wanted] {
        metrics.run_id = run;
        metrics.elapsed_ms += 600;
        serde_json::to_writer(&mut history, &metrics).unwrap();
        history.write_all(b"\n").unwrap();
    }
    let first = app.read_metrics(&q)?;
    assert_eq!(first.next_offset, Some(1));
    let second = app.read_metrics(&MetricsPageRequest {
        offset: 1,
        ..q.clone()
    })?;
    assert_eq!(second.next_offset, None);
    assert_eq!(second.items[0].step, first.items[0].step);
    assert_eq!(second.items[0].elapsed_ms, first.items[0].elapsed_ms + 1200);
    history.write_all(b"{\"incomplete\":").unwrap();
    assert!(
        app.read_metrics(&MetricsPageRequest {
            limit: 3,
            ..q.clone()
        })
        .unwrap_err()
        .message
        .contains("Interrupted trailing")
    );
    manifest.metrics_path = "../foreign.jsonl".into();
    fs::write(&manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
    assert!(
        app.read_metrics(&q)
            .unwrap_err()
            .message
            .contains("Unsafe metrics path")
    );
    Ok(())
}
