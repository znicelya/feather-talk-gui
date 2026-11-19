use super::*;
use crate::resources::Catalog;
use std::{fs, sync::mpsc, time::Duration};

const WAIT: Duration = Duration::from_secs(5);

fn fixture() -> (tempfile::TempDir, AppState, mpsc::Receiver<WorkerEvent>) {
    let dir = tempfile::tempdir().unwrap();
    let resources = ResourceManager::new(
        dir.path().join("cache"),
        Catalog {
            schema_version: 1,
            defaults: vec![],
            components: vec![],
        },
    )
    .unwrap();
    let projects = ProjectStore::new(dir.path().join("projects.json")).unwrap();
    let (send, events) = mpsc::channel();
    let app = AppState::new(projects, resources, move |event| {
        let _ = send.send(event);
    });
    (dir, app, events)
}

fn project(app: &AppState, root: &Path, name: &str) -> ProjectSummary {
    let dir = root.join(name);
    fs::create_dir(&dir).unwrap();
    let source_video = dir.join("source.mp4");
    fs::write(&source_video, b"test source").unwrap();
    app.create_project(NewProject {
        name: name.into(),
        directory: dir,
        source_video,
    })
    .unwrap()
}

#[test]
fn project_switch_does_not_reassign_events() -> Result<()> {
    let (dir, app, events) = fixture();
    let first = project(&app, dir.path(), "first");
    let second = project(&app, dir.path(), "second");
    let (release, waiting) = mpsc::channel();
    let run = TrainingRunId::new();
    let model = ModelRef {
        project_id: first.id,
        run_id: run,
        checkpoint_id: "step-1".into(),
    };
    let checkpoint = model.clone();
    let id = app.submit(
        JobKind::Train,
        Some(first.id),
        Some(run),
        move |app, task, _| {
            app.record(WorkerEvent::new(
                Some(task),
                4,
                EventPayload::Started {
                    job: JobKind::Train,
                },
            )?)?;
            waiting.recv_timeout(WAIT).unwrap();
            Ok(Conclusion::Local(EventPayload::Completed {
                result: ArtifactResult::Model(checkpoint),
            }))
        },
    )?;
    let started = events.recv_timeout(WAIT).unwrap();
    assert_eq!(app.get_project(second.id)?.id, second.id);

    let wrong_run = ModelRef {
        run_id: TrainingRunId::new(),
        ..model.clone()
    };
    let wrong_project = ModelRef {
        project_id: second.id,
        ..model.clone()
    };
    for (owner, seq, reference) in [
        (TaskId::new(), 8, model.clone()),
        (id, started.seq, model.clone()),
        (id, 8, wrong_run),
        (id, 8, wrong_project),
    ] {
        assert_eq!(
            app.record(WorkerEvent::new(
                Some(owner),
                seq,
                EventPayload::CheckpointCommitted {
                    model_ref: reference
                }
            )?)
            .unwrap_err()
            .code,
            ErrorCode::Protocol
        );
    }
    let before = app.snapshot()?.unwrap();
    assert_eq!(before.latest_event, Some(started));
    assert!(before.checkpoint.is_none());

    let copy = dir.path().join("another current location");
    fs::create_dir(&copy).unwrap();
    fs::copy(&first.source_video, copy.join("source.mp4")).unwrap();
    fs::copy(
        first.directory.join("project.json"),
        copy.join("project.json"),
    )
    .unwrap();
    assert_eq!(app.open_project(&copy).unwrap_err().code, ErrorCode::Busy);
    assert_eq!(app.project_directory(first.id)?, first.directory);

    release.send(()).unwrap();
    let ended = events.recv_timeout(WAIT).unwrap();
    let snapshot = app.snapshot()?.unwrap();
    assert_eq!(snapshot.project_id, Some(first.id));
    assert_eq!(snapshot.task_id, id);
    assert_eq!(snapshot.status, TaskStatus::Completed);
    assert_eq!(snapshot.latest_event, Some(ended));
    assert_eq!(snapshot.checkpoint, Some(model));
    assert_eq!(app.open_project(&copy)?.id, first.id);
    Ok(())
}

#[test]
fn close_waits_for_work_and_preserves_the_cancelled_checkpoint() -> Result<()> {
    let (dir, app, events) = fixture();
    let project = project(&app, dir.path(), "close");
    let (exit, exited) = mpsc::channel();
    let model = ModelRef {
        project_id: project.id,
        run_id: TrainingRunId::new(),
        checkpoint_id: "step-1".into(),
    };
    let checkpoint = model.clone();
    let id = app.submit(
        JobKind::Train,
        Some(project.id),
        None,
        move |app, task, cancel| {
            app.payload(
                task,
                EventPayload::Started {
                    job: JobKind::Train,
                },
            )?;
            exited.recv_timeout(WAIT).unwrap();
            assert!(cancel.is_cancelled());
            app.payload(
                task,
                EventPayload::CheckpointCommitted {
                    model_ref: checkpoint.clone(),
                },
            )?;
            Ok(Conclusion::Local(EventPayload::Cancelled {
                checkpoint: Some(checkpoint),
            }))
        },
    )?;
    events.recv_timeout(WAIT).unwrap();
    assert_eq!(app.request_close()?, CloseState::Waiting);
    assert_eq!(app.request_close()?, CloseState::AlreadyWaiting);
    assert_eq!(app.snapshot()?.unwrap().status, TaskStatus::Cancelling);
    let (closed, observed) = mpsc::channel();
    let closing_app = app.clone();
    let waiter = thread::spawn(move || {
        closing_app.wait_until_idle().unwrap();
        closed
            .send(closing_app.snapshot().unwrap().unwrap())
            .unwrap();
    });
    assert!(observed.recv_timeout(Duration::from_millis(50)).is_err());
    exit.send(()).unwrap();
    let after = observed.recv_timeout(WAIT).unwrap();
    waiter.join().unwrap();
    assert_eq!(after.task_id, id);
    assert_eq!(after.status, TaskStatus::Cancelled);
    assert_eq!(after.checkpoint, Some(model));
    assert_eq!(app.request_close()?, CloseState::Ready);
    Ok(())
}
