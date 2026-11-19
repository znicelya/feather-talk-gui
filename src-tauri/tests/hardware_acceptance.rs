//! Local hardware smoke using the desktop task service and verified runtime packages.
use feather_talk_gui_lib::{
    projects::ProjectStore,
    resources::{Catalog, ResourceManager},
    state::AppState,
};
use feathertalk_core::*;
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

fn required(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("Set {name} before running hardware acceptance"))
}

fn read<T: serde::de::DeserializeOwned>(path: &Path) -> T {
    serde_json::from_slice(&fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display())))
        .unwrap()
}

fn sha(path: &Path) -> String {
    let mut file = fs::File::open(path).unwrap();
    let mut hash = Sha256::new();
    std::io::copy(&mut file, &mut hash).unwrap();
    format!("{:x}", hash.finalize())
}

fn wait(app: &AppState, id: TaskId, check: impl FnMut(&TaskSnapshot)) -> TaskSnapshot {
    let until = Instant::now() + Duration::from_secs(1800);
    let mut check = check;
    loop {
        if let Some(task) = app.snapshot().unwrap().filter(|task| task.task_id == id) {
            check(&task);
            if !task.status.is_active() {
                return task;
            }
        }
        assert!(
            Instant::now() < until,
            "task timed out: {:?}",
            app.snapshot()
        );
        std::thread::sleep(Duration::from_millis(100));
    }
}

fn completed(task: TaskSnapshot) -> ArtifactResult {
    assert_eq!(task.status, TaskStatus::Completed, "{:?}", task.error);
    task.result.unwrap()
}

#[test]
#[ignore = "requires explicitly staged local media, model assets and real hardware"]
fn train_cancel_reopen_resume_export() {
    let root = PathBuf::from(required("FT_SMOKE_ROOT"))
        .canonicalize()
        .unwrap();
    let package = PathBuf::from(required("FT_SMOKE_PACKAGES"));
    let source = PathBuf::from(required("FT_SMOKE_VIDEO"))
        .canonicalize()
        .unwrap();
    let audio = PathBuf::from(required("FT_SMOKE_AUDIO"))
        .canonicalize()
        .unwrap();
    let backend = required("FT_SMOKE_BACKEND");
    let expected = required("FT_SMOKE_DEVICE").to_lowercase();
    let catalog: Catalog = read(&package.join("catalog.json"));
    let imports: BTreeMap<String, PathBuf> = read(&package.join("imports.json"));
    let selection = catalog
        .defaults
        .iter()
        .find(|s| {
            s.target.os == std::env::consts::OS
                && s.target.architecture == std::env::consts::ARCH
                && s.target.backend == backend
        })
        .cloned()
        .expect("catalog has no matching target and backend");
    let kind = match backend.as_str() {
        "cpu" => BackendKind::Flex,
        "vulkan" => BackendKind::Vulkan,
        "cuda" => BackendKind::Cuda,
        "metal" => BackendKind::Metal,
        _ => panic!("unsupported runtime backend {backend}"),
    };
    assert!(!expected.trim().is_empty());
    let log = Arc::new(Mutex::new(Some(
        fs::File::create_new(root.join("task-events.jsonl")).unwrap(),
    )));
    let new_session = || {
        AppState::new(
            ProjectStore::new(root.join("projects.json")).unwrap(),
            ResourceManager::new(root.join("cache"), catalog.clone()).unwrap(),
            {
                let log = log.clone();
                move |event| {
                    let mut slot = log.lock().unwrap();
                    if let Some(out) = slot.as_mut()
                        && (serde_json::to_writer(&mut *out, &event).is_err()
                            || out.write_all(b"\n").is_err())
                    {
                        *slot = None;
                    }
                }
            },
        )
    };
    let app = new_session();
    for pin in &selection.resources {
        let key = format!("{}/{}", pin.id, pin.version);
        let manifest = imports
            .get(&key)
            .unwrap_or_else(|| panic!("Missing local package {key}"));
        eprintln!("Importing {key}");
        let task = app
            .import_resource(
                selection.target.clone(),
                package.join(manifest).canonicalize().unwrap(),
            )
            .unwrap();
        completed(wait(&app, task, |_| {}));
    }
    let inventory = app.get_resources(&selection.target).unwrap();
    assert!(inventory.ready, "{:?}", inventory.items);
    let installed = app.list_devices().unwrap();
    let mut matches = installed
        .iter()
        .filter(|d| d.backend == kind && d.name.to_lowercase().contains(&expected));
    let actual = matches
        .next()
        .expect("requested physical device was not found")
        .clone();
    assert!(
        matches.next().is_none(),
        "choose a device substring matching one adapter only"
    );
    let confirm = |d: &DeviceDescriptor| {
        assert_eq!(d.backend, actual.backend);
        assert_eq!(d.id, actual.id);
        assert_eq!(d.name, actual.name);
    };
    let binding = DevicePolicy::Explicit(DeviceRequest {
        backend: actual.backend,
        device_id: actual.id.clone(),
    });
    let project = app
        .create_project(NewProject {
            name: "FeatherTalk hardware smoke".into(),
            directory: root.join("训练 project"),
            source_video: source.clone(),
        })
        .unwrap();
    let mut train = TrainRequest {
        project_id: project.id,
        batch_mode: BatchMode::Fixed,
        config: TrainConfig {
            epochs: 60,
            batch_size: 1,
            save_every: 60,
            ..TrainConfig::default()
        },
        device: binding.clone(),
        resume: None,
    };
    let started = Instant::now();
    eprintln!("Training on {} ({})", actual.name, actual.id);
    let training = app.start_training(train.clone()).unwrap();
    let mut first_metric = None;
    let stopped = wait(&app, training, |task| {
        if first_metric.is_none()
            && task.status.is_active()
            && let Some(metrics) = &task.metrics
        {
            confirm(&metrics.device);
            assert!(metrics.step > 0 && metrics.losses.total.is_finite());
            app.cancel_task(training).unwrap();
            first_metric = Some(metrics.clone());
        }
    });
    let train_ms = started.elapsed().as_millis();
    let first_metric = first_metric.expect("fresh training did not publish a loss");
    assert_eq!(stopped.status, TaskStatus::Cancelled, "{:?}", stopped.error);
    let before = stopped.checkpoint.expect("cancel did not save a model");
    let old = app
        .get_project(project.id)
        .unwrap()
        .checkpoints
        .into_iter()
        .find(|e| e.model_ref == before)
        .unwrap();
    assert!(old.step >= first_metric.step);
    drop(app);

    let app = new_session();
    assert!(app.get_resources(&selection.target).unwrap().ready);
    let history = app
        .read_metrics(&MetricsPageRequest {
            project_id: project.id,
            run_id: before.run_id,
            offset: 0,
            limit: 32,
        })
        .unwrap();
    assert!(!history.items.is_empty());
    train.config.epochs = old.epoch + 1;
    train.config.save_every = 1;
    train.resume = Some(before.clone());
    let started = Instant::now();
    eprintln!("Resuming from saved step {}", old.step);
    let resumed = app.start_training(train).unwrap();
    let resumed = wait(&app, resumed, |_| {});
    if let Some(metrics) = &resumed.metrics {
        confirm(&metrics.device);
        assert!(metrics.step > old.step && metrics.losses.total.is_finite());
    } else {
        panic!("resumed training did not publish a loss");
    }
    let ArtifactResult::Model(after) = completed(resumed) else {
        panic!("no resumed checkpoint")
    };
    let resume_ms = started.elapsed().as_millis();
    assert_eq!(after.run_id, before.run_id);
    let saved = app
        .get_project(project.id)
        .unwrap()
        .checkpoints
        .into_iter()
        .find(|e| e.model_ref == after)
        .unwrap();
    assert!(saved.step > old.step);

    let started = Instant::now();
    eprintln!("Exporting from resumed step {}", saved.step);
    let output = root.join("导出 video.mp4");
    let export = app
        .start_generation(GenerateRequest {
            project_id: project.id,
            model_ref: after,
            audio_path: audio.clone(),
            output_path: output,
            device: binding,
        })
        .unwrap();
    let ArtifactResult::Video(result) = completed(wait(&app, export, |_| {})) else {
        panic!("no generated video")
    };
    let export_ms = started.elapsed().as_millis();
    confirm(&result.device);
    assert!(result.video_frames > 0);
    let media = inventory
        .items
        .iter()
        .find(|item| item.resource.id == "media")
        .unwrap()
        .path
        .as_ref()
        .unwrap();
    let installed = |role: &str| {
        inventory
            .items
            .iter()
            .find(|item| item.resource.id == role)
            .unwrap()
            .path
            .as_ref()
            .unwrap()
    };
    let component = |role: &str| {
        catalog
            .components
            .iter()
            .find(|c| {
                c.pin
                    == selection
                        .resources
                        .iter()
                        .find(|p| p.id == role)
                        .cloned()
                        .unwrap()
            })
            .unwrap()
    };
    let ffprobe = component("media")
        .executables
        .iter()
        .find(|p| p.file_stem().is_some_and(|n| n == "ffprobe"))
        .unwrap();
    let probe = Command::new(media.join(ffprobe))
        .args([
            "-v",
            "error",
            "-count_frames",
            "-show_streams",
            "-of",
            "json",
        ])
        .arg(&result.output_path)
        .output()
        .unwrap();
    assert!(
        probe.status.success(),
        "{}",
        String::from_utf8_lossy(&probe.stderr)
    );
    let observed: Value = serde_json::from_slice(&probe.stdout).unwrap();
    let streams = observed["streams"].as_array().unwrap();
    assert!(streams.iter().any(|s| s["codec_name"] == "h264"
        && s["avg_frame_rate"] == "25/1"
        && s["nb_read_frames"] == result.video_frames.to_string()));
    assert!(streams.iter().any(|s| s["codec_name"] == "aac"));
    assert_eq!(
        app.completed_video(project.id, export).unwrap(),
        result.output_path
    );

    let mut models = BTreeMap::new();
    for name in [
        "feather_hubert",
        "scrfd_2_5g",
        "pfld_ghost_one",
        "vgg19",
        "mean_face",
    ] {
        let manifest = installed("public-assets").join(name).join("manifest.json");
        models.insert(name, sha(&manifest));
    }
    let components: BTreeMap<_, _> = selection
        .resources
        .iter()
        .filter(|pin| pin.id != "public-assets")
        .map(|pin| {
            let c = component(&pin.id);
            (
                pin.id.as_str(),
                json!({"version": pin.version, "sha256": c.manifest_sha256, "target": c.target}),
            )
        })
        .collect();
    log.lock()
        .unwrap()
        .as_ref()
        .expect("Could not record task events")
        .sync_all()
        .unwrap();
    let summary = json!({
        "device": actual, "target": selection.target, "components": components, "models": models,
        "source_sha256": sha(&source), "audio_sha256": sha(&audio), "output_sha256": sha(&result.output_path),
        "train_ms": train_ms, "resume_ms": resume_ms, "export_ms": export_ms,
        "cancelled_step": old.step, "resumed_step": saved.step, "export": result, "probe": observed,
    });
    let bytes = serde_json::to_vec_pretty(&summary).unwrap();
    fs::write(root.join("smoke-result.json"), bytes).unwrap();
    println!("{}", serde_json::to_string_pretty(&summary).unwrap());
}
