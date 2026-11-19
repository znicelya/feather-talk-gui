use feather_talk_gui_lib::{
    resources::{Catalog, InstalledWorker, ResourceManager},
    worker_host::{HostPoll, WorkerHost},
};
use feathertalk_core::*;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    process::Command,
    time::{Duration, Instant},
};

fn json<T: serde::de::DeserializeOwned>(path: &Path) -> T {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

fn package_directory(base: &Path) -> PathBuf {
    std::env::var_os("FT_WORKER_ACCEPTANCE_PACKAGES")
        .map_or_else(|| base.join("packages"), PathBuf::from)
}

#[test]
#[ignore = "explicit local CPU/CUDA packages"]
fn installed_binding_rejects_wrong_backend_and_changed_executable() {
    let base = PathBuf::from(std::env::var_os("FT_WORKER_ACCEPTANCE_ROOT").expect("artifact root"));
    let packages = package_directory(&base);
    let catalog: Catalog = json(&packages.join("catalog.json"));
    let imports: BTreeMap<String, PathBuf> = json(&packages.join("imports.json"));
    let cpu = catalog
        .defaults
        .iter()
        .find(|s| {
            s.target.os == std::env::consts::OS
                && s.target.architecture == std::env::consts::ARCH
                && s.target.backend == "cpu"
        })
        .cloned()
        .unwrap();
    let worker = cpu.resources.iter().find(|p| p.id == "worker").unwrap();
    let cache = tempfile::tempdir().unwrap();
    let manager = ResourceManager::new(cache.path().into(), catalog).unwrap();
    manager
        .import_local(&imports[&format!("{}/{}", worker.id, worker.version)])
        .unwrap();

    let directory = tempfile::tempdir().unwrap();
    let project_id = ProjectId::new();
    let project = ProjectManifest {
        schema_version: 1,
        id: project_id,
        name: "Binding fixture".into(),
        source_video: SourceVideo {
            path: "source.mkv".into(),
            sha256: "unused".into(),
            size_bytes: 0,
        },
        preprocessing: None,
        model_identity: None,
        checkpoints: vec![],
        metrics_path: "metrics.jsonl".into(),
    };
    fs::write(
        directory.path().join("project.json"),
        serde_json::to_vec(&project).unwrap(),
    )
    .unwrap();
    let binding = ProjectBinding {
        project_id,
        directory: directory.path().into(),
    };
    let job = JobSpec::Train(TrainRequest {
        project_id,
        batch_mode: BatchMode::Fixed,
        config: TrainConfig::default(),
        device: DevicePolicy::Auto,
        resume: None,
    });
    let mut wrong = cpu.clone();
    wrong.target.backend = "cuda".into();
    assert_eq!(
        manager
            .bind_worker(&wrong, binding.clone(), &job)
            .unwrap_err()
            .code,
        ErrorCode::UnsupportedBackend
    );
    let correct_before_tamper = manager
        .bind_worker(&cpu, binding.clone(), &job)
        .unwrap_err();
    assert_eq!(correct_before_tamper.code, ErrorCode::ResourceMissing);
    assert!(correct_before_tamper.message.contains("probe"));
    let partial = manager.inventory(&cpu.target).unwrap();
    assert!(!partial.ready);
    for item in partial.items.iter().filter(|i| !i.ready) {
        let pin = &item.resource;
        manager
            .import_local(&imports[&format!("{}/{}", pin.id, pin.version)])
            .unwrap();
    }
    let full = manager.inventory(&cpu.target).unwrap();
    assert!(full.ready, "{full:?}");

    let installed = cache.path().join(&worker.id).join(&worker.version);
    let name = if cfg!(windows) {
        "feathertalk-worker.exe"
    } else {
        "feathertalk-worker"
    };
    let mut file = fs::OpenOptions::new()
        .write(true)
        .open(installed.join(name))
        .unwrap();
    file.seek(SeekFrom::Start(0)).unwrap();
    file.write_all(b"x").unwrap();
    drop(file);
    let corrupt = manager.bind_worker(&cpu, binding, &job).unwrap_err();
    assert_eq!(corrupt.code, ErrorCode::ResourceCorrupt);
    assert!(corrupt.message.contains("SHA-256 mismatch"));
    assert!(!manager.inventory(&cpu.target).unwrap().ready);
}

fn next(host: &mut WorkerHost) -> Option<WorkerEvent> {
    let until = Instant::now() + Duration::from_secs(900);
    loop {
        match host.next_event(Duration::from_secs(1)).unwrap() {
            HostPoll::Event(event) => {
                println!("{}", serde_json::to_string(&event).unwrap());
                return Some(event);
            }
            HostPoll::Ended => return None,
            HostPoll::Timeout => assert!(Instant::now() < until, "worker event deadline"),
        }
    }
}

fn wait_for_marker(host: &mut WorkerHost, task: TaskId, marker: &Path, timeout: Duration) {
    let until = Instant::now() + timeout;
    while !marker.is_file() {
        assert!(
            Instant::now() < until,
            "gate not reached: {}",
            marker.display()
        );
        match host.next_event(Duration::from_millis(100)).unwrap() {
            HostPoll::Timeout => {}
            HostPoll::Event(event) => {
                println!("{}", serde_json::to_string(&event).unwrap());
                assert_eq!(event.task_id, Some(task));
                assert!(
                    matches!(
                        event.payload,
                        EventPayload::Started { .. } | EventPayload::Progress { .. }
                    ),
                    "worker advanced past the gate or failed: {event:?}"
                );
            }
            HostPoll::Ended => panic!("worker exited before {}", marker.display()),
        }
    }
    println!("observed gate {}", marker.display());
}

fn assert_backend(device: &DeviceDescriptor, backend: &str) {
    match backend {
        "cpu" => assert!(matches!(
            device.backend,
            BackendKind::Flex | BackendKind::Cpu
        )),
        "cuda" => {
            assert_eq!(device.backend, BackendKind::Cuda);
            assert!(device.name.contains("RTX 2060"), "{device:?}");
        }
        _ => panic!("unsupported acceptance backend {backend}"),
    }
}

fn ready(installed: InstalledWorker, backend: &str) -> WorkerHost {
    let mut host = installed.spawn().unwrap();
    let EventPayload::Ready {
        capabilities,
        devices,
    } = next(&mut host).unwrap().payload
    else {
        panic!("worker did not announce readiness");
    };
    assert!(capabilities.generation && capabilities.training);
    assert!(!devices.is_empty());
    devices.iter().for_each(|d| assert_backend(d, backend));
    host
}

fn run_tool(path: &Path, fill: impl FnOnce(&mut Command)) -> Vec<u8> {
    let mut command = Command::new(path);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    fill(&mut command);
    let result = command.output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    result.stdout
}

#[test]
#[ignore = "explicit local worker packages, real media, and available hardware"]
fn installed_train_cancel_crash_and_restart() {
    exercise_installed_worker(false);
}

#[test]
#[ignore = "explicit instrumented debug worker (dev-worker-acceptance), real media and hardware"]
fn installed_cancels_during_update_and_recovers_active_training_checkpoint() {
    exercise_installed_worker(true);
}

fn exercise_installed_worker(gates: bool) {
    let base = PathBuf::from(std::env::var_os("FT_WORKER_ACCEPTANCE_ROOT").expect("artifact root"));
    let base = base.canonicalize().unwrap();
    let backend = std::env::var("FT_WORKER_ACCEPTANCE_BACKEND").expect("cpu or cuda");
    let reference_video = PathBuf::from(
        std::env::var_os("FT_WORKER_ACCEPTANCE_VIDEO").expect("one local fixture video"),
    );
    let packages = package_directory(&base);
    let cache = base.join("cache");
    let mut catalog: Catalog = json(&packages.join("catalog.json"));
    let imports: BTreeMap<String, PathBuf> = json(&packages.join("imports.json"));
    let selection = catalog
        .defaults
        .iter()
        .find(|s| {
            s.target.os == std::env::consts::OS
                && s.target.architecture == std::env::consts::ARCH
                && s.target.backend == backend
        })
        .cloned()
        .expect("local packaged target");
    let media_pin = selection
        .resources
        .iter()
        .find(|p| p.id == "media")
        .unwrap();
    let manager = ResourceManager::new(cache.clone(), catalog.clone()).unwrap();
    let before = manager.inventory_selection(&selection).unwrap();
    for item in before.items.iter().filter(|i| !i.ready) {
        let pin = &item.resource;
        manager
            .import_local(&imports[&format!("{}/{}", pin.id, pin.version)])
            .unwrap();
    }
    let inventory = manager.inventory(&selection.target).unwrap();
    assert!(inventory.ready, "{inventory:?}");
    let media = inventory
        .items
        .iter()
        .find(|i| i.resource == *media_pin)
        .unwrap()
        .path
        .as_ref()
        .unwrap();
    let exe = |name: &str| {
        media.join("bin").join(if cfg!(windows) {
            format!("{name}.exe")
        } else {
            name.into()
        })
    };

    let directory = tempfile::Builder::new()
        .prefix(&format!("{backend} worker "))
        .tempdir_in(&base)
        .unwrap()
        .keep();
    println!("Project fixture: {}", directory.display());
    let source = directory.join("source.mkv");
    if gates {
        // Two frames make a one-pair epoch: the resumed job publishes a real
        // periodic checkpoint on its first complete update, before job completion.
        run_tool(&exe("ffmpeg"), |cmd| {
            cmd.args(["-nostdin", "-v", "error", "-i"])
                .arg(&reference_video)
                .args([
                    "-map",
                    "0:v:0",
                    "-map",
                    "0:a:0",
                    "-vf",
                    "trim=end_frame=2",
                    "-c:v",
                    "ffv1",
                    "-c:a",
                    "pcm_s16le",
                ])
                .arg(&source);
        });
    } else {
        fs::copy(&reference_video, &source).unwrap();
    }
    let audio = directory.join("driving audio.wav");
    run_tool(&exe("ffmpeg"), |cmd| {
        cmd.args(["-nostdin", "-v", "error", "-i"])
            .arg(&reference_video)
            .args(["-map", "0:a:0", "-vn", "-ac", "1", "-ar", "16000"])
            .arg(&audio);
    });
    let project = ProjectManifest {
        schema_version: 1,
        id: ProjectId::new(),
        name: "Derived worker integration fixture".into(),
        source_video: SourceVideo {
            path: "source.mkv".into(),
            sha256: format!("{:x}", Sha256::digest(fs::read(&source).unwrap())),
            size_bytes: fs::metadata(&source).unwrap().len(),
        },
        preprocessing: None,
        model_identity: None,
        checkpoints: vec![],
        metrics_path: "metrics.jsonl".into(),
    };
    let project_path = directory.join("project.json");
    fs::write(&project_path, serde_json::to_vec_pretty(&project).unwrap()).unwrap();
    let binding = ProjectBinding {
        project_id: project.id,
        directory: directory.clone(),
    };
    let train = JobSpec::Train(TrainRequest {
        batch_mode: if backend == "cuda" && !gates {
            BatchMode::Auto
        } else {
            BatchMode::Fixed
        },
        project_id: project.id,
        config: TrainConfig {
            epochs: 3,
            batch_size: 1,
            save_every: 1,
            ..Default::default()
        },
        device: DevicePolicy::Auto,
        resume: None,
    });

    let mut missing_runtime = selection.clone();
    missing_runtime.resources.retain(|p| p.id != "probe");
    let error = manager
        .bind_worker(&missing_runtime, binding.clone(), &train)
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::ResourceMissing);
    let mut foreign = binding.clone();
    foreign.project_id = ProjectId::new();
    assert_eq!(
        manager
            .bind_worker(&selection, foreign, &train)
            .unwrap_err()
            .code,
        ErrorCode::ModelMismatch
    );

    let mut host = ready(
        manager
            .bind_worker(&selection, binding.clone(), &train)
            .unwrap(),
        &backend,
    );
    let task = TaskId::new();
    let started = Instant::now();
    let gate_dir = directory.join(".worker-acceptance");
    if gates {
        fs::create_dir(&gate_dir).unwrap();
        fs::write(gate_dir.join("cancel-after-forward"), []).unwrap();
    }
    host.start(task, train.clone()).unwrap();
    assert_eq!(
        host.start(TaskId::new(), train.clone()).unwrap_err().code,
        ErrorCode::Busy
    );
    if gates {
        wait_for_marker(
            &mut host,
            task,
            &gate_dir.join("forward-complete"),
            Duration::from_secs(900),
        );
        let cancellation_at = Instant::now();
        host.cancel(task).unwrap();
        wait_for_marker(
            &mut host,
            task,
            &gate_dir.join("cancel-observed"),
            Duration::from_secs(5),
        );
        assert!(
            json::<ProjectManifest>(&project_path)
                .checkpoints
                .is_empty()
        );
        assert!(
            !directory
                .join(&project.metrics_path)
                .metadata()
                .is_ok_and(|m| m.len() > 0)
        );
        println!(
            "{backend}: worker observed Cancel inside an uncommitted optimizer step after {:.2?}",
            cancellation_at.elapsed()
        );
        fs::write(gate_dir.join("release-forward"), []).unwrap();
    }
    let mut first_metric = None;
    let mut checkpoint = None;
    let mut progress = gates; // The gate waiter already drained initial progress.
    loop {
        let event = next(&mut host).expect("training event");
        assert_eq!(event.task_id, Some(task));
        match event.payload {
            EventPayload::Progress { .. } => progress = true,
            EventPayload::Metrics(metric) => {
                assert_backend(&metric.device, &backend);
                assert!(metric.step > 0 && metric.losses.total.is_finite());
                if first_metric.is_none() {
                    if !gates {
                        host.cancel(task).unwrap();
                    }
                    first_metric = Some(metric);
                }
            }
            EventPayload::CheckpointCommitted { model_ref } => checkpoint = Some(model_ref),
            EventPayload::Cancelled {
                checkpoint: Some(model),
            } => {
                assert_eq!(checkpoint.as_ref(), Some(&model));
                break;
            }
            EventPayload::Failed { error } => panic!("training failed: {error:?}"),
            EventPayload::Completed { .. } | EventPayload::Cancelled { .. } => {
                panic!("training did not save after cancellation")
            }
            _ => {}
        }
    }
    let checkpoint = checkpoint.unwrap();
    let first_metric = first_metric.expect("a real, committed training step");
    assert!(progress);
    assert_eq!(first_metric.run_id, checkpoint.run_id);
    let indexed: ProjectManifest = json(&project_path);
    let saved = indexed
        .checkpoints
        .iter()
        .find(|e| e.model_ref == checkpoint)
        .unwrap();
    assert!(saved.step >= first_metric.step);
    if gates {
        assert_eq!((saved.step, saved.epoch), (1, 1));
    }
    assert!(
        directory
            .join(&indexed.metrics_path)
            .metadata()
            .unwrap()
            .len()
            > 0
    );
    println!(
        "{backend}: cancellation saved step {} after {:.2?}",
        saved.step,
        started.elapsed()
    );

    let mut generate = GenerateRequest {
        project_id: project.id,
        audio_path: audio,
        output_path: directory.join("prior output.mp4"),
        model_ref: checkpoint.clone(),
        device: DevicePolicy::Auto,
    };
    fs::write(&generate.output_path, b"preexisting export bytes").unwrap();
    let crash_task = TaskId::new();
    let (crashed_reference, expected_index) = if gates {
        fs::remove_file(gate_dir.join("cancel-after-forward")).unwrap();
        fs::write(gate_dir.join("hold-checkpoint"), []).unwrap();
        let JobSpec::Train(mut resume) = train else {
            unreachable!()
        };
        resume.resume = Some(checkpoint.clone());
        host.start(crash_task, JobSpec::Train(resume)).unwrap();
        let reference = loop {
            let event = next(&mut host).expect("resumed training event before forced exit");
            assert_eq!(event.task_id, Some(crash_task));
            match event.payload {
                EventPayload::CheckpointCommitted { model_ref } => break model_ref,
                EventPayload::Failed { .. }
                | EventPayload::Completed { .. }
                | EventPayload::Cancelled { .. } => {
                    panic!("resumed job did not stay active after a periodic save: {event:?}");
                }
                _ => {}
            }
        };
        wait_for_marker(
            &mut host,
            crash_task,
            &gate_dir.join("checkpoint-held"),
            Duration::from_secs(5),
        );
        assert_eq!(reference.run_id, checkpoint.run_id);
        assert_ne!(reference, checkpoint);
        let updated: ProjectManifest = json(&project_path);
        assert_eq!(updated.checkpoints.len(), indexed.checkpoints.len() + 1);
        assert!(updated.checkpoints.contains(saved));
        let periodic = updated
            .checkpoints
            .iter()
            .find(|e| e.model_ref == reference)
            .unwrap();
        assert_eq!((periodic.step, periodic.epoch), (2, 2));
        generate.model_ref = reference.clone();
        (Some(reference), updated)
    } else {
        host.start(crash_task, JobSpec::Generate(generate.clone()))
            .unwrap();
        loop {
            let event = next(&mut host).expect("generation event before forced exit");
            assert_eq!(event.task_id, Some(crash_task));
            match event.payload {
                EventPayload::Progress { stage, .. } if stage == "audio-features" => break,
                EventPayload::Failed { .. }
                | EventPayload::Completed { .. }
                | EventPayload::Cancelled { .. } => {
                    panic!("generation ended before forced exit: {event:?}");
                }
                _ => {}
            }
        }
        (None, indexed)
    };
    host.terminate().unwrap();
    let mut failures = 0;
    while let Some(event) = next(&mut host) {
        assert_eq!(event.task_id, Some(crash_task));
        match event.payload {
            EventPayload::Failed { error } => {
                failures += 1;
                assert!(
                    error
                        .evidence
                        .iter()
                        .any(|e| e.contains("exit=") && e.contains("last_stage="))
                );
                if let Some(reference) = &crashed_reference {
                    assert!(
                        error
                            .evidence
                            .iter()
                            .any(|e| e.contains("committed_checkpoint=Some(")
                                && e.contains(&reference.checkpoint_id))
                    );
                }
            }
            EventPayload::Completed { .. } | EventPayload::Cancelled { .. } => {
                panic!("forced exit looked like a completed or cancelled job")
            }
            _ => {}
        }
    }
    assert_eq!(failures, 1);
    assert_eq!(
        fs::read(&generate.output_path).unwrap(),
        b"preexisting export bytes"
    );
    assert_eq!(json::<ProjectManifest>(&project_path), expected_index);

    // A later catalog no longer offers the original model bundle. The accepted checkpoint still does.
    catalog
        .components
        .retain(|component| component.pin.id != "public-assets");
    let mut later = selection.clone();
    later
        .resources
        .iter_mut()
        .find(|pin| pin.id == "public-assets")
        .unwrap()
        .version = "unconfigured".into();
    let retained = ResourceManager::new(cache, catalog).unwrap();
    assert!(!retained.inventory_selection(&later).unwrap().ready);
    generate.output_path = directory.join("\u{9a8c}\u{6536} video.mp4");
    let job = JobSpec::Generate(generate.clone());
    let started = Instant::now();
    let mut host = ready(
        retained.bind_worker(&later, binding, &job).unwrap(),
        &backend,
    );
    let task = TaskId::new();
    host.start(task, job).unwrap();
    let output = loop {
        let event = next(&mut host).expect("event after worker restart");
        assert_eq!(event.task_id, Some(task));
        match event.payload {
            EventPayload::Completed {
                result: ArtifactResult::Video(result),
            } => break result,
            EventPayload::Failed { error } => panic!("generation failed after restart: {error:?}"),
            EventPayload::Completed { .. } | EventPayload::Cancelled { .. } => {
                panic!("unexpected generation result")
            }
            _ => {}
        }
    };
    assert_backend(&output.device, &backend);
    assert_eq!(output.video_frames, 3);
    assert_eq!(
        output.output_path,
        generate.output_path.canonicalize().unwrap()
    );
    host.shutdown().unwrap();
    assert!(next(&mut host).is_none(), "duplicate terminal on shutdown");
    let streams: serde_json::Value = serde_json::from_slice(&run_tool(&exe("ffprobe"), |cmd| {
        cmd.args(["-v", "error", "-show_streams", "-of", "json"])
            .arg(&output.output_path);
    }))
    .unwrap();
    let streams = streams["streams"].as_array().unwrap();
    assert!(streams.iter().any(|s| s["codec_name"] == "h264"
        && s["avg_frame_rate"] == "25/1"
        && s["nb_frames"] == "3"));
    assert!(streams.iter().any(|s| s["codec_name"] == "aac"));
    println!(
        "{backend}: restarted export {} in {:.2?}",
        output.output_path.display(),
        started.elapsed()
    );
}
