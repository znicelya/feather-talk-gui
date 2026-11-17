use feathertalk_core::*;
use serde::{Serialize, de::DeserializeOwned};
use serde_json::json;
fn roundtrip<T: Serialize + DeserializeOwned + PartialEq + std::fmt::Debug>(v: &T) {
    assert_eq!(
        &serde_json::from_str::<T>(&serde_json::to_string(v).unwrap()).unwrap(),
        v
    );
}
fn device() -> DeviceDescriptor {
    DeviceDescriptor {
        id: "cpu:0".into(),
        name: "CPU".into(),
        vendor: None,
        driver: None,
        backend: BackendKind::Cpu,
        memory_bytes: None,
    }
}
fn model() -> ModelRef {
    ModelRef {
        project_id: ProjectId::new(),
        run_id: TrainingRunId::new(),
        checkpoint_id: "epoch-5".into(),
    }
}
fn metrics() -> TrainingMetrics {
    TrainingMetrics {
        project_id: ProjectId::new(),
        run_id: TrainingRunId::new(),
        step: 42,
        epoch: 5,
        elapsed_ms: 300,
        lr: 0.001,
        losses: LossScalars {
            full: 1.,
            mouth: 2.,
            temporal: 3.,
            temporal_mouth: 4.,
            perceptual: 5.,
            mouth_gradient: 6.,
            total: 21.,
        },
        device: device(),
        precision: Precision::F32,
        batch_size: 8,
    }
}
#[test]
fn defaults_match_python() {
    let c = TrainConfig::default();
    assert_eq!(TrainConfig::default().batch_size, 8);
    assert_eq!(TrainConfig::default().epochs, 60);
    assert_eq!((c.lr, c.save_every), (0.001, 5));
    assert_eq!(
        (c.beta1, c.beta2, c.epsilon, c.weight_decay),
        (0.9, 0.999, 1e-8, 0.)
    );
    assert_eq!(
        c.loss,
        LossConfig {
            full: 1.,
            mouth: 4.,
            temporal: 0.5,
            temporal_mouth: 4.,
            perceptual: 0.01,
            mouth_gradient: 0.5,
            temporal_kernel: 5,
            temporal_stride: 1
        }
    );
    c.validate().unwrap();
}
#[test]
fn rejects_invalid_hyperparameters() {
    let defaults = serde_json::to_value(TrainConfig::default()).unwrap();
    for (key, value) in [
        ("batch_size", json!(0)),
        ("epochs", json!(0)),
        ("save_every", json!(0)),
        ("lr", json!(0)),
        ("beta1", json!(1)),
        ("beta2", json!(-0.1)),
        ("epsilon", json!(0)),
        ("weight_decay", json!(-1)),
    ] {
        let mut c = defaults.clone();
        c[key] = value;
        assert_eq!(
            serde_json::from_value::<TrainConfig>(c)
                .unwrap()
                .validate()
                .unwrap_err()
                .code,
            ErrorCode::InvalidConfig,
            "{key}"
        );
    }
    for (key, value) in [
        ("temporal_kernel", json!(4)),
        ("temporal_kernel", json!(0)),
        ("temporal_stride", json!(0)),
        ("full", json!(-1)),
    ] {
        let mut c = defaults.clone();
        c["loss"][key] = value;
        assert!(
            serde_json::from_value::<TrainConfig>(c)
                .unwrap()
                .validate()
                .is_err(),
            "{key}"
        );
    }
    let mut c = TrainConfig::default();
    c.lr = f64::NAN;
    assert!(c.validate().is_err());
    c = TrainConfig::default();
    c.loss.mouth = f64::INFINITY;
    assert!(c.validate().is_err());
}
#[test]
fn protocol_roundtrips() {
    let id = TaskId::new();
    let m = model();
    let payloads = vec![
        EventPayload::Started {
            job: JobKind::Train,
        },
        EventPayload::Progress {
            stage: "prepare".into(),
            completed: 1,
            total: Some(10),
        },
        EventPayload::Metrics(metrics()),
        EventPayload::CheckpointCommitted {
            model_ref: m.clone(),
        },
        EventPayload::Completed {
            result: ArtifactResult::Model(m.clone()),
        },
        EventPayload::Completed {
            result: ArtifactResult::Video(ExportResult {
                output_path: "out.mp4".into(),
                video_frames: 50,
                elapsed_ms: 100,
                device: device(),
            }),
        },
        EventPayload::Cancelled {
            checkpoint: Some(m.clone()),
        },
        EventPayload::Failed {
            error: AppError::new(ErrorCode::NoAudio, "prepare", "No audio"),
        },
    ];
    for (seq, payload) in payloads.into_iter().enumerate() {
        let event = WorkerEvent::new(Some(id), seq as u64, payload).unwrap();
        roundtrip(&event);
        let v = serde_json::to_value(&event).unwrap();
        assert_eq!(v["task_id"], json!(id));
        assert_eq!(v["protocol_version"], 1);
        let mut bad = v.clone();
        bad["task_id"] = json!(null);
        assert!(serde_json::from_value::<WorkerEvent>(bad).is_err());
        let mut bad = v;
        bad["protocol_version"] = json!(2);
        assert!(serde_json::from_value::<WorkerEvent>(bad).is_err());
    }
    let ready = EventPayload::Ready {
        capabilities: WorkerCapabilities {
            engine_version: "0.1.0".into(),
            components: vec![ComponentCapability {
                name: "feathertalk".into(),
                version: "1".into(),
                available: true,
            }],
            training: true,
            generation: true,
        },
        devices: vec![device()],
    };
    roundtrip(&WorkerEvent::new(None, 0, ready.clone()).unwrap());
    assert!(WorkerEvent::new(Some(id), 0, ready).is_err());
    assert_eq!(
        serde_json::to_value(metrics()).unwrap()["losses"],
        json!({"full":1.,"mouth":2.,"temporal":3.,"temporal_mouth":4.,"perceptual":5.,"mouth_gradient":6.,"total":21.})
    );
    let requests = vec![
        WorkerRequest::Hello {
            protocol_version: 1,
        },
        WorkerRequest::Start {
            task_id: id,
            job: JobSpec::Train(TrainRequest {
                project_id: m.project_id,
                config: TrainConfig::default(),
                device: DevicePolicy::Auto,
                resume: Some(m.clone()),
            }),
        },
        WorkerRequest::Start {
            task_id: id,
            job: JobSpec::Generate(GenerateRequest {
                project_id: m.project_id,
                audio_path: "a.wav".into(),
                output_path: "o.mp4".into(),
                model_ref: m,
                device: DevicePolicy::Explicit(DeviceRequest {
                    backend: BackendKind::Vulkan,
                    device_id: "gpu:0".into(),
                }),
            }),
        },
        WorkerRequest::Cancel { task_id: id },
        WorkerRequest::Shutdown,
    ];
    for r in requests {
        roundtrip(&r);
    }
    assert_eq!(
        serde_json::to_value(ErrorCode::OutOfMemory).unwrap(),
        "out_of_memory"
    );
}
#[test]
fn project_accepts_one_video() {
    let v = json!({"name":"Alice","directory":"projects/alice","source_video":"alice.mp4"});
    let p: NewProject = serde_json::from_value(v.clone()).unwrap();
    p.validate().unwrap();
    roundtrip(&p);
    let mut bad = v.clone();
    bad["source_video"] = json!(["a.mp4", "b.mp4"]);
    assert!(serde_json::from_value::<NewProject>(bad).is_err());
    let mut bad = v;
    bad["source_videos"] = json!(["a.mp4", "b.mp4"]);
    assert!(serde_json::from_value::<NewProject>(bad).is_err());
}
#[test]
fn cancellation_is_shared_and_observer_is_object_safe() {
    let c = Cancellation::new();
    let other = c.clone();
    assert!(!other.is_cancelled());
    c.cancel();
    assert!(other.is_cancelled());
    struct Sink;
    impl TaskObserver for Sink {
        fn emit(&self, _: EventPayload) -> Result<()> {
            Ok(())
        }
    }
    let observer: &dyn TaskObserver = &Sink;
    observer
        .emit(EventPayload::Cancelled { checkpoint: None })
        .unwrap();
}

#[test]
fn published_protocol_examples_match_rust_contracts() {
    let schema: serde_json::Value =
        serde_json::from_str(include_str!("../../../contracts/protocol-v1.json")).unwrap();
    let examples = schema["examples"]
        .as_array()
        .expect("published protocol examples");
    assert!(examples.len() >= 3);
    for example in examples {
        if example.get("payload").is_some() {
            let event: WorkerEvent = serde_json::from_value(example.clone()).unwrap();
            roundtrip(&event);
        } else {
            let request: WorkerRequest = serde_json::from_value(example.clone()).unwrap();
            roundtrip(&request);
        }
    }
}
#[test]
fn persistence_and_query_contracts_roundtrip() {
    let m = model();
    let identity = ModelIdentity {
        kind: "feathertalk".into(),
        architecture_version: "1".into(),
        source_sha256: "a".repeat(64),
        weights_sha256: "b".repeat(64),
    };
    let manifest = ProjectManifest {
        schema_version: 1,
        id: m.project_id,
        name: "Alice".into(),
        source_video: SourceVideo {
            path: "source.mp4".into(),
            sha256: "c".repeat(64),
            size_bytes: 1024,
        },
        preprocessing: Some(PreprocessingIdentity {
            version: "1".into(),
            source_sha256: "c".repeat(64),
            config_sha256: "d".repeat(64),
            cache_directory: "cache/v1".into(),
            component_identities: vec![identity.clone()],
        }),
        model_identity: Some(identity.clone()),
        checkpoints: vec![CheckpointEntry {
            model_ref: m.clone(),
            relative_path: "checkpoints/epoch-5".into(),
            sha256: "b".repeat(64),
            size_bytes: 256,
            epoch: 5,
            step: 42,
            config: TrainConfig::default(),
            identity: identity.clone(),
        }],
        metrics_path: "metrics.jsonl".into(),
    };
    roundtrip(&manifest);
    roundtrip(&ProjectSummary {
        id: m.project_id,
        name: manifest.name.clone(),
        directory: "alice".into(),
        source_video: "source.mp4".into(),
        latest_model: Some(m.clone()),
    });
    roundtrip(&AssetManifest {
        identity,
        relative_files: vec!["model.bin".into()],
        sizes: [("model.bin".into(), 256)].into(),
        hashes: [("model.bin".into(), "b".repeat(64))].into(),
        urls: [(
            "model.bin".into(),
            vec!["https://example.org/model.bin".into()],
        )]
        .into(),
        min_engine_version: "0.1.0".into(),
        provenance: "Converted from verified upstream weights".into(),
    });
    roundtrip(&TaskSnapshot {
        task_id: TaskId::new(),
        project_id: m.project_id,
        job: JobKind::Train,
        status: TaskStatus::Completed,
        last_seq: 10,
        latest_event: None,
        result: Some(ArtifactResult::Model(m.clone())),
        error: None,
    });
    roundtrip(&MetricsPageRequest {
        project_id: m.project_id,
        run_id: m.run_id,
        offset: 0,
        limit: 50,
    });
    roundtrip(&MetricsPage {
        project_id: m.project_id,
        run_id: m.run_id,
        items: vec![metrics()],
        next_offset: Some(1),
    });
    assert!(serde_json::from_value::<TaskId>(json!("not-a-uuid")).is_err());
    assert!(serde_json::from_value::<ProjectId>(json!("not-a-uuid")).is_err());
    assert!(serde_json::from_value::<TrainingRunId>(json!("not-a-uuid")).is_err());
}

#[test]
fn invalid_event_cannot_be_serialized() {
    let mut event = WorkerEvent::new(
        Some(TaskId::new()),
        1,
        EventPayload::Cancelled { checkpoint: None },
    )
    .unwrap();
    event.task_id = None;
    assert!(serde_json::to_string(&event).is_err());
    event.task_id = Some(TaskId::new());
    event.protocol_version = 2;
    assert!(serde_json::to_string(&event).is_err());
}
