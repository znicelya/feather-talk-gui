#[allow(dead_code)]
mod training_fixture;
use feathertalk_core::*;
use feathertalk_engine::{
    data::{
        prepare_project,
        prepared::{file_hash, reopen_generation},
    },
    inference::{generate, renderer::FrameRenderer},
    media::MediaTools,
    training::{
        checkpoint::{load_checkpoint, save_checkpoint},
        state::TrainingState,
    },
};
use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
};
struct Observe;
struct CancelAfterFrame(Cancellation);
impl TaskObserver for CancelAfterFrame {
    fn emit(&self, event: EventPayload) -> Result<()> {
        if matches!(event,EventPayload::Progress{ref stage,completed:1,..} if stage=="render") {
            self.0.cancel();
        }
        Ok(())
    }
}
impl TaskObserver for Observe {
    fn emit(&self, e: EventPayload) -> Result<()> {
        eprintln!("{e:?}");
        Ok(())
    }
}
fn tools(base: &Path) -> MediaTools {
    let bin = base.join(".superpowers/toolchain/ffmpeg/ffmpeg-9.0.2-essentials_build/bin");
    MediaTools {
        ffmpeg: bin.join("ffmpeg.exe"),
        ffprobe: bin.join("ffprobe.exe"),
    }
}
#[test]
fn actual_demo_offline_same_checkpoint() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let tools = tools(&base);
    let device = training_fixture::device();
    let backend = format!("{:?}", device.descriptor.backend);
    eprintln!("ACTUAL DEVICE {:?}", device.descriptor);
    let root = base.join(".artifacts/task10-demo-project");
    fs::create_dir_all(&root).unwrap();
    let root = fs::canonicalize(root).unwrap();
    let assets = base.join(".artifacts/assets");
    if !root.join("project.json").exists() {
        fs::copy(
            base.join(".artifacts/integration-media/demo-right-4frames.mkv"),
            root.join("source.mkv"),
        )
        .unwrap();
        let p = ProjectManifest {
            schema_version: 1,
            id: ProjectId::new(),
            name: "internal derived demo functional fixture".into(),
            source_video: SourceVideo {
                path: "source.mkv".into(),
                sha256: file_hash(&root.join("source.mkv")).unwrap(),
                size_bytes: fs::metadata(root.join("source.mkv")).unwrap().len(),
            },
            preprocessing: None,
            model_identity: None,
            checkpoints: vec![],
            metrics_path: "metrics.jsonl".into(),
        };
        fs::write(root.join("project.json"), serde_json::to_vec(&p).unwrap()).unwrap();
    }
    let p: ProjectManifest =
        serde_json::from_slice(&fs::read(root.join("project.json")).unwrap()).unwrap();
    let prepared = if p.checkpoints.is_empty() {
        let prepared = prepare_project(
            &root,
            &p,
            &assets,
            &tools,
            &device,
            &Observe,
            &Cancellation::new(),
        )
        .unwrap();
        eprintln!(
            "ACTUAL PREPARED resources={} features={}",
            prepared.canonical_frame_count, prepared.feature_frame_count
        );
        let request = TrainRequest {
            project_id: p.id,
            config: TrainConfig {
                batch_size: 1,
                ..Default::default()
            },
            device: DevicePolicy::Auto,
            resume: None,
            batch_mode: BatchMode::Fixed,
        };
        let state = TrainingState::fresh(&request, &prepared, &device).unwrap();
        fs::create_dir_all(root.join("checkpoints")).unwrap();
        save_checkpoint(&state, &root.join("checkpoints/initial")).unwrap();
        prepared
    } else {
        let reference = &p.checkpoints[0].model_ref;
        let state = load_checkpoint(&root, reference, &device).unwrap();
        reopen_generation(
            &root,
            &p,
            &state.metadata.preprocessing.cache_directory,
            &state.metadata.prepared_manifest_sha256,
        )
        .unwrap()
    };
    let p: ProjectManifest =
        serde_json::from_slice(&fs::read(root.join("project.json")).unwrap()).unwrap();
    let mut reference = p.checkpoints.last().unwrap().model_ref.clone();
    let mut state = load_checkpoint(&root, &reference, &device).unwrap();
    if state.metadata.global_step == 0 {
        assert_eq!(
            backend, "Cuda",
            "Run CUDA once to create a trained common checkpoint"
        );
        state.set_learning_rate(0.001).unwrap();
        let vgg = feathertalk_engine::models::vgg::VggFeatures::load(
            &assets.join("vgg19"),
            &device.device.clone().autodiff(),
        )
        .unwrap();
        let _metrics = feathertalk_engine::training::trainer::complete_step(
            &mut state,
            &prepared,
            &vgg,
            &device,
            &Cancellation::new(),
        )
        .unwrap();
        eprintln!("ACTUAL TRAINED STEP {}", state.metadata.global_step);
        reference = save_checkpoint(&state, &root.join("checkpoints/step1")).unwrap();
    }
    eprintln!(
        "COMMON CHECKPOINT {:?} step={}",
        reference, state.metadata.global_step
    );
    drop(state);
    let output = root.join(format!("{backend}.mp4"));
    let request = GenerateRequest {
        project_id: p.id,
        audio_path: root.join("source.mkv"),
        output_path: output,
        model_ref: reference.clone(),
        device: DevicePolicy::Auto,
    };
    let result = generate(
        &request,
        &prepared,
        &assets,
        &tools,
        &device,
        &Observe,
        &Cancellation::new(),
    )
    .unwrap();
    eprintln!("ACTUAL EXPORT {result:?}");
    assert_eq!(result.video_frames, 3);
    if backend == "Cuda" {
        let old = fs::read(&result.output_path).unwrap();
        let cancel = Cancellation::new();
        let failure = generate(
            &request,
            &prepared,
            &assets,
            &tools,
            &device,
            &CancelAfterFrame(cancel.clone()),
            &cancel,
        )
        .unwrap_err();
        assert_eq!(failure.code, ErrorCode::Cancelled);
        assert_eq!(fs::read(&result.output_path).unwrap(), old);
        assert!(!fs::read_dir(&root).unwrap().any(|e| {
            e.unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".export-")
        }));
    }
    // Exact common canonical input and common stored features isolate character device parity.
    let raw = root.join("canonical.bgr");
    if !raw.exists() {
        let status = Command::new(&tools.ffmpeg)
            .args(["-nostdin", "-y", "-v", "error", "-i"])
            .arg(&prepared.canonical_video)
            .args(["-pix_fmt", "bgr24", "-f", "rawvideo"])
            .arg(&raw)
            .status()
            .unwrap();
        assert!(status.success());
    }
    let state = load_checkpoint(&root, &reference, &device).unwrap();
    let mut renderer = FrameRenderer::new(
        state.model,
        fs::File::open(raw).unwrap(),
        640,
        600,
        prepared
            .frames
            .iter()
            .map(|f| feathertalk_engine::data::landmarks::Landmarks {
                points: f.landmarks.clone().try_into().unwrap(),
            })
            .collect(),
    )
    .unwrap();
    let features = feathertalk_engine::data::audio::AudioFeatures {
        identity: ModelIdentity {
            kind: "test".into(),
            architecture_version: "test".into(),
            source_sha256: "test".into(),
            weights_sha256: "test".into(),
        },
        video_frames: prepared.feature_frame_count,
        values: (*prepared.features).clone(),
    };
    let mut all = vec![];
    for k in 0..features.video_frames as u64 {
        all.extend(renderer.render(k, &features).unwrap().data);
    }
    fs::write(root.join(format!("{backend}.bgr")), &all).unwrap();
    fs::write(
        root.join(format!("{backend}-checkpoint.json")),
        serde_json::to_vec(&reference).unwrap(),
    )
    .unwrap();
    let other = if backend == "Flex" { "Cuda" } else { "Flex" };
    if root.join(format!("{other}.bgr")).exists() {
        let other_reference: ModelRef = serde_json::from_slice(
            &fs::read(root.join(format!("{other}-checkpoint.json"))).unwrap(),
        )
        .unwrap();
        assert_eq!(other_reference, reference);
        let cpu = fs::read(root.join(format!("{other}.bgr"))).unwrap();
        assert_eq!(cpu.len(), all.len());
        let max = cpu
            .iter()
            .zip(&all)
            .map(|(a, b)| a.abs_diff(*b))
            .max()
            .unwrap();
        let mean = cpu
            .iter()
            .zip(&all)
            .map(|(a, b)| a.abs_diff(*b) as f64)
            .sum::<f64>()
            / all.len() as f64;
        eprintln!("COMMON INPUT PASTE max={max} mean={mean}");
        assert!(max <= 2 && mean <= 0.1);
    }
}
#[test]
fn step_zero_checkpoint_requires_training() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let (root, prepared) = training_fixture::fixture();
    let device = training_fixture::device();
    let request = TrainRequest {
        project_id: prepared.project_id,
        config: TrainConfig {
            batch_size: 1,
            ..Default::default()
        },
        device: DevicePolicy::Auto,
        resume: None,
        batch_mode: BatchMode::Fixed,
    };
    let state = TrainingState::fresh(&request, &prepared, &device).unwrap();
    let reference = save_checkpoint(&state, &root.join("checkpoints/initial")).unwrap();

    let request = GenerateRequest {
        project_id: prepared.project_id,
        audio_path: root.join("deliberately-missing-audio"),
        output_path: root.join("untrained.mp4"),
        model_ref: reference,
        device: DevicePolicy::Auto,
    };
    let e = generate(
        &request,
        &prepared,
        &base.join(".artifacts/assets"),
        &tools(&base),
        &device,
        &Observe,
        &Cancellation::new(),
    )
    .unwrap_err();
    assert_eq!(e.code, ErrorCode::InvalidConfig);
    assert!(e.message.contains("Train"));
}
