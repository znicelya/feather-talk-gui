pub mod export;
mod process;
pub mod renderer;
pub mod session;
use crate::{
    ExecutionDevice,
    data::{PreparedProject, audio::extract_audio_features_cancellable, landmarks::Landmarks},
    media::{
        MediaTools,
        ffmpeg::{check, error},
    },
    models::feather_hubert::FeatherHubert,
    training::checkpoint::load_checkpoint,
};
use feathertalk_core::*;
use session::{AudioSource, FrameSink};
use std::{path::Path, process::Command, time::Instant};
/// Offline, whole-utterance feature extraction followed by ordered eval and atomic publication.
pub fn generate(
    request: &GenerateRequest,
    prepared: &PreparedProject,
    assets: &Path,
    tools: &MediaTools,
    device: &ExecutionDevice,
    observer: &dyn TaskObserver,
    cancel: &Cancellation,
) -> Result<ExportResult> {
    let start = Instant::now();
    check(cancel)?;
    prepared.validate()?;
    if request.project_id != prepared.project_id
        || request.model_ref.project_id != prepared.project_id
    {
        return Err(error("Generation project/model/prepared identity mismatch"));
    }
    if let DevicePolicy::Explicit(d) = &request.device {
        if d.device_id != device.descriptor.id || d.backend != device.descriptor.backend {
            return Err(error(
                "Generation device differs from selected execution device",
            ));
        }
    }
    let state = load_checkpoint(&prepared.project_directory, &request.model_ref, device)?;
    if state.metadata.global_step == 0 {
        let mut error = AppError::new(
            ErrorCode::InvalidConfig,
            "generate",
            "Train the character before generating video; this checkpoint has no completed training step",
        );
        error
            .actions
            .push("Complete at least one training step and select its committed checkpoint".into());
        return Err(error);
    }
    state.validate_prepared(prepared)?;
    let _ = renderer::FramePicker::new(prepared.canonical_frame_count)?;
    let parent = request
        .output_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let stage = process::Staging::new(parent)?;
    let pcm = stage.0.join("audio.f32");
    let mut command = Command::new(&tools.ffmpeg);
    command
        .args(["-nostdin", "-y", "-v", "error", "-i"])
        .arg(&request.audio_path)
        .args([
            "-map",
            "0:a:0",
            "-vn",
            "-af",
            "pan=mono|c0=c0,aresample=16000",
            "-c:a",
            "pcm_f32le",
            "-f",
            "f32le",
        ])
        .arg(&pcm);
    process::run(&mut command, cancel)?;
    let mut source = session::FileAudioSource::open(&pcm)?;
    let mut samples = vec![];
    while let Some(batch) = source.read(session::AUDIO_BATCH_LIMIT)? {
        check(cancel)?;
        samples.extend(batch.samples);
    }
    source.finish()?;
    observer.emit(EventPayload::Progress {
        stage: "audio-features".into(),
        completed: 0,
        total: None,
    })?;
    let hubert = FeatherHubert::load(&assets.join("feather_hubert"), &device.device)?;
    if !prepared
        .preprocessing
        .component_identities
        .iter()
        .any(|identity| identity == hubert.identity())
    {
        return Err(AppError::new(
            ErrorCode::ResourceCorrupt,
            "generate",
            "Driving-audio HuBERT identity differs from the checkpoint's prepared dependency",
        ));
    }
    let features = extract_audio_features_cancellable(&hubert, &samples, cancel)?;
    drop(hubert);
    drop(samples);
    if features.video_frames == 0 {
        return Err(error(
            "Driving audio produces zero video frames; supply longer audio",
        ));
    }
    let mut probe = Command::new(&tools.ffprobe);
    probe
        .args(["-v", "error", "-show_streams", "-of", "json"])
        .arg(&prepared.canonical_video);
    let v: serde_json::Value =
        serde_json::from_slice(&process::run(&mut probe, cancel)?).map_err(error)?;
    let streams = v["streams"]
        .as_array()
        .ok_or_else(|| error("Canonical streams absent"))?;
    let video = streams
        .iter()
        .find(|s| s["codec_type"] == "video")
        .ok_or_else(|| error("Canonical video absent"))?;
    let width = video["width"].as_u64().unwrap_or(0) as usize;
    let height = video["height"].as_u64().unwrap_or(0) as usize;
    if width == 0 || height == 0 || width > 16384 || height > 16384 {
        return Err(error("Unsupported canonical canvas"));
    }
    let raw = stage.0.join("resources.bgr");
    let mut decode = Command::new(&tools.ffmpeg);
    decode
        .args(["-nostdin", "-y", "-v", "error", "-i"])
        .arg(&prepared.canonical_video)
        .args([
            "-map", "0:v:0", "-an", "-pix_fmt", "bgr24", "-f", "rawvideo",
        ])
        .arg(&raw);
    process::run(&mut decode, cancel)?;
    let landmarks = prepared
        .frames
        .iter()
        .map(|f| {
            Ok(Landmarks {
                points: f
                    .landmarks
                    .clone()
                    .try_into()
                    .map_err(|_| error("Prepared landmarks require 110 points"))?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let mut renderer = renderer::FrameRenderer::new(
        state.model,
        std::fs::File::open(raw).map_err(error)?,
        width,
        height,
        landmarks,
    )?;
    let mut sink = export::FileFrameSink::new(
        &request.output_path,
        &pcm,
        width,
        height,
        features.video_frames as u64,
        tools,
    )?;
    for k in 0..features.video_frames as u64 {
        check(cancel)?;
        let frame = renderer.render(k, &features)?;
        sink.write(
            session::TimestampedFrame {
                pts: k,
                time_base: session::VIDEO_TIME_BASE,
                frame,
            },
            cancel,
        )?;
        observer.emit(EventPayload::Progress {
            stage: "render".into(),
            completed: k + 1,
            total: Some(features.video_frames as u64),
        })?;
    }
    let output_path = sink.finish(cancel)?;
    Ok(ExportResult {
        output_path,
        video_frames: features.video_frames as u64,
        elapsed_ms: start.elapsed().as_millis() as u64,
        device: device.descriptor.clone(),
    })
}
