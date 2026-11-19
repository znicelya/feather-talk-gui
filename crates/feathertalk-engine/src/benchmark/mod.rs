//! Run the same prepared input and committed checkpoint on a named device.
mod policy;
mod report;
pub use policy::select_backend;
pub use report::{BenchmarkReport, BenchmarkSample, WorkloadIdentity};

use crate::{
    ExecutionDevice,
    data::{
        prepare_project,
        prepared::{file_hash, reopen_generation},
    },
    inference::generate,
    media::MediaTools,
    models::vgg::VggFeatures,
    open_device, run_operator_checks,
    training::{
        checkpoint::{MaterializedSnapshot, load_checkpoint},
        trainer::complete_step,
    },
};
use feathertalk_core::*;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    path::{Component, Path, PathBuf},
    sync::Mutex,
    time::Instant,
};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkSpec {
    pub project: PathBuf,
    pub checkpoint: String,
    pub assets: PathBuf,
    pub audio: PathBuf,
    pub ffmpeg: PathBuf,
    pub ffprobe: PathBuf,
    pub reference: PathBuf,
    pub scratch: PathBuf,
    pub device: DeviceRequest,
    pub precision: Precision,
    pub batch_size: u32,
    pub prefetch_capacity: u32,
    pub sample_loss: bool,
    pub warmup: u32,
    pub repetitions: u32,
}

fn invalid(e: impl ToString) -> AppError {
    AppError::new(ErrorCode::InvalidConfig, "benchmark", e.to_string())
}

fn elapsed(start: Instant) -> u64 {
    start.elapsed().as_millis() as u64
}

fn inside(root: &Path, relative: &Path) -> Result<PathBuf> {
    if relative.as_os_str().is_empty()
        || relative
            .components()
            .any(|part| !matches!(part, Component::Normal(_)))
    {
        return Err(invalid(
            "Benchmark fixture must have project-relative inputs",
        ));
    }
    let path = root.join(relative);
    if fs::canonicalize(&path).map_err(invalid)? != path {
        return Err(invalid("Benchmark fixture contains a redirected path"));
    }
    Ok(path)
}

fn new_root(base: &Path) -> Result<Scratch> {
    let base = fs::canonicalize(base).map_err(invalid)?;
    let root = base.join(format!("feathertalk-benchmark-{}", TaskId::new()));
    fs::create_dir(&root).map_err(invalid)?;
    Ok(Scratch(root))
}

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn copy_file(from: &Path, to: &Path) -> Result<()> {
    if !fs::symlink_metadata(from).map_err(invalid)?.is_file() {
        return Err(invalid("Benchmark input must be a regular file"));
    }
    let parent = to
        .parent()
        .ok_or_else(|| invalid("Missing scratch parent"))?;
    fs::create_dir_all(parent).map_err(invalid)?;
    fs::copy(from, to).map_err(invalid)?;
    Ok(())
}

fn copy_tree(from: &Path, to: &Path) -> Result<()> {
    fs::create_dir_all(to).map_err(invalid)?;
    for entry in fs::read_dir(from).map_err(invalid)? {
        let entry = entry.map_err(invalid)?;
        let kind = entry.file_type().map_err(invalid)?;
        let target = to.join(entry.file_name());
        if kind.is_dir() {
            copy_tree(&entry.path(), &target)?;
        } else {
            copy_file(&entry.path(), &target)?;
        }
    }
    Ok(())
}

struct Inputs {
    project: ProjectManifest,
    model: ModelRef,
    checkpoint_manifest: PathBuf,
    tools: MediaTools,
    pinned_generation: PathBuf,
    pinned_sha256: String,
}

impl Inputs {
    fn stage(spec: &BenchmarkSpec, sandbox: &Path) -> Result<Self> {
        let project_root = fs::canonicalize(&spec.project).map_err(invalid)?;
        if sandbox.starts_with(&project_root) {
            return Err(invalid(
                "Place benchmark scratch outside the source project",
            ));
        }
        let mut project = crate::training::checkpoint::read_project(&project_root)?;
        let chosen = project
            .checkpoints
            .iter()
            .find(|entry| entry.model_ref.checkpoint_id == spec.checkpoint)
            .filter(|entry| entry.step > 0)
            .ok_or_else(|| invalid("Choose a committed trained checkpoint"))?
            .clone();
        let manifest = inside(&project_root, &chosen.relative_path)?;
        if manifest.file_name().and_then(|name| name.to_str()) != Some("manifest.json") {
            return Err(invalid("Expected checkpoint manifest.json"));
        }
        let checkpoint = manifest
            .parent()
            .ok_or_else(|| invalid("Missing checkpoint directory"))?;
        let snapshot = MaterializedSnapshot::read_private(checkpoint)?;
        let metadata = snapshot.metadata();
        let generation = metadata.preprocessing.cache_directory.clone();
        let prepared = inside(&project_root, &generation)?;
        // Desktop projects may reference an absolute source outside the project.
        // Keep that input read-only; rebasing it would invalidate the checkpoint.
        if project.source_video.path.is_absolute() {
            if !fs::metadata(&project.source_video.path)
                .map_err(invalid)?
                .is_file()
            {
                return Err(invalid("Benchmark source must be a regular file"));
            }
        } else {
            let source = inside(&project_root, &project.source_video.path)?;
            copy_file(&source, &sandbox.join(&project.source_video.path))?;
        }
        for name in ["manifest.json", "model.bpk", "adam.bpk"] {
            copy_file(
                &checkpoint.join(name),
                &sandbox
                    .join(checkpoint.strip_prefix(&project_root).map_err(invalid)?)
                    .join(name),
            )?;
        }
        copy_tree(&prepared, &sandbox.join(&generation))?;
        project.checkpoints = vec![chosen.clone()];
        fs::write(
            sandbox.join("project.json"),
            serde_json::to_vec(&project).map_err(invalid)?,
        )
        .map_err(invalid)?;
        let tools = MediaTools {
            ffmpeg: fs::canonicalize(&spec.ffmpeg).map_err(invalid)?,
            ffprobe: fs::canonicalize(&spec.ffprobe).map_err(invalid)?,
        };
        Ok(Self {
            project,
            model: chosen.model_ref,
            checkpoint_manifest: sandbox.join(chosen.relative_path),
            tools,
            pinned_generation: generation,
            pinned_sha256: metadata.prepared_manifest_sha256.clone(),
        })
    }

    fn identity(&self, spec: &BenchmarkSpec, root: &Path) -> Result<WorkloadIdentity> {
        Ok(WorkloadIdentity {
            engine_sha256: file_hash(&std::env::current_exe().map_err(invalid)?)?,
            source_sha256: file_hash(&root.join(&self.project.source_video.path))?,
            audio_sha256: file_hash(&spec.audio)?,
            checkpoint_sha256: file_hash(&self.checkpoint_manifest)?,
            prepared_sha256: file_hash(&root.join(&self.pinned_generation).join("manifest.json"))?,
            assets_sha256: file_hash(&spec.assets.join("manifest.json"))?,
            ffmpeg_sha256: file_hash(&self.tools.ffmpeg)?,
            ffprobe_sha256: file_hash(&self.tools.ffprobe)?,
            reference_sha256: file_hash(&spec.reference.join("manifest.json"))?,
            os: std::env::consts::OS.into(),
            architecture: std::env::consts::ARCH.into(),
            precision: spec.precision,
            requested_batch: spec.batch_size,
            prefetch_capacity: spec.prefetch_capacity,
            sample_loss: spec.sample_loss,
        })
    }
}

#[derive(Default)]
struct StageTimes {
    audio_features: Option<Instant>,
    render: Option<Instant>,
    encode: Option<Instant>,
}
struct MeasureExport<'a> {
    observer: &'a dyn TaskObserver,
    times: Mutex<StageTimes>,
}
impl TaskObserver for MeasureExport<'_> {
    fn emit(&self, event: EventPayload) -> Result<()> {
        if let EventPayload::Progress {
            stage,
            completed: 0,
            ..
        } = &event
        {
            let mut times = self.times.lock().map_err(invalid)?;
            let to = match stage.as_str() {
                "audio-features" => Some(&mut times.audio_features),
                "render" => Some(&mut times.render),
                "encode" => Some(&mut times.encode),
                _ => None,
            };
            if let Some(to) = to {
                *to = Some(Instant::now());
            }
        }
        self.observer.emit(event)
    }
}

fn trial(
    spec: &BenchmarkSpec,
    input: &Inputs,
    root: &Path,
    device: &ExecutionDevice,
    observer: &dyn TaskObserver,
) -> Result<BenchmarkSample> {
    let cancel = Cancellation::new();
    let start = Instant::now();
    {
        let _prepared = prepare_project(
            root,
            &input.project,
            &spec.assets,
            &input.tools,
            device,
            observer,
            &cancel,
        )?;
    }
    let prepare_ms = elapsed(start);

    let start = Instant::now();
    let prepared = reopen_generation(
        root,
        &input.project,
        &input.pinned_generation,
        &input.pinned_sha256,
    )?;
    let mut state = load_checkpoint(root, &input.model, device)?;
    let checkpoint_load_ms = elapsed(start);
    let vgg = VggFeatures::load(
        &spec.assets.join("vgg19"),
        &device.device.clone().autodiff(),
    )?;
    let epochs = state
        .metadata
        .epoch
        .checked_add(1)
        .ok_or_else(|| invalid("Checkpoint epoch overflow"))?;
    state.metadata.config.epochs = state.metadata.config.epochs.max(epochs);
    state.metadata.config.prefetch_capacity = spec.prefetch_capacity;
    state.metadata.config.batch_size = spec.batch_size;
    state.metadata.requested_batch = spec.batch_size;
    state.metadata.actual_batch = spec.batch_size;
    state.metadata.batch_mode = BatchMode::Fixed;
    let start = Instant::now();
    let updated = complete_step(&mut state, &prepared, &vgg, device, &cancel)?;
    let total = spec.sample_loss.then(|| updated.losses.to_scalars().total);
    let training_step_ms = elapsed(start);
    let training_pairs = updated.pairs.len() as u32;
    drop(updated);
    drop(vgg);
    drop(state);

    let measure = MeasureExport {
        observer,
        times: Mutex::new(StageTimes::default()),
    };
    let start = Instant::now();
    let video = generate(
        &GenerateRequest {
            project_id: input.project.id,
            model_ref: input.model.clone(),
            audio_path: spec.audio.clone(),
            output_path: root.join("output.mp4"),
            device: DevicePolicy::Explicit(spec.device.clone()),
        },
        &prepared,
        &spec.assets,
        &input.tools,
        device,
        &measure,
        &cancel,
    )?;
    let end = Instant::now();
    let phases = measure.times.into_inner().map_err(invalid)?;
    let between = |a: Option<Instant>, b: Option<Instant>| {
        a.zip(b)
            .map(|(a, b)| b.duration_since(a).as_millis() as u64)
    };
    Ok(BenchmarkSample {
        prepare_ms,
        checkpoint_load_ms,
        training_step_ms,
        training_pairs,
        training_total_loss: total,
        audio_features_ms: between(phases.audio_features, phases.render),
        render_ms: between(phases.render, phases.encode),
        encode_ms: between(phases.encode, Some(end)),
        generation_ms: end.duration_since(start).as_millis() as u64,
        video_frames: video.video_frames,
    })
}

pub fn run_benchmark(spec: &BenchmarkSpec, observer: &dyn TaskObserver) -> Result<BenchmarkReport> {
    if spec.precision != Precision::F32
        || !(1..=1024).contains(&spec.batch_size)
        || spec.prefetch_capacity > 32
        || !(1..=16).contains(&spec.warmup)
        || !(3..=32).contains(&spec.repetitions)
    {
        return Err(invalid(
            "Use FP32, batch 1–1024, prefetch 0–32, 1–16 warmup runs and 3–32 timed repetitions",
        ));
    }
    let sandbox = new_root(&spec.scratch)?;
    let input = Inputs::stage(spec, &sandbox.0)?;
    let workload = input.identity(spec, &sandbox.0)?;
    let start = Instant::now();
    let device = open_device(&spec.device)?;
    let device_init_ms = elapsed(start);
    let mut report = BenchmarkReport {
        schema_version: 1,
        workload,
        device: device.descriptor.clone(),
        device_init_ms,
        warmup: spec.warmup,
        ..Default::default()
    };
    let run = |stage: &str, index, total| {
        observer.emit(EventPayload::Progress {
            stage: stage.into(),
            completed: index,
            total: Some(total),
        })?;
        trial(spec, &input, &sandbox.0, &device, observer)
    };
    report.cold_runs.push(run("benchmark-cold", 0, 1)?);
    for i in 0..spec.warmup {
        run("benchmark-warmup", u64::from(i), u64::from(spec.warmup))?;
    }
    for i in 0..spec.repetitions {
        report.warm_runs.push(run(
            "benchmark-warm",
            u64::from(i),
            u64::from(spec.repetitions),
        )?);
    }
    let checks = run_operator_checks(&device, &spec.reference)?;
    report.validation_failures = checks
        .checks
        .into_iter()
        .filter(|check| !check.passed)
        .map(|check| check.name)
        .collect();
    if input.identity(spec, &sandbox.0)? != report.workload {
        report
            .validation_failures
            .push("Input identities changed during the benchmark".into());
    }
    report.validation_passed = report.validation_failures.is_empty();
    Ok(report)
}
