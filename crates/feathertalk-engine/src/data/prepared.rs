//! Immutable generations; index.json is the only commit point. Original media is read-only.
use super::{
    audio::extract_audio_features_cancellable,
    geometry::{PreparedFrame, invalid, prepare_frame},
    landmarks::{BgrFrame, load_mean_face, locate_face},
};
use crate::{
    ExecutionDevice,
    media::{
        MediaTools,
        ffmpeg::{self, check},
        probe::probe,
    },
    models::{feather_hubert::FeatherHubert, pfld::PfldGhostOne, scrfd::Scrfd},
    verified_assets::VerifiedAssets,
};
use feathertalk_core::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    path::{Component, Path, PathBuf},
    sync::Arc,
};
const POLICY: &str = "prepare-v2-dependency-cache;ffv1-bgr24-no-jpeg;first-presented-video-origin;fps25;first-channel-f32-16k;crop152-inner144-roi-source;hubert-whole-normalize";
pub(crate) fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub fn file_hash(path: &Path) -> Result<String> {
    let mut f = fs::File::open(path).map_err(invalid)?;
    let mut h = Sha256::new();
    let mut buf = [0; 65536];
    loop {
        let n = f.read(&mut buf).map_err(invalid)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
fn preparation_fingerprint(
    project: &ProjectManifest,
    manifests: &[AssetManifest],
    tools: &MediaTools,
    device: &DeviceDescriptor,
) -> Result<String> {
    Ok(digest(
        &serde_json::to_vec(&(
            POLICY,
            env!("CARGO_PKG_VERSION"),
            project.id,
            &project.source_video.sha256,
            manifests,
            file_hash(&tools.ffmpeg)?,
            file_hash(&tools.ffprobe)?,
            device,
        ))
        .map_err(invalid)?,
    ))
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FrameRecord {
    pub file: PathBuf,
    pub sha256: String,
    pub landmarks: Vec<[i32; 2]>,
}
#[derive(Clone)]
pub struct PreparedProject {
    pub project_directory: PathBuf,
    pub project_id: ProjectId,
    pub identity: String,
    pub preprocessing: PreprocessingIdentity,
    pub canonical_video: PathBuf,
    pub canonical_audio: PathBuf,
    pub canonical_frame_count: usize,
    pub feature_frame_count: usize,
    pub usable_frames: usize,
    pub frames: Vec<FrameRecord>,
    pub features: Arc<Vec<f32>>,
    pub(crate) generation: PathBuf,
}
impl PreparedProject {
    pub fn validate(&self) -> Result<()> {
        if self.usable_frames != self.frames.len().min(self.features.len() / 2048)
            || self.feature_frame_count.checked_mul(2048) != Some(self.features.len())
            || self.canonical_frame_count != self.frames.len()
            || self.features.iter().any(|v| !v.is_finite())
        {
            return Err(invalid("Invalid prepared dataset cardinality"));
        }
        Ok(())
    }
    pub fn read_frame(&self, index: usize) -> Result<PreparedFrame> {
        let record = self
            .frames
            .get(index)
            .ok_or_else(|| invalid("Frame index out of bounds"))?;
        safe_relative(&record.file)?;
        let path = self.generation.join(&record.file);
        if !fs::canonicalize(&path)
            .map_err(invalid)?
            .starts_with(&self.project_directory)
        {
            return Err(invalid("Frame path escapes project root"));
        }
        let bytes = fs::read(path).map_err(invalid)?;
        if digest(&bytes) != record.sha256 || bytes.len() != 4 * 4 * 144 * 144 {
            return Err(AppError::new(
                ErrorCode::ResourceCorrupt,
                "prepared-frame",
                "Prepared frame is missing or corrupt; prepare the project again",
            ));
        }
        let mut values: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        let mask = values.split_off(3 * 144 * 144);
        let target = values;
        let mut masked = target.clone();
        for c in 0..3 {
            for y in 4..134 {
                masked[c * 144 * 144 + y * 144 + 4..c * 144 * 144 + y * 144 + 139].fill(0.);
            }
        }
        Ok(PreparedFrame {
            target,
            masked,
            mask,
        })
    }
}
#[derive(Serialize, Deserialize)]
struct Generation {
    project_id: ProjectId,
    identity: String,
    input_identity: String,
    preprocessing: PreprocessingIdentity,
    #[serde(default)]
    dependencies: Option<Dependencies>,
    frames: Vec<FrameRecord>,
    feature_frames: usize,
    files: BTreeMap<PathBuf, String>,
}
#[derive(Serialize, Deserialize)]
struct Index {
    generation: PathBuf,
    manifest_sha256: String,
    input_identity: String,
}
#[derive(Clone, Serialize)]
struct StageInputs {
    source: String,
    canonical: String,
    face: String,
    feature: String,
}
#[derive(Serialize, Deserialize)]
struct Dependencies {
    source: String,
    canonical: String,
    face: String,
    feature: String,
}
struct Candidate {
    directory: PathBuf,
    generation: Generation,
}
struct Artifacts {
    dependencies: Dependencies,
    frames: Vec<FrameRecord>,
    feature_frames: usize,
    files: BTreeMap<PathBuf, String>,
}
fn key(value: &impl Serialize) -> Result<String> {
    Ok(digest(&serde_json::to_vec(value).map_err(invalid)?))
}
fn stage_inputs(
    project: &ProjectManifest,
    manifests: &[AssetManifest],
    tools: &MediaTools,
    device: &DeviceDescriptor,
) -> Result<StageInputs> {
    let component = |kind: &str| {
        manifests
            .iter()
            .find(|m| m.identity.kind == kind)
            .ok_or_else(|| invalid(format!("Missing verified component {kind}")))
    };
    let source = key(&(
        project.id,
        &project.source_video.sha256,
        project.source_video.size_bytes,
    ))?;
    let engine = env!("CARGO_PKG_VERSION");
    Ok(StageInputs {
        source: source.clone(),
        canonical: key(&(
            "canonical-v1;ffv1-bgr24-no-jpeg;first-presented-video-origin;fps25;first-channel-f32-16k",
            engine,
            &source,
            file_hash(&tools.ffmpeg)?,
            file_hash(&tools.ffprobe)?,
        ))?,
        face: key(&(
            "face-v1;crop152-inner144-roi-source-numpy2-f32",
            engine,
            device,
            component("scrfd_2_5g")?,
            component("pfld_ghost_one")?,
            component("mean_face")?,
        ))?,
        feature: key(&(
            "features-v1;whole-normalize;chunk320000;even-tokens",
            engine,
            device,
            component("feather_hubert")?,
        ))?,
    })
}
/// Validate the entire requested group before copying any of it. Destination is a
/// new owned staging directory; copies are rehashed to detect concurrent changes.
fn reuse_files(
    root: &Path,
    stage: &Path,
    old: &Candidate,
    paths: &[PathBuf],
    cancel: &Cancellation,
) -> Result<bool> {
    for path in paths {
        check(cancel)?;
        if safe_relative(path).is_err() {
            return Ok(false);
        }
        let source = old.directory.join(path);
        let Some(expected) = old.generation.files.get(path) else {
            return Ok(false);
        };
        let valid = fs::canonicalize(&source)
            .ok()
            .is_some_and(|p| p.starts_with(&old.directory) && p.starts_with(root))
            && file_hash(&source).ok().as_ref() == Some(expected);
        if !valid {
            return Ok(false);
        }
    }
    for path in paths {
        check(cancel)?;
        let target = stage.join(path);
        fs::create_dir_all(
            target
                .parent()
                .ok_or_else(|| invalid("Invalid artifact parent"))?,
        )
        .map_err(invalid)?;
        fs::copy(old.directory.join(path), &target).map_err(ffmpeg::error)?;
        if file_hash(&target)? != old.generation.files[path] {
            return Err(invalid(
                "Cache artifact changed during reuse; retry preparation",
            ));
        }
    }
    Ok(true)
}
/// The production dependency orchestrator. Only the stage implementations are
/// callbacks, allowing protocol tests to count rebuilds without faking models.
fn build_artifacts(
    root: &Path,
    stage: &Path,
    previous: Option<&Candidate>,
    inputs: &StageInputs,
    cancel: &Cancellation,
    canonical: impl FnOnce() -> Result<()>,
    faces: impl FnOnce() -> Result<Vec<FrameRecord>>,
    features: impl FnOnce() -> Result<usize>,
) -> Result<Artifacts> {
    check(cancel)?;
    // No v1 migration guesses: generations without explicit dependency keys rebuild.
    let old = previous.filter(|p| {
        p.generation
            .dependencies
            .as_ref()
            .is_some_and(|d| d.source == inputs.source)
    });
    let canonical_key = key(&(&inputs.source, &inputs.canonical))?;
    let mut reused = false;
    if let Some(old) = old {
        if old.generation.dependencies.as_ref().unwrap().canonical == canonical_key {
            reused = reuse_files(
                root,
                stage,
                old,
                &["video.mkv".into(), "audio.wav".into()],
                cancel,
            )?;
        }
    }
    if !reused {
        canonical()?;
    }
    check(cancel)?;
    let video_hash = file_hash(&stage.join("video.mkv"))?;
    let audio_hash = file_hash(&stage.join("audio.wav"))?;
    let face_key = key(&(&canonical_key, &video_hash, &inputs.face))?;
    let feature_key = key(&(&canonical_key, &audio_hash, &inputs.feature))?;
    let mut frame_records = None;
    if let Some(old) = old {
        if old.generation.dependencies.as_ref().unwrap().face == face_key {
            let records = &old.generation.frames;
            // Exact contiguous owned paths: do not copy arbitrary manifest destinations.
            let valid = !records.is_empty()
                && records.iter().enumerate().all(|(i, r)| {
                    r.file == PathBuf::from(format!("frames/{i}.f32"))
                        && old.generation.files.get(&r.file) == Some(&r.sha256)
                });
            if valid
                && reuse_files(
                    root,
                    stage,
                    old,
                    &records.iter().map(|r| r.file.clone()).collect::<Vec<_>>(),
                    cancel,
                )?
            {
                frame_records = Some(records.clone());
            }
        }
    }
    let frames = match frame_records {
        Some(frames) => frames,
        None => faces()?,
    };
    check(cancel)?;
    let mut feature_frames = None;
    if let Some(old) = old {
        if old.generation.dependencies.as_ref().unwrap().feature == feature_key
            && reuse_files(root, stage, old, &["features.f32".into()], cancel)?
        {
            feature_frames = Some(old.generation.feature_frames);
        }
    }
    let feature_frames = match feature_frames {
        Some(count) => count,
        None => features()?,
    };
    check(cancel)?;
    let mut files = BTreeMap::new();
    for name in ["video.mkv", "audio.wav", "features.f32"] {
        files.insert(PathBuf::from(name), file_hash(&stage.join(name))?);
    }
    for frame in &frames {
        files.insert(frame.file.clone(), frame.sha256.clone());
    }
    Ok(Artifacts {
        dependencies: Dependencies {
            source: inputs.source.clone(),
            canonical: canonical_key,
            face: face_key,
            feature: feature_key,
        },
        frames,
        feature_frames,
        files,
    })
}
fn safe_relative(path: &Path) -> Result<()> {
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|p| !matches!(p, Component::Normal(_)))
    {
        return Err(invalid("Unsafe project-relative path"));
    }
    Ok(())
}
fn owned_directory(root: &Path, path: &Path) -> Result<PathBuf> {
    safe_relative(path)?;
    let mut current = root.to_path_buf();
    for c in path.components() {
        current.push(c);
        if current.exists() {
            if !fs::canonicalize(&current)
                .map_err(invalid)?
                .starts_with(root)
            {
                return Err(invalid("Project cache escapes its root"));
            }
        } else {
            fs::create_dir(&current).map_err(invalid)?;
        }
    }
    Ok(current)
}
fn candidate(root: &Path, index: &Index, project: &ProjectManifest) -> Result<Candidate> {
    safe_relative(&index.generation)?;
    let directory = root.join(&index.generation);
    if !fs::canonicalize(&directory)
        .map_err(invalid)?
        .starts_with(root)
    {
        return Err(invalid("Cache generation escapes project root"));
    }
    let bytes = fs::read(directory.join("manifest.json")).map_err(invalid)?;
    if digest(&bytes) != index.manifest_sha256 {
        return Err(invalid("Cache manifest corrupted"));
    }
    let m: Generation = serde_json::from_slice(&bytes).map_err(invalid)?;
    if m.project_id != project.id || m.input_identity != index.input_identity {
        return Err(invalid("Prepared project identity mismatch"));
    }
    Ok(Candidate {
        directory,
        generation: m,
    })
}
fn load(root: &Path, index: &Index, project: &ProjectManifest) -> Result<PreparedProject> {
    let Candidate {
        directory,
        generation: m,
    } = candidate(root, index, project)?;
    for name in ["video.mkv", "audio.wav", "features.f32"] {
        if !m.files.contains_key(Path::new(name)) {
            return Err(invalid("Incomplete cache file index"));
        }
    }
    for record in &m.frames {
        if m.files.get(&record.file) != Some(&record.sha256) {
            return Err(invalid("Frame index hash mismatch"));
        }
    }
    for (path, hash) in &m.files {
        safe_relative(path)?;
        if !fs::canonicalize(directory.join(path))
            .map_err(invalid)?
            .starts_with(root)
        {
            return Err(invalid("Artifact escapes project root"));
        }
        if file_hash(&directory.join(path))? != *hash {
            return Err(invalid("Cache artifact corrupted"));
        }
    }
    let bytes = fs::read(directory.join("features.f32")).map_err(invalid)?;
    if bytes.len() % 4 != 0 {
        return Err(invalid("Corrupt feature bytes"));
    }
    let features = Arc::new(
        bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>(),
    );
    let prepared = PreparedProject {
        project_directory: root.to_path_buf(),
        project_id: project.id,
        identity: m.identity,
        preprocessing: m.preprocessing,
        canonical_video: directory.join("video.mkv"),
        canonical_audio: directory.join("audio.wav"),
        canonical_frame_count: m.frames.len(),
        feature_frame_count: m.feature_frames,
        usable_frames: m.frames.len().min(m.feature_frames),
        frames: m.frames,
        features,
        generation: directory,
    };
    prepared.validate()?;
    Ok(prepared)
}

/// Reopen a checkpoint-pinned immutable generation, independent of the latest cache index.
pub fn reopen_generation(
    root: &Path,
    project: &ProjectManifest,
    relative: &Path,
    manifest_sha256: &str,
) -> Result<PreparedProject> {
    safe_relative(relative)?;
    let root = fs::canonicalize(root).map_err(invalid)?;
    let path = root.join(relative).join("manifest.json");
    if !fs::canonicalize(&path).map_err(invalid)?.starts_with(&root) {
        return Err(invalid("Prepared manifest escapes project"));
    }
    let bytes = fs::read(path).map_err(invalid)?;
    if digest(&bytes) != manifest_sha256 {
        return Err(invalid("Pinned prepared manifest changed"));
    }
    let generation: Generation = serde_json::from_slice(&bytes).map_err(invalid)?;
    let index = Index {
        generation: relative.into(),
        manifest_sha256: manifest_sha256.into(),
        input_identity: generation.input_identity,
    };
    let prepared = load(&root, &index, project)?;
    if prepared.preprocessing.source_sha256 != project.source_video.sha256
        || prepared.preprocessing.cache_directory != relative
        || project
            .preprocessing
            .as_ref()
            .is_some_and(|p| p != &prepared.preprocessing)
    {
        return Err(invalid("Pinned preprocessing does not match project"));
    }
    Ok(prepared)
}
struct Cleanup {
    path: PathBuf,
    directory: bool,
}
impl Drop for Cleanup {
    fn drop(&mut self) {
        if self.directory {
            let _ = fs::remove_dir_all(&self.path);
        } else {
            let _ = fs::remove_file(&self.path);
        }
    }
}
fn progress(
    observer: &dyn TaskObserver,
    stage: &str,
    completed: usize,
    total: Option<usize>,
) -> Result<()> {
    observer.emit(EventPayload::Progress {
        stage: stage.into(),
        completed: completed as u64,
        total: total.map(|n| n as u64),
    })
}
pub fn prepare_project(
    project_directory: &Path,
    project: &ProjectManifest,
    assets: &Path,
    tools: &MediaTools,
    device: &ExecutionDevice,
    observer: &dyn TaskObserver,
    cancel: &Cancellation,
) -> Result<PreparedProject> {
    check(cancel)?;
    let root = fs::canonicalize(project_directory).map_err(invalid)?;
    let persisted: ProjectManifest =
        serde_json::from_slice(&fs::read(root.join("project.json")).map_err(invalid)?)
            .map_err(invalid)?;
    if persisted != *project {
        return Err(invalid(
            "Project manifest does not match accepted project directory",
        ));
    }
    safe_relative(&project.metrics_path)?;
    if let Some(p) = &project.preprocessing {
        safe_relative(&p.cache_directory)?;
    }
    let source = if project.source_video.path.is_absolute() {
        project.source_video.path.clone()
    } else {
        root.join(&project.source_video.path)
    };
    if fs::metadata(&source).map_err(invalid)?.len() != project.source_video.size_bytes
        || file_hash(&source)? != project.source_video.sha256
    {
        return Err(invalid(
            "Source video changed; reopen the project with its new source identity",
        ));
    }
    let cache = owned_directory(&root, Path::new("cache/prepared"))?;
    let lock = cache.join("prepare.lock");
    let _file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&lock)
        .map_err(|_| {
            AppError::new(
                ErrorCode::Busy,
                "prepare",
                "Project preparation is already running (or a stale prepare.lock must be removed)",
            )
        })?;
    drop(_file);
    let _lock = Cleanup {
        path: lock,
        directory: false,
    };
    progress(observer, "verify-components", 0, Some(4))?;
    let mut components = vec![];
    let mut manifests = vec![];
    for (i, name) in [
        "feather_hubert",
        "scrfd_2_5g",
        "pfld_ghost_one",
        "mean_face",
    ]
    .iter()
    .enumerate()
    {
        check(cancel)?;
        let a = VerifiedAssets::load(&assets.join(name))?;
        components.push(a.manifest.identity.clone());
        manifests.push(a.manifest);
        progress(observer, "verify-components", i + 1, Some(4))?;
    }
    let stage_inputs = stage_inputs(project, &manifests, tools, &device.descriptor)?;
    let input_identity = key(&(
        preparation_fingerprint(project, &manifests, tools, &device.descriptor)?,
        &stage_inputs,
    ))?;
    let index_path = cache.join("index.json");
    let previous = fs::read(&index_path)
        .ok()
        .and_then(|bytes| serde_json::from_slice::<Index>(&bytes).ok())
        .and_then(|index| candidate(&root, &index, project).ok());
    if let Ok(bytes) = fs::read(&index_path) {
        if let Ok(index) = serde_json::from_slice::<Index>(&bytes) {
            if index.input_identity == input_identity {
                if let Ok(prepared) = load(&root, &index, project) {
                    check(cancel)?;
                    progress(observer, "prepared-cache", 1, Some(1))?;
                    return Ok(prepared);
                }
            }
        }
    }
    let stage = cache.join(format!("pending-{}", ProjectId::new()));
    fs::create_dir(&stage).map_err(invalid)?;
    let cleanup = Cleanup {
        path: stage.clone(),
        directory: true,
    };
    let Artifacts {
        dependencies,
        frames,
        feature_frames,
        files,
    } = build_artifacts(
        &root,
        &stage,
        previous.as_ref(),
        &stage_inputs,
        cancel,
        || {
            progress(observer, "canonical-media", 0, None)?;
            ffmpeg::canonicalize(&source, &stage, tools, cancel)
        },
        || build_face_frames(&stage, assets, tools, device, observer, cancel),
        || build_audio_features(&stage, assets, device, observer, cancel),
    )?;
    let count = frames.len();
    if count.min(feature_frames) < 2 {
        return Err(invalid(
            "At least two aligned video/audio frames are required",
        ));
    }
    let identity = digest(&serde_json::to_vec(&(&input_identity, &files)).map_err(invalid)?);
    let relative = PathBuf::from(format!("cache/prepared/{identity}-{}", ProjectId::new()));
    let preprocessing = PreprocessingIdentity {
        version: POLICY.into(),
        source_sha256: project.source_video.sha256.clone(),
        config_sha256: input_identity.clone(),
        cache_directory: relative.clone(),
        component_identities: components,
    };
    let generation = Generation {
        project_id: project.id,
        identity,
        input_identity: input_identity.clone(),
        preprocessing,
        frames,
        feature_frames,
        dependencies: Some(dependencies),
        files,
    };
    let manifest = serde_json::to_vec(&generation).map_err(invalid)?;
    fs::write(stage.join("manifest.json"), &manifest).map_err(invalid)?;
    for path in generation
        .files
        .keys()
        .cloned()
        .chain(std::iter::once(PathBuf::from("manifest.json")))
    {
        fs::File::open(stage.join(path))
            .map_err(invalid)?
            .sync_all()
            .map_err(invalid)?;
    }
    check(cancel)?;
    if file_hash(&source)? != project.source_video.sha256 {
        return Err(invalid("Source changed during preparation"));
    }
    fs::rename(&stage, root.join(&relative)).map_err(invalid)?;
    drop(cleanup);
    let index = Index {
        generation: relative,
        manifest_sha256: digest(&manifest),
        input_identity,
    };
    let result = load(&root, &index, project)?;
    check(cancel)?;
    let current: ProjectManifest =
        serde_json::from_slice(&fs::read(root.join("project.json")).map_err(invalid)?)
            .map_err(invalid)?;
    if current != *project {
        return Err(invalid(
            "Project manifest changed during preparation; retry with the current project",
        ));
    }
    let pending = cache.join(format!("index-{}.tmp", ProjectId::new()));
    let _pending = Cleanup {
        path: pending.clone(),
        directory: false,
    };
    let mut file = fs::File::create(&pending).map_err(invalid)?;
    file.write_all(&serde_json::to_vec(&index).map_err(invalid)?)
        .map_err(invalid)?;
    file.sync_all().map_err(invalid)?;
    drop(file);
    fs::rename(pending, index_path).map_err(invalid)?;
    progress(observer, "prepared", count, Some(count))?;
    Ok(result)
}

fn build_face_frames(
    stage: &Path,
    assets: &Path,
    tools: &MediaTools,
    device: &ExecutionDevice,
    observer: &dyn TaskObserver,
    cancel: &Cancellation,
) -> Result<Vec<FrameRecord>> {
    let info = probe(&stage.join("video.mkv"), &stage, tools, cancel)?;
    let mut raw = ffmpeg::decode_frames(&stage, tools, cancel)?;
    let bytes_per_frame = info.width * info.height * 3;
    let count = raw.metadata().map_err(invalid)?.len() as usize / bytes_per_frame;
    if count == 0 || raw.metadata().map_err(invalid)?.len() as usize % bytes_per_frame != 0 {
        return Err(invalid("Canonical decoder produced invalid frame bytes"));
    }
    fs::create_dir(stage.join("frames")).map_err(invalid)?;
    let detector = Scrfd::load(&assets.join("scrfd_2_5g"), &device.device)?;
    let landmark_model = PfldGhostOne::load(&assets.join("pfld_ghost_one"), &device.device)?;
    let mean = load_mean_face(&assets.join("mean_face"))?;
    let mut frames = vec![];
    for index in 0..count {
        check(cancel)?;
        let mut data = vec![0; bytes_per_frame];
        raw.read_exact(&mut data).map_err(invalid)?;
        let frame = BgrFrame {
            width: info.width,
            height: info.height,
            stride: info.width * 3,
            data,
        };
        let attach = |mut e: AppError| {
            e.evidence.push(format!(
                "canonical_frame={index}; timestamp_seconds={:.6}; stage=frame-landmarks",
                index as f64 / 25.
            ));
            e
        };
        let landmarks = locate_face(&detector, &landmark_model, &mean, &frame).map_err(attach)?;
        let prepared = prepare_frame(&frame, &landmarks).map_err(attach)?;
        let file = PathBuf::from(format!("frames/{index}.f32"));
        let bytes: Vec<_> = prepared
            .target
            .iter()
            .chain(&prepared.mask)
            .flat_map(|v| v.to_le_bytes())
            .collect();
        fs::write(stage.join(&file), &bytes).map_err(ffmpeg::error)?;
        let hash = digest(&bytes);
        frames.push(FrameRecord {
            file,
            sha256: hash,
            landmarks: landmarks.points.to_vec(),
        });
        progress(observer, "frame-landmarks", index + 1, Some(count))?;
    }
    drop(raw);
    fs::remove_file(stage.join("decoded.bgr")).map_err(invalid)?;
    drop(detector);
    drop(landmark_model);
    Ok(frames)
}
fn build_audio_features(
    stage: &Path,
    assets: &Path,
    device: &ExecutionDevice,
    observer: &dyn TaskObserver,
    cancel: &Cancellation,
) -> Result<usize> {
    check(cancel)?;
    progress(observer, "audio-features", 0, None)?;
    let samples = ffmpeg::read_waveform(&stage.join("audio.wav"))?;
    let model = FeatherHubert::load(&assets.join("feather_hubert"), &device.device)?;
    let features = extract_audio_features_cancellable(&model, &samples, cancel)?;
    check(cancel)?;
    let mut file = fs::File::create(stage.join("features.f32")).map_err(ffmpeg::error)?;
    for v in features.values {
        file.write_all(&v.to_le_bytes()).map_err(ffmpeg::error)?;
    }
    file.sync_all().map_err(invalid)?;
    drop(file);
    Ok(features.video_frames)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn root() -> PathBuf {
        let p = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../../.artifacts/task8-cache/{}", ProjectId::new()));
        fs::create_dir_all(&p).unwrap();
        fs::canonicalize(p).unwrap()
    }
    fn project() -> ProjectManifest {
        ProjectManifest {
            schema_version: 1,
            id: ProjectId::new(),
            name: "cache protocol fixture".into(),
            source_video: SourceVideo {
                path: "source.mkv".into(),
                sha256: "source-sha".into(),
                size_bytes: 0,
            },
            preprocessing: None,
            model_identity: None,
            checkpoints: vec![],
            metrics_path: "metrics.jsonl".into(),
        }
    }
    #[test]
    fn cache_invalidates_when_model_changes() {
        let root = root();
        let tools = MediaTools {
            ffmpeg: root.join("ffmpeg"),
            ffprobe: root.join("ffprobe"),
        };
        fs::write(&tools.ffmpeg, b"ffmpeg executable identity").unwrap();
        fs::write(&tools.ffprobe, b"ffprobe executable identity").unwrap();
        let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.artifacts/assets");
        let mut manifests: Vec<AssetManifest> = [
            "feather_hubert",
            "scrfd_2_5g",
            "pfld_ghost_one",
            "mean_face",
        ]
        .iter()
        .map(|name| {
            serde_json::from_slice(&fs::read(assets.join(name).join("manifest.json")).unwrap())
                .unwrap()
        })
        .collect();
        let device = DeviceDescriptor {
            id: "flex:0".into(),
            name: "CPU".into(),
            vendor: None,
            driver: None,
            backend: BackendKind::Flex,
            memory_bytes: None,
        };
        let p = project();
        let old = preparation_fingerprint(&p, &manifests, &tools, &device).unwrap();
        assert_eq!(
            old,
            preparation_fingerprint(&p, &manifests, &tools, &device).unwrap()
        );
        let before = stage_inputs(&p, &manifests, &tools, &device).unwrap();
        manifests[0].identity.weights_sha256 = "changed audio model hash".into();
        assert_ne!(
            old,
            preparation_fingerprint(&p, &manifests, &tools, &device).unwrap()
        );
        let after = stage_inputs(&p, &manifests, &tools, &device).unwrap();
        assert_eq!(before.source, after.source);
        assert_eq!(before.canonical, after.canonical);
        assert_eq!(before.face, after.face);
        assert_ne!(before.feature, after.feature);
        manifests[1]
            .hashes
            .insert("config.json".into(), "new detector config".into());
        let face_changed = stage_inputs(&p, &manifests, &tools, &device).unwrap();
        assert_eq!(after.canonical, face_changed.canonical);
        assert_eq!(after.feature, face_changed.feature);
        assert_ne!(after.face, face_changed.face);
        fs::write(&tools.ffmpeg, b"changed executable").unwrap();
        let changed = preparation_fingerprint(&p, &manifests, &tools, &device).unwrap();
        manifests[0].identity.weights_sha256 = "another model hash".into();
        assert_ne!(
            changed,
            preparation_fingerprint(&p, &manifests, &tools, &device).unwrap()
        );
    }
    #[test]
    fn cache_rejects_corruption_and_project_mismatch() {
        let root = root();
        let p = project();
        let relative = PathBuf::from("cache/prepared/generation");
        let dir = owned_directory(&root, &relative).unwrap();
        fs::create_dir(dir.join("frames")).unwrap();
        let mut files = BTreeMap::new();
        for name in ["video.mkv", "audio.wav"] {
            fs::write(dir.join(name), b"protocol fixture").unwrap();
            files.insert(PathBuf::from(name), file_hash(&dir.join(name)).unwrap());
        }
        fs::write(dir.join("features.f32"), vec![0; 2 * 2048 * 4]).unwrap();
        files.insert(
            "features.f32".into(),
            file_hash(&dir.join("features.f32")).unwrap(),
        );
        let mut frames = vec![];
        for i in 0..3 {
            let file = PathBuf::from(format!("frames/{i}.f32"));
            fs::write(dir.join(&file), vec![0; 4 * 144 * 144 * 4]).unwrap();
            let sha256 = file_hash(&dir.join(&file)).unwrap();
            files.insert(file.clone(), sha256.clone());
            frames.push(FrameRecord {
                file,
                sha256,
                landmarks: vec![],
            });
        }
        let generation = Generation {
            project_id: p.id,
            dependencies: None,
            identity: "output identity".into(),
            input_identity: "inputs".into(),
            preprocessing: PreprocessingIdentity {
                version: POLICY.into(),
                source_sha256: p.source_video.sha256.clone(),
                config_sha256: "inputs".into(),
                cache_directory: relative.clone(),
                component_identities: vec![],
            },
            frames,
            feature_frames: 2,
            files,
        };
        let bytes = serde_json::to_vec(&generation).unwrap();
        fs::write(dir.join("manifest.json"), &bytes).unwrap();
        let index = Index {
            generation: relative,
            manifest_sha256: digest(&bytes),
            input_identity: "inputs".into(),
        };
        let cached = load(&root, &index, &p).unwrap();
        assert_eq!(cached.canonical_frame_count, 3);
        assert_eq!(cached.feature_frame_count, 2);
        assert_eq!(cached.usable_frames, 2);
        cached.read_frame(0).unwrap();
        let mut other = p.clone();
        other.id = ProjectId::new();
        assert!(load(&root, &index, &other).is_err());
        fs::write(dir.join("features.f32"), b"broken").unwrap();
        assert!(load(&root, &index, &p).is_err());
        fs::write(dir.join("frames/0.f32"), b"broken").unwrap();
        assert!(cached.read_frame(0).is_err());
    }
    #[test]
    fn dependency_cache_orchestration_rebuilds_only_affected_stages() {
        // Protocol fixture: these callbacks count disk stages, not neural acceptance.
        fn run(
            root: &Path,
            previous: Option<&Candidate>,
            input: &StageInputs,
        ) -> (PathBuf, Artifacts, [u32; 3]) {
            let stage = root.join(format!("stage-{}", ProjectId::new()));
            fs::create_dir(&stage).unwrap();
            let counts = std::cell::RefCell::new([0; 3]);
            let result = build_artifacts(
                root,
                &stage,
                previous,
                input,
                &Cancellation::new(),
                || {
                    counts.borrow_mut()[0] += 1;
                    fs::write(stage.join("video.mkv"), b"video").unwrap();
                    fs::write(stage.join("audio.wav"), b"audio").unwrap();
                    Ok(())
                },
                || {
                    counts.borrow_mut()[1] += 1;
                    fs::create_dir(stage.join("frames")).unwrap();
                    let file = PathBuf::from("frames/0.f32");
                    fs::write(stage.join(&file), b"frame").unwrap();
                    Ok(vec![FrameRecord {
                        file,
                        sha256: digest(b"frame"),
                        landmarks: vec![[1, 2]; 110],
                    }])
                },
                || {
                    counts.borrow_mut()[2] += 1;
                    fs::write(stage.join("features.f32"), b"features").unwrap();
                    Ok(1)
                },
            )
            .unwrap();
            (stage, result, counts.into_inner())
        }
        let root = root();
        let p = project();
        let inputs = StageInputs {
            source: "source".into(),
            canonical: "canonical".into(),
            face: "face-models-config".into(),
            feature: "audio-model-config".into(),
        };
        let (directory, a, counts) = run(&root, None, &inputs);
        assert_eq!(counts, [1, 1, 1]);
        let previous = Candidate {
            directory,
            generation: Generation {
                project_id: p.id,
                identity: "output".into(),
                input_identity: "input".into(),
                dependencies: Some(a.dependencies),
                preprocessing: PreprocessingIdentity {
                    version: POLICY.into(),
                    source_sha256: "source".into(),
                    config_sha256: "input".into(),
                    cache_directory: "cache".into(),
                    component_identities: vec![],
                },
                frames: a.frames,
                feature_frames: a.feature_frames,
                files: a.files,
            },
        };
        assert_eq!(run(&root, Some(&previous), &inputs).2, [0, 0, 0]);
        let mut changed = inputs.clone();
        changed.feature = "new audio weights".into();
        assert_eq!(run(&root, Some(&previous), &changed).2, [0, 0, 1]);
        fs::write(previous.directory.join("features.f32"), b"corrupt").unwrap();
        assert_eq!(run(&root, Some(&previous), &inputs).2, [0, 0, 1]);
        fs::write(previous.directory.join("features.f32"), b"features").unwrap();
        changed = inputs.clone();
        changed.face = "new face model".into();
        assert_eq!(run(&root, Some(&previous), &changed).2, [0, 1, 0]);
        changed = inputs.clone();
        changed.canonical = "new canonical config".into();
        assert_eq!(run(&root, Some(&previous), &changed).2, [1, 1, 1]);
        changed = inputs.clone();
        changed.source = "new source".into();
        assert_eq!(run(&root, Some(&previous), &changed).2, [1, 1, 1]);
        fs::write(previous.directory.join("frames/0.f32"), b"corrupt").unwrap();
        assert_eq!(run(&root, Some(&previous), &inputs).2, [0, 1, 0]);
        fs::write(previous.directory.join("frames/0.f32"), b"frame").unwrap();
        fs::write(previous.directory.join("audio.wav"), b"corrupt").unwrap();
        assert_eq!(run(&root, Some(&previous), &inputs).2, [1, 0, 0]);
        fs::write(previous.directory.join("audio.wav"), b"audio").unwrap();
        let failed_stage = root.join("failed-stage");
        fs::create_dir(&failed_stage).unwrap();
        let old_hashes = previous.generation.files.clone();
        changed = inputs.clone();
        changed.feature = "failed new feature model".into();
        let error = build_artifacts(
            &root,
            &failed_stage,
            Some(&previous),
            &changed,
            &Cancellation::new(),
            || panic!("Canonical stage must be reused"),
            || panic!("Face stage must be reused"),
            || Err(invalid("synthetic feature failure")),
        );
        assert!(error.is_err());
        for (file, hash) in old_hashes {
            assert_eq!(file_hash(&previous.directory.join(file)).unwrap(), hash);
        }
        // Reuse never allows destinations from an untrusted frame index.
        let mut unsafe_previous = previous;
        unsafe_previous.generation.frames[0].file = "../escape.f32".into();
        assert_eq!(run(&root, Some(&unsafe_previous), &inputs).2, [0, 1, 0]);
        assert!(!root.join("escape.f32").exists());
    }
}
