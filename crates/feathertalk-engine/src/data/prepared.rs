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
const POLICY: &str = "prepare-v1;ffv1-bgr24-no-jpeg;first-presented-video-origin;fps25;first-channel-f32-16k;crop152-inner144-roi-source;hubert-whole-normalize";
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
fn load(root: &Path, index: &Index, project: &ProjectManifest) -> Result<PreparedProject> {
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
    let input_identity = preparation_fingerprint(project, &manifests, tools, &device.descriptor)?;
    let index_path = cache.join("index.json");
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
    progress(observer, "canonical-media", 0, None)?;
    ffmpeg::canonicalize(&source, &stage, tools, cancel)?;
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
    let mut files = BTreeMap::new();
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
        files.insert(file.clone(), hash.clone());
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
    check(cancel)?;
    progress(observer, "audio-features", 0, None)?;
    let samples = ffmpeg::read_waveform(&stage.join("audio.wav"))?;
    let model = FeatherHubert::load(&assets.join("feather_hubert"), &device.device)?;
    let features = extract_audio_features_cancellable(&model, &samples, cancel)?;
    check(cancel)?;
    if count.min(features.video_frames) < 2 {
        return Err(invalid(
            "At least two aligned video/audio frames are required",
        ));
    }
    let mut file = fs::File::create(stage.join("features.f32")).map_err(ffmpeg::error)?;
    for v in features.values {
        file.write_all(&v.to_le_bytes()).map_err(ffmpeg::error)?;
    }
    file.sync_all().map_err(invalid)?;
    drop(file);
    for name in ["video.mkv", "audio.wav", "features.f32"] {
        files.insert(PathBuf::from(name), file_hash(&stage.join(name))?);
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
        feature_frames: features.video_frames,
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
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../.artifacts/assets/feather_hubert/manifest.json");
        let mut manifests: Vec<AssetManifest> =
            vec![serde_json::from_slice(&fs::read(path).unwrap()).unwrap()];
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
        manifests[0].identity.weights_sha256 = "changed audio model hash".into();
        assert_ne!(
            old,
            preparation_fingerprint(&p, &manifests, &tools, &device).unwrap()
        );
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
}
