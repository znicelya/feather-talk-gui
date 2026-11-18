use feathertalk_core::*;
use feathertalk_engine::data::prepared::reopen_generation;
use std::{
    fs,
    path::{Path, PathBuf},
};
fn digest(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    format!("{:x}", Sha256::digest(bytes))
}
fn file_hash(path: &Path) -> std::io::Result<String> {
    Ok(digest(&fs::read(path)?))
}
pub fn fixture() -> (PathBuf, feathertalk_engine::data::prepared::PreparedProject) {
    fixture_frames(2)
}
pub fn fixture_frames(
    count: usize,
) -> (PathBuf, feathertalk_engine::data::prepared::PreparedProject) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
        "../../.artifacts/task9c-projects/{}",
        ProjectId::new()
    ));
    fs::create_dir_all(root.join("cache/generation/frames")).unwrap();
    fs::create_dir(root.join("checkpoints")).unwrap();
    let root = fs::canonicalize(root).unwrap();
    fs::write(
        root.join("source.mkv"),
        b"owned state protocol source fixture",
    )
    .unwrap();
    let source = SourceVideo {
        path: "source.mkv".into(),
        sha256: file_hash(&root.join("source.mkv")).unwrap(),
        size_bytes: fs::metadata(root.join("source.mkv")).unwrap().len(),
    };
    let p = ProjectManifest {
        schema_version: 1,
        id: ProjectId::new(),
        name: "state protocol".into(),
        source_video: source.clone(),
        preprocessing: None,
        model_identity: None,
        checkpoints: vec![],
        metrics_path: "metrics.jsonl".into(),
    };
    fs::write(root.join("project.json"), serde_json::to_vec(&p).unwrap()).unwrap();
    let prep = PreprocessingIdentity {
        version: "fixture-v1".into(),
        source_sha256: source.sha256,
        config_sha256: "fixture-inputs".into(),
        cache_directory: "cache/generation".into(),
        component_identities: vec![],
    };
    let generation = root.join("cache/generation");
    let mut files = std::collections::BTreeMap::new();
    for (name, bytes) in [
        ("video.mkv", vec![1]),
        ("audio.wav", vec![2]),
        ("features.f32", vec![0; 2 * 2048 * 4]),
        (
            "frames/0.f32",
            (0..4 * 144 * 144)
                .flat_map(|i| (if i >= 3 * 144 * 144 { 1.0f32 } else { 0.2f32 }).to_le_bytes())
                .collect(),
        ),
        (
            "frames/1.f32",
            (0..4 * 144 * 144)
                .flat_map(|i| (if i >= 3 * 144 * 144 { 1.0f32 } else { 0.2f32 }).to_le_bytes())
                .collect(),
        ),
    ] {
        fs::write(generation.join(name), &bytes).unwrap();
        files.insert(name, digest(&bytes));
    }
    let features = vec![0; count * 2048 * 4];
    fs::write(generation.join("features.f32"), &features).unwrap();
    files.insert("features.f32", digest(&features));
    // Extra frames may reference the same immutable fixture file.
    let frames:Vec<_>=(0..count).map(|_|serde_json::json!({"file":"frames/0.f32","sha256":files["frames/0.f32"],"landmarks":[]})).collect();
    let m = serde_json::json!({"project_id":p.id,"identity":"owned-fixed-data-v1","input_identity":"fixture-inputs","preprocessing":prep,"frames":frames,"feature_frames":count,"files":files});
    fs::write(
        generation.join("manifest.json"),
        serde_json::to_vec(&m).unwrap(),
    )
    .unwrap();
    let prepared = reopen_generation(
        &root,
        &p,
        Path::new("cache/generation"),
        &file_hash(&generation.join("manifest.json")).unwrap(),
    )
    .unwrap();
    (root, prepared)
}
pub fn device() -> feathertalk_engine::ExecutionDevice {
    let backend = match std::env::var("FT_TEST_BACKEND").as_deref() {
        Ok("cuda") => BackendKind::Cuda,
        Ok("vulkan") => BackendKind::Vulkan,
        _ => BackendKind::Flex,
    };
    feathertalk_engine::open_device(&DeviceRequest {
        backend,
        device_id: format!(
            "{}:0",
            match backend {
                BackendKind::Cuda => "cuda",
                BackendKind::Vulkan => "vulkan",
                _ => "flex",
            }
        ),
    })
    .unwrap()
}
