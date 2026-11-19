use super::*;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn model(root: &Path, version: &str) -> Component {
    let path = root.join("public-assets").join(version);
    let files: Vec<PathBuf> = [
        "mean_face",
        "feather_hubert",
        "pfld_ghost_one",
        "scrfd_2_5g",
        "vgg19",
    ]
    .map(|kind| PathBuf::from(format!("{kind}/manifest.json")))
    .into();
    for name in &files {
        let dest = path.join(name);
        fs::create_dir_all(dest.parent().unwrap()).unwrap();
        fs::write(dest, version).unwrap();
    }
    let manifest = AssetManifest {
        identity: ModelIdentity {
            kind: "feathertalk-public-assets".into(),
            architecture_version: "test".into(),
            source_sha256: sha(version.as_bytes()),
            weights_sha256: sha(version.as_bytes()),
        },
        sizes: files
            .iter()
            .map(|p| (p.clone(), version.len() as u64))
            .collect(),
        hashes: files
            .iter()
            .map(|p| (p.clone(), sha(version.as_bytes())))
            .collect(),
        relative_files: files,
        urls: BTreeMap::new(),
        min_engine_version: "0.1.0".into(),
        provenance: "bundled resource fixture".into(),
    };
    let raw = serde_json::to_vec(&manifest).unwrap();
    fs::write(path.join("manifest.json"), &raw).unwrap();
    Component {
        pin: ResourcePin {
            id: "public-assets".into(),
            version: version.into(),
        },
        kind: "model_bundle".into(),
        target: None,
        manifest,
        manifest_size: raw.len() as u64,
        manifest_sha256: sha(&raw),
        manifest_urls: vec![],
        supplemental: BTreeMap::new(),
        executables: vec![],
        notices: vec![],
        redistribution_approved: false,
        license: "NOASSERTION".into(),
    }
}

#[test]
fn bundled_model_seeds_only_native_catalog_pins_and_rejects_corruption() {
    let temp = tempfile::tempdir().unwrap();
    let root = temp.path().join("bootstrap");
    let cache = temp.path().join("cache");
    let native = model(&root, "native");
    let foreign = model(&root, "foreign");
    let target = TargetPlatform {
        os: std::env::consts::OS.into(),
        architecture: std::env::consts::ARCH.into(),
        backend: "cpu".into(),
    };
    let foreign_target = TargetPlatform {
        os: if cfg!(windows) { "linux" } else { "windows" }.into(),
        architecture: "x86_64".into(),
        backend: "cpu".into(),
    };
    let catalog = Catalog {
        schema_version: 1,
        components: vec![native.clone(), foreign.clone()],
        defaults: vec![
            ResourceSelection {
                target: target.clone(),
                resources: vec![native.pin.clone()],
            },
            ResourceSelection {
                target: foreign_target,
                resources: vec![foreign.pin],
            },
        ],
    };
    let payload = root.join("public-assets/native/vgg19/manifest.json");
    fs::write(&payload, b"broken").unwrap();
    let manager = ResourceManager::new(cache.clone(), catalog).unwrap();
    manager.install_bundled(&root);
    assert!(!manager.status(&native.pin, &target).ready);
    fs::write(&payload, b"native").unwrap();
    manager.install_bundled(&root);
    assert!(manager.status(&native.pin, &target).ready);
    assert!(!cache.join("public-assets/foreign").exists());
}
