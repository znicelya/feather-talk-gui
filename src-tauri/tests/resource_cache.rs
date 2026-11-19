use feather_talk_gui_lib::resources::*;
use feathertalk_core::*;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    fs,
    io::{Read, Write},
    net::TcpListener,
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
};
fn hash(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
fn target() -> TargetPlatform {
    TargetPlatform {
        os: "windows".into(),
        architecture: "x86_64".into(),
        backend: "cpu".into(),
    }
}
struct Observer;
impl TaskObserver for Observer {
    fn emit(&self, _: EventPayload) -> Result<()> {
        Ok(())
    }
}
fn fixture(version: &str, bytes: &[u8]) -> (tempfile::TempDir, Component) {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("data"), bytes).unwrap();
    let manifest = AssetManifest {
        identity: ModelIdentity {
            kind: "test".into(),
            architecture_version: version.into(),
            source_sha256: hash(bytes),
            weights_sha256: hash(bytes),
        },
        relative_files: vec!["data".into()],
        sizes: BTreeMap::from([("data".into(), bytes.len() as u64)]),
        hashes: BTreeMap::from([("data".into(), hash(bytes))]),
        urls: BTreeMap::new(),
        min_engine_version: "0.1.0".into(),
        provenance: "test fixture".into(),
    };
    let raw = serde_json::to_vec(&manifest).unwrap();
    fs::write(dir.path().join("manifest.json"), &raw).unwrap();
    (
        dir,
        Component {
            pin: ResourcePin {
                id: "test".into(),
                version: version.into(),
            },
            target: None,
            kind: "model".into(),
            manifest,
            manifest_sha256: hash(&raw),
            manifest_size: raw.len() as u64,
            manifest_urls: vec![],
            supplemental: BTreeMap::new(),
            executables: vec![],
            notices: vec![],
            redistribution_approved: false,
            license: "NOASSERTION".into(),
        },
    )
}
fn selection(c: &Component) -> ResourceSelection {
    ResourceSelection {
        target: target(),
        resources: vec![c.pin.clone()],
    }
}
fn manager(root: PathBuf, cs: Vec<Component>) -> ResourceManager {
    let defaults = vec![selection(&cs[0])];
    ResourceManager::new(
        root,
        Catalog {
            schema_version: 1,
            components: cs,
            defaults,
        },
    )
    .unwrap()
}
fn server(
    body: Vec<u8>,
    declared: usize,
) -> (String, Arc<AtomicUsize>, std::thread::JoinHandle<()>) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    l.set_nonblocking(true).unwrap();
    let url = format!("http://{}/data", l.local_addr().unwrap());
    let count = Arc::new(AtomicUsize::new(0));
    let c = count.clone();
    let h = std::thread::spawn(move || {
        let end = std::time::Instant::now() + std::time::Duration::from_secs(3);
        while std::time::Instant::now() < end {
            if let Ok((mut s, _)) = l.accept() {
                c.fetch_add(1, Ordering::SeqCst);
                let _ = s.read(&mut [0; 4096]);
                let _ = write!(
                    s,
                    "HTTP/1.1 200 OK\r\nContent-Length: {declared}\r\nConnection: close\r\n\r\n"
                );
                let _ = s.write_all(&body);
            } else {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
    });
    (url, count, h)
}
#[test]
fn partial_and_bad_hash_never_become_ready() {
    for body in [b"short".to_vec(), b"bad payload!".to_vec()] {
        let (_src, mut c) = fixture("1", b"good payload");
        let (url, count, h) = server(body, 12);
        c.manifest.urls.insert("data".into(), vec![url]);
        let root = tempfile::tempdir().unwrap();
        let m = manager(root.path().into(), vec![c.clone()]);
        let corrupt_inventory = m.inventory(&target()).unwrap();
        assert!(!corrupt_inventory.ready);
        assert!(
            m.prepare(&selection(&c), &Observer, &Cancellation::new())
                .is_err()
        );
        assert!(!m.inventory(&target()).unwrap().ready);
        h.join().unwrap();
        assert!(count.load(Ordering::SeqCst) > 0);
    }
}
#[test]
fn offline_uses_complete_versions() {
    let (src, c) = fixture("1", b"complete");
    let root = tempfile::tempdir().unwrap();
    let m = manager(root.path().into(), vec![c.clone()]);
    m.import_local(&src.path().join("manifest.json")).unwrap();
    let complete_inventory = m.inventory(&target()).unwrap();
    assert!(complete_inventory.ready);
    let p = complete_inventory.items[0].path.as_ref().unwrap();
    assert_eq!(fs::read(p.join("data")).unwrap(), b"complete");
    assert!(
        m.prepare(&selection(&c), &Observer, &Cancellation::new())
            .unwrap()
            .ready
    );
    fs::write(p.join("data"), b"tampered").unwrap();
    assert!(!m.inventory(&target()).unwrap().ready);
}
#[test]
fn upgrade_preserves_project_dependency() {
    let (old, a) = fixture("1", b"old");
    let (new, b) = fixture("2", b"new");
    let root = tempfile::tempdir().unwrap();
    let m = manager(root.path().into(), vec![a.clone(), b.clone()]);
    m.import_local(&old.path().join("manifest.json")).unwrap();
    m.import_local(&new.path().join("manifest.json")).unwrap();
    for c in [a, b] {
        assert!(m.inventory_selection(&selection(&c)).unwrap().ready);
    }
}
#[test]
fn executable_matches_target() {
    let (src, mut c) = fixture("1", b"not executable");
    c.executables = vec!["data".into()];
    c.target = Some(target());
    let root = tempfile::tempdir().unwrap();
    let m = manager(root.path().into(), vec![c]);
    assert!(m.import_local(&src.path().join("manifest.json")).is_err());
}

#[test]
fn retained_identity_survives_catalog_update_and_unrelated_corruption() {
    let (old, a) = fixture("1", b"old");
    let (new, b) = fixture("2", b"new");
    let root = tempfile::tempdir().unwrap();
    let m = manager(root.path().into(), vec![a.clone(), b.clone()]);
    m.import_local(&old.path().join("manifest.json")).unwrap();
    m.import_local(&new.path().join("manifest.json")).unwrap();
    let m = manager(root.path().into(), vec![b.clone()]);
    let old_root = m.resolve_identity(&a.manifest.identity, &target()).unwrap();
    assert_eq!(fs::read(old_root.join("data")).unwrap(), b"old");
    fs::write(root.path().join("test/2/data"), b"bad").unwrap();
    assert_eq!(
        m.resolve_identity(&a.manifest.identity, &target()).unwrap(),
        old_root
    );
    assert!(m.inventory_selection(&selection(&a)).unwrap().ready);
}
#[test]
fn successful_download_retry_offline_and_cancel() {
    let (src, mut c) = fixture("1", b"complete");
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    l.set_nonblocking(true).unwrap();
    let base = format!("http://{}", l.local_addr().unwrap());
    c.manifest
        .urls
        .insert("data".into(), vec![format!("{base}/data")]);
    let raw = serde_json::to_vec(&c.manifest).unwrap();
    c.manifest_sha256 = hash(&raw);
    c.manifest_size = raw.len() as u64;
    c.manifest_urls = vec![format!("{base}/manifest.json")];
    fs::write(src.path().join("manifest.json"), &raw).unwrap();
    let count = Arc::new(AtomicUsize::new(0));
    let cnt = count.clone();
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let stopped = stop.clone();
    let h = std::thread::spawn(move || {
        while !stopped.load(Ordering::SeqCst) {
            if let Ok((mut s, _)) = l.accept() {
                let n = cnt.fetch_add(1, Ordering::SeqCst);
                let mut request = [0; 4096];
                let len = s.read(&mut request).unwrap();
                if n == 0 {
                    let _ = write!(
                        s,
                        "HTTP/1.1 503 Unavailable\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
                    );
                    continue;
                }
                let body = if String::from_utf8_lossy(&request[..len]).contains("/manifest.json") {
                    raw.as_slice()
                } else {
                    b"complete"
                };
                let _ = write!(
                    s,
                    "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                    body.len()
                );
                let _ = s.write_all(body);
            } else {
                std::thread::sleep(std::time::Duration::from_millis(5));
            }
        }
    });
    let root = tempfile::tempdir().unwrap();
    let m = manager(root.path().into(), vec![c.clone()]);
    let cancel = Cancellation::new();
    cancel.cancel();
    assert_eq!(
        m.prepare(&selection(&c), &Observer, &cancel)
            .unwrap_err()
            .code,
        ErrorCode::Cancelled
    );
    assert_eq!(count.load(Ordering::SeqCst), 0);
    assert!(
        m.prepare(&selection(&c), &Observer, &Cancellation::new())
            .unwrap()
            .ready
    );
    assert_eq!(count.load(Ordering::SeqCst), 3);
    count.store(0, Ordering::SeqCst);
    assert!(m.inventory(&target()).unwrap().ready);
    assert!(
        m.prepare(&selection(&c), &Observer, &Cancellation::new())
            .unwrap()
            .ready
    );
    assert_eq!(count.load(Ordering::SeqCst), 0);
    stop.store(true, Ordering::SeqCst);
    h.join().unwrap();
    assert!(
        m.prepare(&selection(&c), &Observer, &Cancellation::new())
            .unwrap()
            .ready
    );
}
#[test]
fn portable_paths_and_inventory_conflicts_rejected() {
    let (_, base) = fixture("1", b"x");
    for name in ["../escape", "/absolute", "a/../b", "A.", "CON", "a ", "a:b"] {
        let mut c = base.clone();
        c.manifest.relative_files = vec![name.into()];
        c.manifest.sizes = BTreeMap::from([(name.into(), 1)]);
        c.manifest.hashes = BTreeMap::from([(name.into(), hash(b"x"))]);
        let root = tempfile::tempdir().unwrap();
        assert!(
            ResourceManager::new(
                root.path().into(),
                Catalog {
                    schema_version: 1,
                    components: vec![c],
                    defaults: vec![]
                }
            )
            .is_err(),
            "{name}"
        );
    }
    let mut c = base;
    c.manifest.relative_files.push("data".into());
    let root = tempfile::tempdir().unwrap();
    assert!(
        ResourceManager::new(
            root.path().into(),
            Catalog {
                schema_version: 1,
                components: vec![c],
                defaults: vec![]
            }
        )
        .is_err()
    );
}
#[test]
fn corrupt_import_does_not_publish_or_destroy_previous_version() {
    let (old, a) = fixture("1", b"old");
    let (bad, b) = fixture("2", b"new");
    let root = tempfile::tempdir().unwrap();
    let m = manager(root.path().into(), vec![a.clone(), b.clone()]);
    m.import_local(&old.path().join("manifest.json")).unwrap();
    fs::write(bad.path().join("data"), b"BAD").unwrap();
    assert!(m.import_local(&bad.path().join("manifest.json")).is_err());
    assert!(!m.inventory_selection(&selection(&b)).unwrap().ready);
    assert!(m.inventory_selection(&selection(&a)).unwrap().ready);
    assert!(!root.path().join("test/2").exists());
}
#[test]
fn binary_headers_match_os_and_architecture() {
    for (os, arch, mut bytes) in [
        ("windows", "x86_64", vec![0u8; 128]),
        ("linux", "aarch64", vec![0u8; 64]),
        ("macos", "aarch64", vec![0u8; 64]),
    ] {
        match os {
            "windows" => {
                bytes[..2].copy_from_slice(b"MZ");
                bytes[60..64].copy_from_slice(&64u32.to_le_bytes());
                bytes[64..68].copy_from_slice(b"PE\0\0");
                bytes[68..70].copy_from_slice(&0x8664u16.to_le_bytes());
                bytes[88..90].copy_from_slice(&0x20bu16.to_le_bytes());
            }
            "linux" => {
                bytes[..4].copy_from_slice(b"\x7fELF");
                bytes[4] = 2;
                bytes[5] = 1;
                bytes[16..18].copy_from_slice(&2u16.to_le_bytes());
                bytes[18..20].copy_from_slice(&183u16.to_le_bytes());
            }
            _ => {
                bytes[..4].copy_from_slice(&[0xcf, 0xfa, 0xed, 0xfe]);
                bytes[4..8].copy_from_slice(&0x0100000cu32.to_le_bytes());
                bytes[12..16].copy_from_slice(&2u32.to_le_bytes());
            }
        }
        let (src, mut c) = fixture("1", &bytes);
        c.executables = vec!["data".into()];
        c.target = Some(TargetPlatform {
            os: os.into(),
            architecture: arch.into(),
            backend: "cpu".into(),
        });
        let root = tempfile::tempdir().unwrap();
        let m = manager(root.path().into(), vec![c.clone()]);
        m.import_local(&src.path().join("manifest.json")).unwrap();
        c.pin.version = "2".into();
        c.target.as_mut().unwrap().architecture = if arch == "x86_64" {
            "aarch64"
        } else {
            "x86_64"
        }
        .into();
        let root = tempfile::tempdir().unwrap();
        let m = manager(root.path().into(), vec![c]);
        assert!(m.import_local(&src.path().join("manifest.json")).is_err());
    }
}
#[test]
fn selection_policy_and_missing_worker_are_honest() {
    let (_, c) = fixture("1", b"x");
    let root = tempfile::tempdir().unwrap();
    let m = manager(root.path().into(), vec![c]);
    assert!(
        !m.inventory(&TargetPlatform {
            os: "macos".into(),
            architecture: "x86_64".into(),
            backend: "cpu".into()
        })
        .unwrap()
        .ready
    );
    assert!(
        m.inventory(&TargetPlatform {
            os: "macos".into(),
            architecture: "x86_64".into(),
            backend: "metal".into()
        })
        .is_err()
    );
}
#[test]
fn symlink_escape_is_rejected() {
    let (src, c) = fixture("1", b"x");
    let outside = tempfile::tempdir().unwrap();
    fs::write(outside.path().join("data"), b"x").unwrap();
    fs::remove_file(src.path().join("data")).unwrap();
    #[cfg(windows)]
    std::os::windows::fs::symlink_file(outside.path().join("data"), src.path().join("data"))
        .unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink(outside.path().join("data"), src.path().join("data")).unwrap();
    let root = tempfile::tempdir().unwrap();
    let m = manager(root.path().into(), vec![c]);
    assert!(m.import_local(&src.path().join("manifest.json")).is_err());
}
#[test]
#[ignore = "requires explicit locally packaged development resources; no network"]
fn actual_development_import_tamper_and_repair() {
    let repo = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .to_path_buf();
    let catalog: Catalog =
        serde_json::from_slice(&fs::read(repo.join("resources/development-catalog.json")).unwrap())
            .unwrap();
    let root = tempfile::tempdir().unwrap();
    let m = ResourceManager::new(root.path().into(), catalog.clone()).unwrap();
    for c in &catalog.components {
        m.import_local(
            &repo
                .join(".artifacts/task11-input")
                .join(&c.pin.id)
                .join("manifest.json"),
        )
        .unwrap();
        let mut t = c.target.clone().unwrap_or_else(target);
        if t.backend == "any" {
            t.backend = "cpu".into();
        }
        let s = ResourceSelection {
            target: t,
            resources: vec![c.pin.clone()],
        };
        assert!(m.inventory_selection(&s).unwrap().ready);
    }
    assert!(!m.inventory(&target()).unwrap().ready);
    assert!(
        m.inventory(&target())
            .unwrap()
            .items
            .iter()
            .any(|i| i.resource.id == "worker" && !i.ready)
    );
    let c = &catalog.components[0];
    assert_eq!(c.manifest.relative_files.len(), 20);
    let s = selection(c);
    let inv = m.inventory_selection(&s).unwrap();
    let installed = inv.items[0].path.as_ref().unwrap();
    for name in [
        "feather_hubert",
        "pfld_ghost_one",
        "scrfd_2_5g",
        "vgg19",
        "mean_face",
    ] {
        let rel = PathBuf::from(name).join("manifest.json");
        let bytes = fs::read(installed.join(&rel)).unwrap();
        assert_eq!(
            bytes,
            fs::read(repo.join(".artifacts/assets").join(&rel)).unwrap()
        );
        let nested: AssetManifest = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            m.resolve_identity(&nested.identity, &target()).unwrap(),
            installed.join(name)
        );
    }
    fs::write(installed.join("mean_face/model.safetensors"), b"corrupt").unwrap();
    assert!(!m.inventory_selection(&s).unwrap().ready);
    m.import_local(&repo.join(".artifacts/task11-input/public-assets/manifest.json"))
        .unwrap();
    assert!(m.inventory_selection(&s).unwrap().ready);
}
#[test]
fn cache_os_lock_is_released_and_import_can_retry() {
    use fs2::FileExt;
    let (src, c) = fixture("1", b"x");
    let root = tempfile::tempdir().unwrap();
    let m = manager(root.path().into(), vec![c]);
    let f = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(root.path().join(".lock"))
        .unwrap();
    f.lock_exclusive().unwrap();
    assert_eq!(
        m.import_local(&src.path().join("manifest.json"))
            .unwrap_err()
            .code,
        ErrorCode::Busy
    );
    drop(f);
    m.import_local(&src.path().join("manifest.json")).unwrap();
    assert!(m.inventory(&target()).unwrap().ready);
}
#[test]
fn changed_bytes_cannot_reuse_immutable_pin() {
    let (src, a) = fixture("1", b"old");
    let (other, b) = fixture("1", b"new");
    let root = tempfile::tempdir().unwrap();
    let m = manager(root.path().into(), vec![a.clone()]);
    m.import_local(&src.path().join("manifest.json")).unwrap();
    let next = manager(root.path().into(), vec![b]);
    assert!(
        next.import_local(&other.path().join("manifest.json"))
            .is_err()
    );
    assert!(m.inventory(&target()).unwrap().ready);
}
#[test]
fn cancellation_during_download_never_publishes() {
    struct Canceller(Cancellation);
    impl TaskObserver for Canceller {
        fn emit(&self, _: EventPayload) -> Result<()> {
            self.0.cancel();
            Ok(())
        }
    }
    let payload = vec![42u8; 1024 * 1024];
    let (_src, mut c) = fixture("1", &payload);
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    c.manifest.urls.insert(
        "data".into(),
        vec![format!("http://{}/data", l.local_addr().unwrap())],
    );
    let h = std::thread::spawn(move || {
        let (mut s, _) = l.accept().unwrap();
        let _ = s.read(&mut [0; 4096]);
        write!(
            s,
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
            payload.len()
        )
        .unwrap();
        for chunk in payload.chunks(65536) {
            if s.write_all(chunk).is_err() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(30));
        }
    });
    let root = tempfile::tempdir().unwrap();
    let m = manager(root.path().into(), vec![c.clone()]);
    let cancel = Cancellation::new();
    assert_eq!(
        m.prepare(&selection(&c), &Canceller(cancel.clone()), &cancel)
            .unwrap_err()
            .code,
        ErrorCode::Cancelled
    );
    h.join().unwrap();
    assert!(!m.inventory(&target()).unwrap().ready);
    assert!(!root.path().join("test/1").exists());
    assert!(!fs::read_dir(root.path()).unwrap().any(|e| {
        e.unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".staging-")
    }));
}
#[test]
fn cached_symlink_escape_is_not_ready() {
    let (src, c) = fixture("1", b"x");
    let root = tempfile::tempdir().unwrap();
    let m = manager(root.path().into(), vec![c]);
    m.import_local(&src.path().join("manifest.json")).unwrap();
    let file = root.path().join("test/1/data");
    fs::remove_file(&file).unwrap();
    #[cfg(windows)]
    std::os::windows::fs::symlink_file(src.path().join("data"), &file).unwrap();
    #[cfg(unix)]
    std::os::unix::fs::symlink(src.path().join("data"), &file).unwrap();
    assert!(!m.inventory(&target()).unwrap().ready);
}

/// A real nested strict AssetManifest, pinned by an independent bundle manifest.
fn nested_bundle(
    version: &str,
    model_bytes: &[u8],
) -> (tempfile::TempDir, Component, ModelIdentity) {
    let (nested_dir, nested) = fixture("model-v1", model_bytes);
    let (dir, mut bundle) = fixture(version, version.as_bytes());
    fs::create_dir(dir.path().join("model")).unwrap();
    let mut files = BTreeMap::from([(PathBuf::from("data"), version.as_bytes().to_vec())]);
    for name in ["manifest.json", "data"] {
        let bytes = fs::read(nested_dir.path().join(name)).unwrap();
        let relative = PathBuf::from(format!("model/{name}"));
        fs::write(dir.path().join(&relative), &bytes).unwrap();
        files.insert(relative, bytes);
    }
    bundle.manifest.relative_files = files.keys().cloned().collect();
    bundle.manifest.sizes = files
        .iter()
        .map(|(p, b)| (p.clone(), b.len() as u64))
        .collect();
    bundle.manifest.hashes = files.iter().map(|(p, b)| (p.clone(), hash(b))).collect();
    let raw = serde_json::to_vec(&bundle.manifest).unwrap();
    fs::write(dir.path().join("manifest.json"), &raw).unwrap();
    bundle.manifest_sha256 = hash(&raw);
    bundle.manifest_size = raw.len() as u64;
    (dir, bundle, nested.manifest.identity)
}

fn assert_nested_lookup_skips_damage(damage: &str) {
    let (first, a, _) = nested_bundle("01", b"unrelated");
    let (second, b, wanted) = nested_bundle("02", b"wanted");
    let root = tempfile::tempdir().unwrap();
    let m = manager(root.path().into(), vec![a, b]);
    m.import_local(&first.path().join("manifest.json")).unwrap();
    m.import_local(&second.path().join("manifest.json"))
        .unwrap();
    // Generation names are sorted by the resolver: damaged 01 is visited before healthy 02.
    if damage == "metadata" {
        let path = root.path().join("test/01/.component.json");
        let mut c: Component = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        c.manifest.relative_files.push("data".into());
        fs::write(path, serde_json::to_vec(&c).unwrap()).unwrap();
    } else {
        fs::write(
            root.path().join("test/01/model/manifest.json"),
            b"damaged nested manifest",
        )
        .unwrap();
    }
    let expected = root.path().canonicalize().unwrap().join("test/02/model");
    assert_eq!(
        m.resolve_identity(&wanted, &target()).unwrap(),
        expected,
        "{damage}"
    );
}
#[test]
fn nested_lookup_skips_invalid_metadata() {
    assert_nested_lookup_skips_damage("metadata");
}

#[test]
fn runtime_identity_is_not_a_model_dependency() {
    let (source, mut runtime) = fixture("runtime", b"runtime bytes");
    runtime.kind = "media".into();
    runtime.target = Some(target());
    let identity = runtime.manifest.identity.clone();
    let root = tempfile::tempdir().unwrap();
    let m = manager(root.path().into(), vec![runtime]);
    m.import_local(&source.path().join("manifest.json"))
        .unwrap();
    assert!(m.inventory(&target()).unwrap().ready);
    assert_eq!(
        m.resolve_identity(&identity, &target()).unwrap_err().code,
        ErrorCode::ResourceMissing
    );
}

#[test]
fn nested_lookup_skips_unrelated_manifest_corruption() {
    assert_nested_lookup_skips_damage("nested");
}

#[test]
fn nested_lookup_skips_corrupt_matching_bundle_and_never_returns_unverified_match() {
    let (first, a, _) = nested_bundle("01", b"unchanged model");
    let (second, b, wanted) = nested_bundle("02", b"unchanged model");
    let root = tempfile::tempdir().unwrap();
    let m = manager(root.path().into(), vec![a, b]);
    m.import_local(&first.path().join("manifest.json")).unwrap();
    m.import_local(&second.path().join("manifest.json"))
        .unwrap();
    fs::write(root.path().join("test/01/data"), b"xx").unwrap();
    let expected = root.path().canonicalize().unwrap().join("test/02/model");
    assert_eq!(m.resolve_identity(&wanted, &target()).unwrap(), expected);
    fs::write(root.path().join("test/02/data"), b"yy").unwrap();
    let error = m.resolve_identity(&wanted, &target()).unwrap_err();
    assert_eq!(error.code, ErrorCode::ResourceMissing);
    assert!(
        error
            .evidence
            .iter()
            .any(|e| e.contains("01") && e.contains("SHA-256"))
    );
    assert!(
        error
            .evidence
            .iter()
            .any(|e| e.contains("02") && e.contains("SHA-256"))
    );
    assert!(error.evidence.len() <= 8);
    assert!(error.evidence.iter().all(|e| e.chars().count() <= 512));
}

#[test]
fn nested_lookup_bounds_candidate_diagnostics_and_preserves_root_errors() {
    let root = tempfile::tempdir().unwrap();
    let (_, base, wanted) = nested_bundle("base", b"wanted");
    let cache = root.path().join("cache");
    let m = manager(cache.clone(), vec![base]);
    for index in 0..12 {
        let version = format!("{index:02}");
        let (source, c, _) = nested_bundle(&version, b"other");
        let writer = manager(cache.clone(), vec![c]);
        writer
            .import_local(&source.path().join("manifest.json"))
            .unwrap();
        let metadata = cache.join("test").join(&version).join(".component.json");
        let mut c: Component = serde_json::from_slice(&fs::read(&metadata).unwrap()).unwrap();
        c.manifest.min_engine_version = "x".repeat(2000);
        fs::write(metadata, serde_json::to_vec(&c).unwrap()).unwrap();
    }
    let error = m.resolve_identity(&wanted, &target()).unwrap_err();
    assert_eq!(error.code, ErrorCode::ResourceMissing);
    assert_eq!(error.evidence.len(), 8);
    assert!(error.evidence.iter().all(|e| e.chars().count() <= 512));
    // Rename our own cache rather than deleting fixtures; opening its root must fail fatally.
    fs::rename(&cache, root.path().join("moved-cache")).unwrap();
    let fatal = m.resolve_identity(&wanted, &target()).unwrap_err();
    assert_eq!(fatal.code, ErrorCode::ResourceCorrupt);
    assert!(fatal.evidence.is_empty());
}
