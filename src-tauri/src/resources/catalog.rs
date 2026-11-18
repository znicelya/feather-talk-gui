use super::fail;
use feathertalk_core::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::{Component as PathComponent, Path, PathBuf},
};
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileSpec {
    pub size: u64,
    pub sha256: String,
    pub urls: Vec<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Catalog {
    pub schema_version: u32,
    pub components: Vec<Component>,
    pub defaults: Vec<ResourceSelection>,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Component {
    pub pin: ResourcePin,
    pub target: Option<TargetPlatform>,
    pub kind: String,
    pub manifest: AssetManifest,
    pub manifest_sha256: String,
    pub manifest_size: u64,
    pub manifest_urls: Vec<String>,
    pub supplemental: BTreeMap<PathBuf, FileSpec>,
    pub executables: Vec<PathBuf>,
    pub notices: Vec<PathBuf>,
    pub redistribution_approved: bool,
    pub license: String,
}
pub(crate) fn normal(path: &Path) -> Result<()> {
    if path.as_os_str().is_empty()
        || path
            .components()
            .any(|c| !matches!(c, PathComponent::Normal(_)))
        || path
            .to_string_lossy()
            .contains(['\\', ':', '*', '?', '"', '<', '>', '|'])
        || path.to_string_lossy().chars().any(|c| c.is_control())
        || path
            .to_string_lossy()
            .split('/')
            .any(|s| s.is_empty() || s == "." || s == "..")
    {
        return Err(fail("unsafe relative path"));
    }
    for part in path.iter() {
        let s = part.to_string_lossy();
        let stem = s.split('.').next().unwrap_or("").to_ascii_uppercase();
        if s.ends_with(['.', ' '])
            || matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL")
            || ((stem.starts_with("COM") || stem.starts_with("LPT"))
                && stem.len() == 4
                && stem.as_bytes()[3].is_ascii_digit())
        {
            return Err(fail("nonportable path"));
        }
    }
    Ok(())
}
pub(crate) fn validate_target(t: &TargetPlatform) -> Result<()> {
    let valid = match (t.os.as_str(), t.architecture.as_str()) {
        ("windows" | "linux", "x86_64") => ["cpu", "cuda", "vulkan"].contains(&t.backend.as_str()),
        ("macos", "x86_64") => t.backend == "cpu",
        ("macos", "aarch64") => ["cpu", "metal"].contains(&t.backend.as_str()),
        _ => false,
    };
    if !valid {
        return Err(AppError::new(
            ErrorCode::UnsupportedBackend,
            "resources",
            "Unsupported target/backend (Intel Mac supports CPU only)",
        ));
    }
    Ok(())
}
pub(crate) fn token(s: &str) -> Result<()> {
    if s.is_empty()
        || s.starts_with('.')
        || !s
            .bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.')
        || s == "."
        || s == ".."
    {
        return Err(fail("unsafe resource identifier"));
    }
    normal(Path::new(s))
}
impl Component {
    pub(crate) fn files(&self) -> BTreeMap<PathBuf, FileSpec> {
        let mut files = self.supplemental.clone();
        for p in &self.manifest.relative_files {
            files.insert(
                p.clone(),
                FileSpec {
                    size: self.manifest.sizes[p],
                    sha256: self.manifest.hashes[p].clone(),
                    urls: self.manifest.urls.get(p).cloned().unwrap_or_default(),
                },
            );
        }
        files.insert(
            "manifest.json".into(),
            FileSpec {
                size: self.manifest_size,
                sha256: self.manifest_sha256.clone(),
                urls: self.manifest_urls.clone(),
            },
        );
        files
    }
    pub(crate) fn validate(&self) -> Result<()> {
        token(&self.pin.id)?;
        token(&self.pin.version)?;
        let m = &self.manifest;
        let set: BTreeSet<_> = m.relative_files.iter().collect();
        if set.len() != m.relative_files.len()
            || set.len() != m.sizes.len()
            || set.len() != m.hashes.len()
            || m.sizes.keys().any(|k| !set.contains(k))
            || m.hashes.keys().any(|k| !set.contains(k))
            || m.urls.keys().any(|k| !set.contains(k))
        {
            return Err(fail("duplicate or inconsistent manifest inventory"));
        }
        if m.relative_files.contains(&PathBuf::from("manifest.json"))
            || self
                .supplemental
                .keys()
                .any(|p| set.contains(p) || p == Path::new("manifest.json"))
        {
            return Err(fail("conflicting package inventory"));
        }
        let minimum = semver::Version::parse(&m.min_engine_version).map_err(fail)?;
        if minimum > semver::Version::parse(env!("CARGO_PKG_VERSION")).unwrap() {
            return Err(fail("requires newer engine"));
        }
        let files = self.files();
        let mut portable = BTreeSet::new();
        let folded: BTreeSet<_> = files
            .keys()
            .map(|p| p.to_string_lossy().to_lowercase())
            .collect();
        for (p, f) in &files {
            normal(p)?;
            if p.to_string_lossy().eq_ignore_ascii_case(".component.json")
                || !portable.insert(p.to_string_lossy().to_lowercase())
                || f.sha256.len() != 64
                || !f
                    .sha256
                    .bytes()
                    .all(|c| c.is_ascii_hexdigit() && !c.is_ascii_uppercase())
            {
                return Err(fail("conflicting paths or invalid SHA-256"));
            }
            for ancestor in p.ancestors().skip(1) {
                if folded.contains(&ancestor.to_string_lossy().to_lowercase()) {
                    return Err(fail("file/directory path conflict"));
                }
            }
            for url in &f.urls {
                let u = reqwest::Url::parse(url).map_err(fail)?;
                if u.scheme() != "https"
                    && !(u.scheme() == "http"
                        && u.host_str()
                            .is_some_and(|h| h == "127.0.0.1" || h == "[::1]" || h == "localhost"))
                {
                    return Err(fail(
                        "downloads require HTTPS (HTTP allowed only on loopback)",
                    ));
                }
            }
        }
        for p in self.executables.iter().chain(&self.notices) {
            if !files.contains_key(p) {
                return Err(fail("executable/notice missing from inventory"));
            }
        }
        if !self.executables.is_empty() && self.target.is_none() {
            return Err(fail("executables require target"));
        }
        Ok(())
    }
}
impl Catalog {
    pub(crate) fn validate(&self) -> Result<()> {
        if self.schema_version != 1 {
            return Err(fail("unsupported catalog version"));
        }
        let mut pins = BTreeSet::new();
        for c in &self.components {
            c.validate()?;
            if !pins.insert((&c.pin.id, &c.pin.version)) {
                return Err(fail("duplicate catalog pin"));
            }
        }
        Ok(())
    }
}
