//! Verify component files before constructing any device tensors.
use feathertalk_core::*;
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeMap,
    path::{Component, Path},
};
pub(crate) fn error(message: impl ToString) -> AppError {
    AppError::new(
        ErrorCode::ResourceCorrupt,
        "model-assets",
        message.to_string(),
    )
}
pub(crate) struct VerifiedAssets {
    pub manifest: AssetManifest,
    pub files: BTreeMap<String, Vec<u8>>,
}
impl VerifiedAssets {
    pub fn load(root: &Path) -> Result<Self> {
        let manifest: AssetManifest =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json")).map_err(error)?)
                .map_err(error)?;
        if manifest.min_engine_version != env!("CARGO_PKG_VERSION") {
            return Err(error("Unsupported minimum engine version"));
        }
        let mut files = BTreeMap::new();
        for path in &manifest.relative_files {
            if path
                .components()
                .any(|c| !matches!(c, Component::Normal(_)))
            {
                return Err(error("Unsafe asset path"));
            }
            let bytes = std::fs::read(root.join(path)).map_err(error)?;
            if manifest.sizes.get(path) != Some(&(bytes.len() as u64))
                || manifest.hashes.get(path) != Some(&format!("{:x}", Sha256::digest(&bytes)))
            {
                return Err(error(format!(
                    "Asset size or SHA256 mismatch: {}",
                    path.display()
                )));
            }
            let key = path
                .to_str()
                .ok_or_else(|| error("Non UTF8 asset path"))?
                .to_owned();
            if files.insert(key, bytes).is_some() {
                return Err(error("Duplicate asset path"));
            }
        }
        if manifest.hashes.get(Path::new("model.safetensors"))
            != Some(&manifest.identity.weights_sha256)
        {
            return Err(error("Weights identity mismatch"));
        }
        Ok(Self { manifest, files })
    }
}
