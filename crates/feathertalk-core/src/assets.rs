use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::PathBuf};
/// Identity of a converted component or trained character model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelIdentity {
    pub kind: String,
    pub architecture_version: String,
    pub source_sha256: String,
    pub weights_sha256: String,
}
/// All file keys and paths are relative to the asset root. Digests are SHA-256.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssetManifest {
    pub identity: ModelIdentity,
    pub relative_files: Vec<PathBuf>,
    pub sizes: BTreeMap<PathBuf, u64>,
    pub hashes: BTreeMap<PathBuf, String>,
    pub urls: BTreeMap<PathBuf, Vec<String>>,
    pub min_engine_version: String,
    pub provenance: String,
}
