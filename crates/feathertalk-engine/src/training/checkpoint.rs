//! Immutable checkpoint generations; project.json is the publication point.
use super::{
    checkpoint_validation as validation,
    state::{TrainingMetadata, TrainingState, adam, invalid},
};
use crate::{
    ExecutionDevice,
    data::prepared::{file_hash, reopen_generation},
    models::feathertalk::FeatherTalk,
};
use burn_store::{BurnpackStore, ModuleSnapshot};
use feathertalk_core::*;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::Write,
    path::{Component, Path, PathBuf},
};

pub(crate) fn digest(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
pub(crate) fn read_project(root: &Path) -> Result<ProjectManifest> {
    let path = contained(root, Path::new("project.json"))?;
    let project: ProjectManifest =
        serde_json::from_slice(&fs::read(path).map_err(invalid)?).map_err(invalid)?;
    if project.schema_version != 1 {
        return Err(invalid("Unsupported project schema"));
    }
    Ok(project)
}
fn contained(root: &Path, relative: &Path) -> Result<PathBuf> {
    if relative.as_os_str().is_empty()
        || relative
            .components()
            .any(|c| !matches!(c, Component::Normal(_)))
    {
        return Err(invalid("Unsafe checkpoint path"));
    }
    let path = fs::canonicalize(root.join(relative)).map_err(invalid)?;
    if !path.starts_with(root) {
        return Err(invalid("Checkpoint path escapes project"));
    }
    Ok(path)
}
pub(super) fn validate_metadata(root: &Path, m: &TrainingMetadata) -> Result<()> {
    m.config.validate()?;
    if m.schema_version != 1
        || m.architecture != "feathertalk-default-v1-447"
        || m.seed_policy.version != 1
        || m.seed_policy.base_seed != m.config.seed
        || m.sampler.seed != m.config.seed
        || m.epoch as u64 != m.sampler.epoch
        || m.epoch > m.config.epochs
        || m.requested_batch != m.config.batch_size
        || m.actual_batch == 0
        || m.actual_batch > m.requested_batch
        || (m.batch_mode == BatchMode::Fixed && m.actual_batch != m.requested_batch)
    {
        return Err(invalid("Invalid training metadata/counters/batch policy"));
    }
    let project = read_project(root)?;
    if project.id != m.project_id
        || project.source_video != m.source
        || project.model_identity != m.project_model_identity
    {
        return Err(invalid(
            "Checkpoint belongs to different project/source/model",
        ));
    }
    let source = if m.source.path.is_absolute() {
        m.source.path.clone()
    } else {
        root.join(&m.source.path)
    };
    if fs::metadata(&source).map_err(invalid)?.len() != m.source.size_bytes
        || file_hash(&source)? != m.source.sha256
    {
        return Err(invalid("Source media changed"));
    }
    let prepared = reopen_generation(
        root,
        &project,
        &m.preprocessing.cache_directory,
        &m.prepared_manifest_sha256,
    )?;
    if prepared.identity != m.prepared_identity || prepared.preprocessing != m.preprocessing {
        return Err(invalid("Checkpoint data identity differs"));
    }
    crate::data::sampler::PairSampler::restore(&prepared, m.sampler.clone())?;
    Ok(())
}
pub(crate) fn model_bytes(model: &FeatherTalk) -> Result<Vec<u8>> {
    let mut store = BurnpackStore::from_bytes(None);
    model.save_into(&mut store).map_err(invalid)?;
    Ok(store.get_bytes().map_err(invalid)?.to_vec())
}
/// Fully materialized host bytes: no tensor handles, lazy collectors or shared BN buffers.
/// Restore always validates against the architecture and project before publishing a state.
pub struct MaterializedSnapshot {
    metadata: TrainingMetadata,
    model: Vec<u8>,
    optimizer: Vec<u8>,
}
impl MaterializedSnapshot {
    pub fn metadata(&self) -> &TrainingMetadata {
        &self.metadata
    }
    pub fn capture(state: &TrainingState) -> Result<Self> {
        validate_metadata(&state.project_directory, &state.metadata)?;
        let model = model_bytes(&state.model)?;
        let optimizer = state.optimizer_bytes()?;
        let template = super::state::with_seed(
            &state.model.devices()[0],
            state.metadata.seed_policy.initialization_seed(),
            || model_bytes(&FeatherTalk::new(&state.model.devices()[0])),
        )?;
        let records = validation::validate_model(&model, &template)?;
        validation::validate_adam(&optimizer, &records, state.metadata.global_step)?;
        Ok(Self {
            metadata: state.metadata.clone(),
            model,
            optimizer,
        })
    }
    pub fn restore(
        &self,
        project_directory: &Path,
        device: &ExecutionDevice,
    ) -> Result<TrainingState> {
        let root = fs::canonicalize(project_directory).map_err(invalid)?;
        validate_metadata(&root, &self.metadata)?;
        super::state::with_seed(
            &device.device,
            self.metadata.seed_policy.initialization_seed(),
            || {
                let mut model = FeatherTalk::new(&device.device.clone().autodiff());
                let records = validation::validate_model(&self.model, &model_bytes(&model)?)?;
                validation::validate_adam(&self.optimizer, &records, self.metadata.global_step)?;
                model
                    .load_from(&mut BurnpackStore::from_bytes(Some(
                        burn_pack::Bytes::from_bytes_vec(self.model.clone()),
                    )))
                    .map_err(invalid)?;
                let optimizer = adam(&self.metadata.config)?
                    .from_bytes(burn_pack::Bytes::from_bytes_vec(self.optimizer.clone()))
                    .map_err(invalid)?;
                validation::equal_pack(&self.model, &model_bytes(&model)?)?;
                validation::equal_pack(&self.optimizer, &optimizer.into_bytes().map_err(invalid)?)?;
                Ok(TrainingState::from_parts(
                    model,
                    optimizer,
                    self.metadata.clone(),
                    root,
                ))
            },
        )
    }
    /// Private probe handoff. Directory must be new; this never publishes a ModelRef.
    pub fn write_private(&self, directory: &Path) -> Result<()> {
        fs::create_dir(directory).map_err(invalid)?;
        write_sync(&directory.join("model.bpk"), &self.model)?;
        write_sync(&directory.join("adam.bpk"), &self.optimizer)?;
        let manifest = DiskManifest {
            metadata: self.metadata.clone(),
            model_sha256: digest(&self.model),
            model_size: self.model.len() as u64,
            optimizer_sha256: digest(&self.optimizer),
            optimizer_size: self.optimizer.len() as u64,
        };
        write_sync(
            &directory.join("manifest.json"),
            &serde_json::to_vec(&manifest).map_err(invalid)?,
        )
    }
    pub fn read_private(directory: &Path) -> Result<Self> {
        let root = fs::canonicalize(directory).map_err(invalid)?;
        let bytes = fs::read(contained(&root, Path::new("manifest.json"))?).map_err(invalid)?;
        Self::read_manifest(&root, &bytes)
    }
    fn read_manifest(root: &Path, bytes: &[u8]) -> Result<Self> {
        let m: DiskManifest = serde_json::from_slice(bytes).map_err(invalid)?;
        let model = fs::read(contained(&root, Path::new("model.bpk"))?).map_err(invalid)?;
        let optimizer = fs::read(contained(&root, Path::new("adam.bpk"))?).map_err(invalid)?;
        if model.len() as u64 != m.model_size
            || digest(&model) != m.model_sha256
            || optimizer.len() as u64 != m.optimizer_size
            || digest(&optimizer) != m.optimizer_sha256
        {
            return Err(invalid("Checkpoint file integrity mismatch"));
        }
        Ok(Self {
            metadata: m.metadata,
            model,
            optimizer,
        })
    }
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DiskManifest {
    metadata: TrainingMetadata,
    model_sha256: String,
    model_size: u64,
    optimizer_sha256: String,
    optimizer_size: u64,
}
fn write_sync(path: &Path, bytes: &[u8]) -> Result<()> {
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(path)
        .map_err(invalid)?;
    file.write_all(bytes).map_err(invalid)?;
    file.sync_all().map_err(invalid)
}
struct Cleanup(PathBuf);
impl Drop for Cleanup {
    fn drop(&mut self) {
        if self.0.is_dir() {
            let _ = fs::remove_dir_all(&self.0);
        } else {
            let _ = fs::remove_file(&self.0);
        }
    }
}
/// destination is a new generation directory under this state's canonical project.
pub fn save_checkpoint(state: &TrainingState, destination: &Path) -> Result<ModelRef> {
    save_with_hook(state, destination, |_| Ok(()))
}
pub(super) fn save_with_hook(
    state: &TrainingState,
    destination: &Path,
    hook: impl Fn(&str) -> Result<()>,
) -> Result<ModelRef> {
    let root = &state.project_directory;
    let parent = fs::canonicalize(
        destination
            .parent()
            .ok_or_else(|| invalid("Missing destination parent"))?,
    )
    .map_err(invalid)?;
    if !parent.starts_with(root) || destination.exists() {
        return Err(invalid("Destination must be new and project-owned"));
    }
    let name = destination
        .file_name()
        .ok_or_else(|| invalid("Missing destination name"))?;
    let destination = parent.join(name);
    let snapshot = MaterializedSnapshot::capture(state)?;
    let id = ProjectId::new().to_string();
    let stage = parent.join(format!(".checkpoint-stage-{id}"));
    let _cleanup = Cleanup(stage.clone());
    snapshot.write_private(&stage)?;
    hook("staged")?;
    fs::rename(&stage, &destination).map_err(invalid)?;
    hook("published")?;
    // All project writers must honor this OS lock. It is released on process exit.
    let lock = root.join(".project-write.lock");
    let lock_file = fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .read(true)
        .write(true)
        .open(&lock)
        .map_err(invalid)?;
    lock_file.try_lock().map_err(invalid)?;
    validate_metadata(root, &state.metadata)?;
    let before = fs::read(root.join("project.json")).map_err(invalid)?;
    let mut project: ProjectManifest = serde_json::from_slice(&before).map_err(invalid)?;
    if project.id != state.metadata.project_id
        || project.source_video != state.metadata.source
        || project.model_identity != state.metadata.project_model_identity
    {
        return Err(invalid("Project binding changed while staging"));
    }
    let model_ref = ModelRef {
        project_id: state.metadata.project_id,
        run_id: state.metadata.run_id,
        checkpoint_id: id.clone(),
    };
    let manifest = fs::read(destination.join("manifest.json")).map_err(invalid)?;
    project.checkpoints.push(CheckpointEntry {
        model_ref: model_ref.clone(),
        relative_path: destination
            .strip_prefix(root)
            .map_err(invalid)?
            .join("manifest.json"),
        sha256: digest(&manifest),
        size_bytes: manifest.len() as u64,
        epoch: state.metadata.epoch,
        step: state.metadata.global_step,
        config: state.metadata.config.clone(),
        identity: ModelIdentity {
            kind: "feathertalk".into(),
            architecture_version: state.metadata.architecture.clone(),
            source_sha256: state.metadata.source.sha256.clone(),
            weights_sha256: digest(&snapshot.model),
        },
    });
    let pending = root.join(format!(".project-{id}.tmp"));
    let _pending = Cleanup(pending.clone());
    write_sync(
        &pending,
        &serde_json::to_vec_pretty(&project).map_err(invalid)?,
    )?;
    hook("index")?;
    if fs::read(root.join("project.json")).map_err(invalid)? != before {
        return Err(invalid("Project changed concurrently"));
    }
    fs::rename(pending, root.join("project.json")).map_err(invalid)?;
    Ok(model_ref)
}
pub fn load_checkpoint(
    project_directory: &Path,
    model_ref: &ModelRef,
    device: &ExecutionDevice,
) -> Result<TrainingState> {
    let root = fs::canonicalize(project_directory).map_err(invalid)?;
    let project = read_project(&root)?;
    let entries: Vec<_> = project
        .checkpoints
        .iter()
        .filter(|e| e.model_ref == *model_ref)
        .collect();
    if project.id != model_ref.project_id || entries.len() != 1 {
        return Err(invalid("ModelRef is not uniquely committed in project"));
    }
    let entry = entries[0];
    let path = contained(&root, &entry.relative_path)?;
    let bytes = fs::read(&path).map_err(invalid)?;
    if bytes.len() as u64 != entry.size_bytes
        || digest(&bytes) != entry.sha256
        || path.file_name().and_then(|s| s.to_str()) != Some("manifest.json")
    {
        return Err(invalid("Committed checkpoint integrity mismatch"));
    }
    let snapshot = MaterializedSnapshot::read_manifest(path.parent().unwrap(), &bytes)?;
    let m = snapshot.metadata();
    if m.project_id != model_ref.project_id
        || m.run_id != model_ref.run_id
        || m.epoch != entry.epoch
        || m.global_step != entry.step
        || m.config != entry.config
        || entry.identity.weights_sha256 != digest(&snapshot.model)
        || entry.identity.architecture_version != m.architecture
        || entry.identity.source_sha256 != m.source.sha256
        || entry.identity.kind != "feathertalk"
    {
        return Err(invalid("Committed checkpoint metadata mismatch"));
    }
    snapshot.restore(&root, device)
}

use burn::module::Module;
