//! Committed training state. Counters describe the next step to execute.
use crate::{
    ExecutionDevice,
    data::{
        prepared::PreparedProject,
        sampler::{PairSampler, SamplerState},
    },
    models::feathertalk::FeatherTalk,
};
use burn::{
    optim::{AdamConfig, GradientsParams, ModuleOptimizer, decay::WeightDecayConfig},
    tensor::Device,
};
use feathertalk_core::*;
use serde::{Deserialize, Serialize};
use std::{
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

pub(crate) fn invalid(e: impl std::fmt::Display) -> AppError {
    AppError::new(
        ErrorCode::ResourceCorrupt,
        "training-checkpoint",
        e.to_string(),
    )
}
static RANDOM: Mutex<()> = Mutex::new(());
/// Application-owned seed replay, not an export of backend RNG internals.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SeedPolicy {
    pub version: u32,
    pub base_seed: u64,
}
impl SeedPolicy {
    pub fn new(base_seed: u64) -> Self {
        Self {
            version: 1,
            base_seed,
        }
    }
    fn derive(&self, domain: u64, index: u64) -> u64 {
        let mut z = self.base_seed ^ domain ^ index.wrapping_mul(0x9e3779b97f4a7c15);
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    pub fn initialization_seed(&self) -> u64 {
        self.derive(0x696e697469616c00, 0)
    }
    pub fn step_seed(&self, next_step: u64) -> u64 {
        self.derive(0x747261696e737465, next_step)
    }
}
/// All application random tensor work must use this boundary. Drains lazy work
/// before releasing the process-wide seed lock; callers must not nest boundaries.
pub fn with_seed<T>(device: &Device, seed: u64, work: impl FnOnce() -> Result<T>) -> Result<T> {
    let _guard = RANDOM.lock().map_err(invalid)?;
    device.sync().map_err(invalid)?;
    device.seed(seed);
    let value = work();
    device.sync().map_err(invalid)?;
    value
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingMetadata {
    pub schema_version: u32,
    pub project_id: ProjectId,
    pub run_id: TrainingRunId,
    pub source: SourceVideo,
    pub project_model_identity: Option<ModelIdentity>,
    pub preprocessing: PreprocessingIdentity,
    pub prepared_identity: String,
    pub prepared_manifest_sha256: String,
    pub architecture: String,
    /// Frozen loss dependency. Pristine step0 may await selection.
    pub vgg_identity: Option<ModelIdentity>,
    pub config: TrainConfig,
    pub batch_mode: BatchMode,
    pub requested_batch: u32,
    pub actual_batch: u32,
    /// Completed epochs; equals sampler epoch. global_step counts committed updates.
    pub epoch: u32,
    pub global_step: u64,
    pub sampler: SamplerState,
    pub seed_policy: SeedPolicy,
}
pub struct TrainingState {
    pub model: FeatherTalk,
    optimizer: ModuleOptimizer,
    optimizer_config: [f64; 5],
    pub metadata: TrainingMetadata,
    pub(crate) project_directory: PathBuf,
}

/// An owned binding for one task. Hashes are verified on entry; the task assumes
/// the pinned cache generation stays immutable until it ends.
pub(super) struct TrainingData {
    prepared: Arc<PreparedProject>,
    source: SourceVideo,
    model: Option<ModelIdentity>,
}
impl TrainingData {
    pub(super) fn verify(state: &TrainingState, prepared: &PreparedProject) -> Result<Self> {
        let pinned =
            super::checkpoint::validate_metadata(state.project_directory(), &state.metadata)?;
        Self::bind(state, prepared, &pinned)
    }

    pub(super) fn bind(
        state: &TrainingState,
        prepared: &PreparedProject,
        pinned: &PreparedProject,
    ) -> Result<Self> {
        Self::check_supplied(prepared, pinned)?;
        Self::from_verified(state, pinned.clone())
    }

    fn check_supplied(prepared: &PreparedProject, pinned: &PreparedProject) -> Result<()> {
        if prepared != pinned {
            return Err(invalid(
                "Prepared in-memory data differs from pinned generation",
            ));
        }
        Ok(())
    }

    // Only prepared data obtained directly from preparation or a verified load.
    pub(super) fn from_verified(state: &TrainingState, prepared: PreparedProject) -> Result<Self> {
        let data = Self {
            prepared: Arc::new(prepared),
            source: state.metadata.source.clone(),
            model: state.metadata.project_model_identity.clone(),
        };
        data.for_state(state)?;
        Ok(data)
    }

    pub(super) fn for_state(&self, state: &TrainingState) -> Result<&Arc<PreparedProject>> {
        let m = &state.metadata;
        let prepared = &self.prepared;
        if m.project_id != prepared.project_id
            || m.prepared_identity != prepared.identity
            || m.preprocessing != prepared.preprocessing
            || state.project_directory != prepared.project_directory
            || m.source != self.source
            || m.project_model_identity != self.model
            || m.prepared_manifest_sha256 != prepared.manifest_sha256
        {
            return Err(invalid(
                "Training state and verified prepared project differ",
            ));
        }
        Ok(prepared)
    }
}
/// Shared persistence/execution domain: lr and epsilon must remain positive in
/// F32 (representable subnormals are allowed). Beta/decay may round to zero.
pub(crate) fn validate_adam_config(c: &TrainConfig) -> Result<()> {
    c.validate()?;
    for v in [c.beta1, c.beta2, c.epsilon, c.weight_decay, c.lr] {
        if !(v as f32).is_finite() {
            return Err(invalid("Adam configuration exceeds F32 range"));
        }
    }
    if c.lr as f32 <= 0. || c.epsilon as f32 <= 0. || c.beta1 as f32 >= 1. || c.beta2 as f32 >= 1. {
        return Err(invalid("Adam configuration rounds outside supported range"));
    }
    Ok(())
}
pub(crate) fn adam(c: &TrainConfig) -> Result<ModuleOptimizer> {
    validate_adam_config(c)?;
    Ok(AdamConfig::new()
        .with_beta_1(c.beta1 as f32)
        .with_beta_2(c.beta2 as f32)
        .with_epsilon(c.epsilon as f32)
        .with_amsgrad(false)
        .with_weight_decay(
            (c.weight_decay != 0.).then(|| WeightDecayConfig::new(c.weight_decay as f32)),
        )
        .init())
}
impl TrainingState {
    pub(crate) fn from_parts(
        model: FeatherTalk,
        optimizer: ModuleOptimizer,
        metadata: TrainingMetadata,
        project_directory: PathBuf,
    ) -> Self {
        let c = &metadata.config;
        let optimizer_config = [c.beta1, c.beta2, c.epsilon, c.weight_decay, c.lr];
        Self {
            model,
            optimizer,
            optimizer_config,
            metadata,
            project_directory,
        }
    }
    /// Start an independent tentative update without a device-to-host roundtrip.
    /// Burn shares tensor storage until use; the optimizer owns its context map,
    /// and FeatherTalk's normalization buffers have an independent mutex.
    pub(super) fn step_candidate(&self) -> Self {
        Self {
            model: self.model.clone(),
            optimizer: self.optimizer.clone(),
            optimizer_config: self.optimizer_config,
            metadata: self.metadata.clone(),
            project_directory: self.project_directory.clone(),
        }
    }
    /// Owned record export; optimizer configuration cannot be replaced by callers.
    pub fn optimizer_bytes(&self) -> Result<Vec<u8>> {
        self.validate_optimizer_config()?;
        Ok(self.optimizer.into_bytes().map_err(invalid)?.to_vec())
    }
    pub fn set_learning_rate(&mut self, learning_rate: f64) -> Result<()> {
        self.validate_optimizer_config()?;
        let mut config = self.metadata.config.clone();
        config.lr = learning_rate;
        validate_adam_config(&config)?;
        self.metadata.config = config;
        self.optimizer_config[4] = learning_rate;
        Ok(())
    }
    fn validate_optimizer_config(&self) -> Result<()> {
        let c = &self.metadata.config;
        validate_adam_config(c)?;
        if self.optimizer_config != [c.beta1, c.beta2, c.epsilon, c.weight_decay, c.lr] {
            return Err(invalid(
                "Adam hyperparameters changed without rebuilding state",
            ));
        }
        Ok(())
    }
    /// One optimizer operation only. The trainer commits sampler/epoch/global_step
    /// together after this succeeds; snapshots reject incomplete optimizer counters.
    pub fn apply_gradients(&mut self, gradients: GradientsParams) -> Result<()> {
        self.validate_optimizer_config()?;
        self.model = self
            .optimizer
            .step(self.metadata.config.lr, self.model.clone(), gradients);
        Ok(())
    }
    pub fn fresh(
        request: &TrainRequest,
        prepared: &PreparedProject,
        device: &ExecutionDevice,
    ) -> Result<Self> {
        Self::fresh_with_data(request, prepared, device).map(|(state, _)| state)
    }

    pub(super) fn fresh_with_data(
        request: &TrainRequest,
        prepared: &PreparedProject,
        device: &ExecutionDevice,
    ) -> Result<(Self, TrainingData)> {
        let root = std::fs::canonicalize(&prepared.project_directory).map_err(invalid)?;
        let project = super::checkpoint::read_project(&root)?;
        super::checkpoint::validate_source(&root, &project.source_video)?;
        let pinned = crate::data::prepared::reopen_generation(
            &root,
            &project,
            &prepared.preprocessing.cache_directory,
            &prepared.manifest_sha256,
        )?;
        TrainingData::check_supplied(prepared, &pinned)?;
        Self::fresh_verified(request, pinned, &project, device)
    }

    pub(super) fn fresh_verified(
        request: &TrainRequest,
        prepared: PreparedProject,
        project: &ProjectManifest,
        device: &ExecutionDevice,
    ) -> Result<(Self, TrainingData)> {
        validate_adam_config(&request.config)?;
        if request.resume.is_some()
            || request.project_id != prepared.project_id
            || request.project_id != project.id
            || prepared.preprocessing.source_sha256 != project.source_video.sha256
            || project
                .preprocessing
                .as_ref()
                .is_some_and(|p| p != &prepared.preprocessing)
            || prepared.usable_frames < 2
        {
            return Err(invalid(
                "Fresh training requires matching prepared project and at least one pair",
            ));
        }
        let policy = SeedPolicy::new(request.config.seed);
        let model = with_seed(&device.device, policy.initialization_seed(), || {
            Ok(FeatherTalk::new(&device.device.clone().autodiff()))
        })?;
        let metadata = TrainingMetadata {
            schema_version: 2,
            project_id: project.id,
            run_id: TrainingRunId::new(),
            source: project.source_video.clone(),
            project_model_identity: project.model_identity.clone(),
            preprocessing: prepared.preprocessing.clone(),
            prepared_identity: prepared.identity.clone(),
            prepared_manifest_sha256: prepared.manifest_sha256.clone(),
            architecture: "feathertalk-default-v1-447".into(),
            vgg_identity: None,
            config: request.config.clone(),
            batch_mode: request.batch_mode,
            requested_batch: request.config.batch_size,
            actual_batch: request.config.batch_size,
            epoch: 0,
            global_step: 0,
            sampler: PairSampler::initial_state(&prepared, request.config.seed),
            seed_policy: policy,
        };
        let state = Self::from_parts(
            model,
            adam(&request.config)?,
            metadata,
            prepared.project_directory.clone(),
        );
        let data = TrainingData::from_verified(&state, prepared)?;
        Ok((state, data))
    }
    pub fn project_directory(&self) -> &Path {
        &self.project_directory
    }
    pub fn validate_prepared(&self, prepared: &PreparedProject) -> Result<()> {
        TrainingData::verify(self, prepared).map(|_| ())
    }
}
