//! Public run orchestration. Worker owns terminal events.
use super::{
    batch_probe::{ProbeContext, TrainingRuntime, probe_batch_size_validated},
    checkpoint::{load_checkpoint_with_data, save_checkpoint},
    metrics::SampledMetrics,
    state::{TrainingData, TrainingState, invalid},
};
use crate::{ExecutionDevice, data::prepared::PreparedProject, models::vgg::VggFeatures};
use feathertalk_core::*;
use serde::{Deserialize, Serialize};
use std::{fs, io::ErrorKind, path::Path, time::Instant};
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StoppingReason {
    Completed,
    Cancelled,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingResult {
    pub committed_model: ModelRef,
    pub run_id: TrainingRunId,
    pub epoch: u32,
    pub global_step: u64,
    pub stopping_reason: StoppingReason,
}
fn publish(state: &TrainingState, observer: &dyn TaskObserver) -> Result<ModelRef> {
    let root = state.project_directory();
    let checkpoints = root.join("checkpoints");
    match fs::create_dir(&checkpoints) {
        Ok(()) => {}
        Err(e) if e.kind() == ErrorKind::AlreadyExists => {}
        Err(e) => return Err(invalid(e)),
    }
    let checkpoints = checkpoints.canonicalize().map_err(invalid)?;
    if !checkpoints.starts_with(root) || !checkpoints.is_dir() {
        return Err(invalid("Checkpoint directory must be inside the project"));
    }
    let reference = save_checkpoint(
        state,
        &checkpoints.join(format!("generation-{}", TrainingRunId::new())),
    )?;
    observer
        .emit(EventPayload::CheckpointCommitted {
            model_ref: reference.clone(),
        })
        .map_err(|mut e| {
            e.evidence.push(format!(
                "Checkpoint is committed despite observer failure: {reference:?}"
            ));
            e
        })?;
    #[cfg(all(debug_assertions, feature = "dev-worker-acceptance"))]
    super::worker_acceptance::after_checkpoint_event(root)?;
    Ok(reference)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::checkpoint::load_checkpoint;
    use crate::training::checkpoint_tests::{device, fixture};
    struct Observe;
    impl TaskObserver for Observe {
        fn emit(&self, _: EventPayload) -> Result<()> {
            Ok(())
        }
    }
    fn execute(fail_history: bool) {
        let (root, prepared) = fixture();
        let device = device();
        eprintln!("ACTUAL HOOK DEVICE {:?}", device.descriptor);
        let request = TrainRequest {
            project_id: prepared.project_id,
            batch_mode: BatchMode::Fixed,
            config: TrainConfig {
                epochs: 2,
                batch_size: 1,
                ..Default::default()
            },
            device: DevicePolicy::Auto,
            resume: None,
        };
        let cancel = Cancellation::new();
        super::super::trainer::CANCEL_AFTER_FORWARD.with(|f| f.set(true));
        super::super::metrics::READ_ONLY_WRITE.with(|f| f.set(fail_history));
        let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.artifacts/assets/vgg19");
        let result = train(
            &request,
            &prepared,
            &assets,
            &device,
            &TrainingRuntime {
                probe_runner: root.join("never.exe"),
            },
            &Observe,
            &cancel,
        );
        assert!(cancel.is_cancelled());
        let project = super::super::checkpoint::read_project(&root).unwrap();
        assert_eq!(project.checkpoints.len(), 1);
        let loaded = load_checkpoint(&root, &project.checkpoints[0].model_ref, &device).unwrap();
        assert_eq!(loaded.metadata.global_step, 1);
        assert_eq!(loaded.metadata.epoch, 1);
        if fail_history {
            let error = result.unwrap_err();
            assert_eq!(error.stage, "training-metrics-after-commit");
            assert!(
                error
                    .evidence
                    .iter()
                    .any(|e| e.contains("Committed state saved"))
            );
            assert_eq!(
                std::fs::metadata(root.join("metrics.jsonl")).unwrap().len(),
                0
            );
            // The failed writer was dropped; a new writer can acquire its lock.
            SampledMetrics::open(
                &root,
                loaded.metadata.project_id,
                loaded.metadata.run_id,
                loaded.metadata.global_step,
            )
            .unwrap();
        } else {
            let result = result.unwrap();
            assert_eq!(result.stopping_reason, StoppingReason::Cancelled);
            assert_eq!(result.committed_model, project.checkpoints[0].model_ref);
        }
    }
    #[test]
    fn midstep_cancel_saves_completed_update() {
        execute(false);
    }
    #[test]
    fn resume_settings_preserve_adam_and_new_run_identity() {
        let (root, prepared) = fixture();
        let device = device();
        let mut request = TrainRequest {
            project_id: prepared.project_id,
            batch_mode: BatchMode::Fixed,
            config: TrainConfig {
                epochs: 2,
                batch_size: 1,
                ..Default::default()
            },
            device: DevicePolicy::Auto,
            resume: None,
        };
        let mut state = TrainingState::fresh(&request, &prepared, &device).unwrap();
        // State protocol fixture only, not full-objective execution evidence.
        super::super::checkpoint_tests::update(&mut state);
        let before = state.optimizer_bytes().unwrap();
        let run = state.metadata.run_id;
        let reference = publish(&state, &Observe).unwrap();
        request.resume = Some(reference);
        for field in ["seed", "loss", "beta1", "beta2", "epsilon", "weight_decay"] {
            let mut invalid = request.clone();
            match field {
                "seed" => invalid.config.seed += 1,
                "loss" => invalid.config.loss.full += 1.,
                "beta1" => invalid.config.beta1 = 0.8,
                "beta2" => invalid.config.beta2 = 0.99,
                "epsilon" => invalid.config.epsilon = 1e-7,
                _ => invalid.config.weight_decay = 0.1,
            };
            assert!(validate_resume(&state, &invalid).is_err());
        }
        request.config.epochs = 3;
        request.config.lr = 0.0002;
        request.config.save_every = 2;
        request.config.prefetch_capacity = 0;
        request.config.batch_size = 3;
        request.batch_mode = BatchMode::Auto;
        let cancel = Cancellation::new();
        cancel.cancel();
        let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.artifacts/assets/vgg19");
        let runtime = TrainingRuntime {
            probe_runner: root.join("never.exe"),
        };
        let resumed = train(
            &request, &prepared, &assets, &device, &runtime, &Observe, &cancel,
        )
        .unwrap();
        assert_eq!(resumed.run_id, run);
        assert_eq!(resumed.global_step, 1);
        let restored = load_checkpoint(&root, &resumed.committed_model, &device).unwrap();
        assert_eq!(restored.metadata.config, request.config);
        assert_eq!(restored.metadata.batch_mode, BatchMode::Auto);
        super::super::checkpoint_validation::equal_pack(
            &before,
            &restored.optimizer_bytes().unwrap(),
        )
        .unwrap();
        request.resume = None;
        let new = train(
            &request, &prepared, &assets, &device, &runtime, &Observe, &cancel,
        )
        .unwrap();
        assert_ne!(new.run_id, run);
        assert_eq!(new.global_step, 0);
        assert_eq!(new.stopping_reason, StoppingReason::Cancelled);
    }
    #[test]
    fn history_io_failure_saves_update_and_drops_writer() {
        execute(true);
    }
}
fn validate_resume(state: &TrainingState, request: &TrainRequest) -> Result<()> {
    let old = &state.metadata.config;
    let new = &request.config;
    if old.seed != new.seed
        || old.loss != new.loss
        || old.beta1 != new.beta1
        || old.beta2 != new.beta2
        || old.epsilon != new.epsilon
        || old.weight_decay != new.weight_decay
    {
        let mut e = invalid("Resume cannot change seed, loss, or Adam beta/epsilon/decay settings");
        e.actions
            .push("Keep checkpoint settings or start a new run.".into());
        return Err(e);
    }
    if new.epochs < state.metadata.epoch {
        return Err(invalid(
            "Total epochs is below already completed epochs; choose at least the completed total",
        ));
    }
    Ok(())
}
/// Explicit runtime and assets are supplied by the resource/worker layer.
/// Resume accepts total epochs, lr, save/prefetch and batch settings, preserving Adam.
/// Seed/loss/Adam beta/epsilon/decay remain pinned. At an already completed target,
/// operational settings have no effect and the original committed reference is returned.
pub fn train(
    request: &TrainRequest,
    prepared: &PreparedProject,
    vgg_assets: &Path,
    device: &ExecutionDevice,
    runtime: &TrainingRuntime,
    observer: &dyn TaskObserver,
    cancel: &Cancellation,
) -> Result<TrainingResult> {
    super::state::validate_adam_config(&request.config)?;
    if request.project_id != prepared.project_id {
        return Err(invalid("Request project differs from prepared data"));
    }
    if let DevicePolicy::Explicit(d) = &request.device {
        if d.device_id != device.descriptor.id || d.backend != device.descriptor.backend {
            return Err(invalid(
                "Request device differs from selected execution device",
            ));
        }
    }
    let (mut state, data) = match &request.resume {
        Some(reference) => {
            let (state, pinned) =
                load_checkpoint_with_data(&prepared.project_directory, reference, device)?;
            let data = TrainingData::bind(&state, prepared, &pinned)?;
            (state, data)
        }
        None => TrainingState::fresh_with_data(request, prepared, device)?,
    };
    let vgg = VggFeatures::load(vgg_assets, &device.device.clone().autodiff())?;
    if state
        .metadata
        .vgg_identity
        .as_ref()
        .is_some_and(|id| id != vgg.identity())
    {
        return Err(invalid(
            "VGG loss dependency differs from committed binding",
        ));
    }
    if request.resume.is_some() {
        validate_resume(&state, request)?;
    }
    let result = |state: &TrainingState, committed_model, stopping_reason| TrainingResult {
        committed_model,
        run_id: state.metadata.run_id,
        epoch: state.metadata.epoch,
        global_step: state.metadata.global_step,
        stopping_reason,
    };
    if state.metadata.epoch == request.config.epochs {
        return Ok(result(
            &state,
            request
                .resume
                .clone()
                .ok_or_else(|| invalid("Completed state lacks checkpoint"))?,
            StoppingReason::Completed,
        ));
    }
    state.set_learning_rate(request.config.lr)?;
    state.metadata.config = request.config.clone();
    state.metadata.batch_mode = request.batch_mode;
    state.metadata.requested_batch = request.config.batch_size;
    state.metadata.actual_batch = request.config.batch_size;
    // Schema2 permits a selected pristine dependency. Child must use this exact package.
    state.metadata.vgg_identity = Some(vgg.identity().clone());
    if cancel.is_cancelled() {
        let reference = publish(&state, observer)?;
        return Ok(result(&state, reference, StoppingReason::Cancelled));
    }
    if !cancel.is_cancelled() {
        match probe_batch_size_validated(
            &state,
            &data,
            device,
            &ProbeContext {
                runtime,
                vgg_assets,
                cancel,
            },
        ) {
            Ok(selection) => {
                state.metadata.actual_batch = selection.candidate;
                observer.emit(EventPayload::Progress {
                    stage: format!(
                        "batch-selected candidate={} effective={} attempts={:?}",
                        selection.candidate, selection.effective_pairs, selection.attempts
                    ),
                    completed: 0,
                    total: None,
                })?;
            }
            Err(e) if e.code == ErrorCode::Cancelled => {
                let reference = publish(&state, observer)?;
                return Ok(result(&state, reference, StoppingReason::Cancelled));
            }
            Err(e) => return Err(e),
        }
    }
    let mut history = SampledMetrics::open(
        state.project_directory(),
        state.metadata.project_id,
        state.metadata.run_id,
        state.metadata.global_step,
    )?;
    let base = history.elapsed_base_ms();
    let start = Instant::now();
    let mut last_saved = None;
    while state.metadata.epoch < request.config.epochs && !cancel.is_cancelled() {
        let before_epoch = state.metadata.epoch;
        let step = match super::trainer::complete_step_validated(
            &mut state, &data, &vgg, device, cancel,
        ) {
            Ok(step) => step,
            Err(mut e) => {
                if e.code == ErrorCode::Cancelled {
                    break;
                }
                e.actions.push(
                    "Try a smaller Fixed batch if the selected device ran out of memory.".into(),
                );
                return Err(e);
            }
        };
        let sample = history.sample(
            &state,
            &step.losses,
            device,
            base.saturating_add(start.elapsed().as_millis() as u64),
            step.pairs.len() as u32,
        );
        match sample {
            Ok(Some(metric)) => observer.emit(EventPayload::Metrics(metric))?,
            Ok(None) => {}
            Err(mut e) => {
                drop(history); // Never reuse a writer after an append/sync failure.
                match publish(&state, observer) {
                    Ok(reference) => e
                        .evidence
                        .push(format!("Committed state saved as {reference:?}")),
                    Err(save) => e
                        .evidence
                        .push(format!("Checkpoint publication failed: {save}")),
                };
                return Err(e);
            }
        }
        if state.metadata.epoch != before_epoch
            && state.metadata.epoch % request.config.save_every == 0
        {
            last_saved = Some((state.metadata.global_step, publish(&state, observer)?));
        }
        if step.cancelled {
            break;
        }
    }
    let reference = match last_saved {
        Some((step, reference)) if step == state.metadata.global_step => reference,
        _ => publish(&state, observer)?,
    };
    let reason = if state.metadata.epoch >= request.config.epochs {
        StoppingReason::Completed
    } else {
        StoppingReason::Cancelled
    };
    Ok(result(&state, reference, reason))
}
