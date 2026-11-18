//! A complete optimizer update is the only publication boundary.
use super::{
    checkpoint::MaterializedSnapshot,
    losses::{LossParts, compute_loss},
    state::{TrainingState, invalid, with_seed},
};
use crate::{
    ExecutionDevice,
    data::{
        prepared::PreparedProject,
        sampler::{HostTensor, PairSampler},
    },
    models::vgg::VggFeatures,
};
use burn::{
    module::{Module, ModuleVisitor, Param},
    optim::GradientsParams,
    tensor::{Bool, Tensor, TensorData},
};
use feathertalk_core::*;

pub struct StepOutcome {
    pub losses: LossParts,
    pub pairs: Vec<[usize; 2]>,
    pub references: Vec<usize>,
    pub seed: u64,
    pub cancelled: bool,
}
fn tensor<const D: usize>(h: HostTensor<D>, device: &burn::tensor::Device) -> Tensor<D> {
    Tensor::from_data(TensorData::new(h.values, h.dims), device)
}
/// Does not implement Auto probing. The caller must first select actual_batch.
/// Cancellation before work leaves state untouched; cancellation during work is
/// reported after a complete commit. Device aborts are process-isolation concerns.
pub fn complete_step(
    state: &mut TrainingState,
    prepared: &PreparedProject,
    vgg: &VggFeatures,
    device: &ExecutionDevice,
    cancel: &Cancellation,
) -> Result<StepOutcome> {
    crate::media::ffmpeg::check(cancel)?;
    state.validate_prepared(prepared)?;
    let d = device.device.clone().autodiff();
    if state.model.devices() != vec![d.clone()] || vgg.devices() != vec![d.clone()] {
        return Err(invalid(
            "Model and frozen VGG must use the selected autodiff device",
        ));
    }
    let next_step = state
        .metadata
        .global_step
        .checked_add(1)
        .ok_or_else(|| invalid("Step overflow"))?;
    if state.metadata.epoch >= state.metadata.config.epochs {
        return Err(invalid("Training epochs already complete"));
    }
    // An independent candidate owns BN, parameters and Adam. Errors never expose
    // partial mutations in the caller's still-committed state. No nested RNG lock.
    let backup = MaterializedSnapshot::capture(state)?;
    let mut candidate = backup.restore(state.project_directory(), device)?;
    drop(backup);
    let mut sampler = PairSampler::restore(prepared, state.metadata.sampler.clone())?;
    if sampler.snapshot().cursor == sampler.snapshot().permutation.len() {
        sampler.advance_epoch()?;
    }
    let batch_size = state.metadata.actual_batch as usize;
    let batch = if state.metadata.config.prefetch_capacity == 0 {
        sampler.next_batch(batch_size)?
    } else {
        let mut prefetched = sampler.prefetch(
            batch_size,
            state.metadata.config.prefetch_capacity as usize,
            cancel.clone(),
        )?;
        let batch = prefetched.next_batch()?;
        sampler = PairSampler::restore(prepared, prefetched.snapshot())?;
        batch
    }
    .ok_or_else(|| invalid("Prepared dataset contains no training pair"))?;
    if sampler.snapshot().cursor == sampler.snapshot().permutation.len() {
        sampler.advance_epoch()?;
    }
    let staged = sampler.snapshot();
    let epoch = u32::try_from(staged.epoch).map_err(invalid)?;
    crate::media::ffmpeg::check(cancel)?;
    let seed = state
        .metadata
        .seed_policy
        .step_seed(state.metadata.global_step);
    let losses = with_seed(&device.device, seed, || {
        let b = batch.pairs.len();
        let prediction = candidate
            .model
            .forward(
                tensor(batch.image.clone(), &d).reshape([b * 2, 6, 144, 144]),
                tensor(batch.audio.clone(), &d).reshape([b * 2, 40, 1024]),
            )
            .reshape([b, 2, 3, 144, 144]);
        #[cfg(test)]
        if FAIL_AFTER_FORWARD.with(|flag| flag.replace(false)) {
            assert!(
                candidate
                    .model
                    .inc
                    .expand_bn
                    .state
                    .get()
                    .count
                    .val()
                    .into_scalar::<i64>()
                    > 0
            );
            return Err(invalid(
                "Injected ordinary error after forward and BN mutation",
            ));
        }
        let output = compute_loss(
            prediction,
            tensor(batch.target.clone(), &d),
            tensor(batch.mask.clone(), &d),
            vgg,
            &candidate.metadata.config.loss,
        );
        let finite_loss = output.total.clone().is_finite().all();
        let gradients = GradientsParams::from_grads(output.total.backward(), &candidate.model);
        struct Check<'a> {
            gradients: &'a GradientsParams,
            finite: Tensor<1, Bool>,
            count: usize,
            missing: bool,
        }
        impl ModuleVisitor for Check<'_> {
            fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
                if p.is_require_grad() {
                    self.count += 1;
                    match self.gradients.get::<D>(p.id) {
                        Some(g) => self.finite = self.finite.clone().bool_and(g.is_finite().all()),
                        None => self.missing = true,
                    }
                }
            }
        }
        let mut check = Check {
            gradients: &gradients,
            finite: finite_loss,
            count: 0,
            missing: false,
        };
        candidate.model.visit(&mut check);
        if check.missing || check.count != 252 || !check.finite.into_scalar::<bool>() {
            return Err(invalid(
                "Full objective requires all 252 finite gradients and finite loss",
            ));
        }
        candidate.apply_gradients(gradients)?;
        Ok(output.parts)
    })?;
    candidate.metadata.global_step = next_step;
    candidate.metadata.epoch = epoch;
    candidate.metadata.sampler = staged;
    // Validate finite updated model and the complete 504-moment/time inventory
    // before publication; this is deliberately correctness-first host overhead.
    MaterializedSnapshot::capture(&candidate)?;
    *state = candidate;
    Ok(StepOutcome {
        losses,
        pairs: batch.pairs,
        references: batch.references,
        seed,
        cancelled: cancel.is_cancelled(),
    })
}

#[cfg(test)]
thread_local! { static FAIL_AFTER_FORWARD: std::cell::Cell<bool> = const { std::cell::Cell::new(false) }; }
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn postforward_error_preserves_complete_committed_state() {
        let (root, prepared) = super::super::checkpoint_tests::fixture();
        let device = super::super::checkpoint_tests::device();
        let request = TrainRequest {
            project_id: prepared.project_id,
            batch_mode: BatchMode::Fixed,
            config: TrainConfig {
                batch_size: 1,
                ..Default::default()
            },
            device: DevicePolicy::Auto,
            resume: None,
        };
        let mut state = TrainingState::fresh(&request, &prepared, &device).unwrap();
        let before_model = super::super::checkpoint::model_bytes(&state.model).unwrap();
        let before_adam = state.optimizer_bytes().unwrap();
        let before = state.metadata.clone();
        let vgg = VggFeatures::load(
            &std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.artifacts/assets/vgg19"),
            &device.device.clone().autodiff(),
        )
        .unwrap();
        FAIL_AFTER_FORWARD.with(|flag| flag.set(true));
        let err = complete_step(&mut state, &prepared, &vgg, &device, &Cancellation::new())
            .err()
            .unwrap();
        assert!(err.message.contains("after forward and BN mutation"));
        assert_eq!(before, state.metadata);
        super::super::checkpoint_validation::equal_pack(
            &before_model,
            &super::super::checkpoint::model_bytes(&state.model).unwrap(),
        )
        .unwrap();
        super::super::checkpoint_validation::equal_pack(
            &before_adam,
            &state.optimizer_bytes().unwrap(),
        )
        .unwrap();
        MaterializedSnapshot::capture(&state)
            .unwrap()
            .restore(&root, &device)
            .unwrap();
    }
}
