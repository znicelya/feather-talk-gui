//! Disposable full-objective candidates. No checkpoint or history publication.
use super::{checkpoint::MaterializedSnapshot, state::TrainingState};
use crate::{ExecutionDevice, data::prepared::PreparedProject};
use feathertalk_core::*;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainingRuntime {
    pub probe_runner: PathBuf,
}
pub struct ProbeContext<'a> {
    pub runtime: &'a TrainingRuntime,
    pub vgg_assets: &'a Path,
    pub cancel: &'a Cancellation,
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProbeSelection {
    pub candidate: u32,
    pub effective_pairs: u32,
    pub attempts: Vec<u32>,
}

pub fn probe_batch_size(
    state: &TrainingState,
    prepared: &PreparedProject,
    device: &ExecutionDevice,
    context: &ProbeContext<'_>,
) -> Result<ProbeSelection> {
    state.validate_prepared(prepared)?;
    let target = state.metadata.requested_batch;
    if state.metadata.batch_mode == BatchMode::Fixed {
        return Ok(ProbeSelection {
            candidate: target,
            effective_pairs: target.min(prepared.usable_frames.saturating_sub(1) as u32),
            attempts: vec![],
        });
    }
    if state.metadata.epoch >= state.metadata.config.epochs {
        return Err(super::state::invalid(
            "Cannot probe an already completed run",
        ));
    }
    let snapshot = MaterializedSnapshot::capture(state)?;
    let mut candidate = target;
    let mut attempts = vec![];
    loop {
        crate::media::ffmpeg::check(context.cancel)?;
        attempts.push(candidate);
        match super::probe_transport::candidate(
            &snapshot,
            state.project_directory(),
            prepared,
            device,
            context,
            candidate,
        ) {
            Ok(effective_pairs) => {
                return Ok(ProbeSelection {
                    candidate,
                    effective_pairs,
                    attempts,
                });
            }
            Err(e) if e.code == ErrorCode::OutOfMemory && candidate > 1 => {
                candidate = (candidate / 2).max(1)
            }
            Err(e) => return Err(e),
        }
    }
}
