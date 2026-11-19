use feathertalk_core::{DeviceDescriptor, Precision};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkloadIdentity {
    pub engine_sha256: String,
    pub source_sha256: String,
    pub audio_sha256: String,
    pub checkpoint_sha256: String,
    pub prepared_sha256: String,
    pub assets_sha256: String,
    pub ffmpeg_sha256: String,
    pub ffprobe_sha256: String,
    pub reference_sha256: String,
    pub os: String,
    pub architecture: String,
    pub precision: Precision,
    pub requested_batch: u32,
    pub prefetch_capacity: u32,
    pub sample_loss: bool,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkSample {
    pub prepare_ms: u64,
    pub checkpoint_load_ms: u64,
    pub training_step_ms: u64,
    pub training_pairs: u32,
    pub training_total_loss: Option<f64>,
    pub audio_features_ms: Option<u64>,
    pub render_ms: Option<u64>,
    pub encode_ms: Option<u64>,
    pub generation_ms: u64,
    pub video_frames: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkReport {
    pub schema_version: u32,
    pub workload: WorkloadIdentity,
    pub device: DeviceDescriptor,
    pub validation_passed: bool,
    pub validation_failures: Vec<String>,
    pub device_init_ms: u64,
    pub warmup: u32,
    pub cold_runs: Vec<BenchmarkSample>,
    pub warm_runs: Vec<BenchmarkSample>,
    pub peak_rss_bytes: Option<u64>,
    pub peak_vram_bytes: Option<u64>,
    pub memory_measurement_note: String,
}

impl Default for BenchmarkReport {
    fn default() -> Self {
        Self {
            schema_version: 1,
            workload: WorkloadIdentity {
                engine_sha256: String::new(),
                source_sha256: String::new(),
                audio_sha256: String::new(),
                checkpoint_sha256: String::new(),
                prepared_sha256: String::new(),
                assets_sha256: String::new(),
                ffmpeg_sha256: String::new(),
                ffprobe_sha256: String::new(),
                reference_sha256: String::new(),
                os: String::new(),
                architecture: String::new(),
                precision: Precision::F32,
                requested_batch: 0,
                prefetch_capacity: 0,
                sample_loss: false,
            },
            device: DeviceDescriptor {
                id: String::new(),
                name: String::new(),
                vendor: None,
                driver: None,
                backend: feathertalk_core::BackendKind::Flex,
                memory_bytes: None,
            },
            validation_passed: false,
            validation_failures: vec![],
            device_init_ms: 0,
            warmup: 0,
            cold_runs: vec![],
            warm_runs: vec![],
            peak_rss_bytes: None,
            peak_vram_bytes: None,
            memory_measurement_note: "Per-workload peak memory is unavailable in this runner"
                .into(),
        }
    }
}

impl BenchmarkReport {
    /// A descriptor without a verifiable driver is useful for an immediate
    /// comparison, but it cannot establish that an on-disk result is current.
    pub fn reusable_for(&self, workload: &WorkloadIdentity, device: &DeviceDescriptor) -> bool {
        self.device == *device
            && device
                .driver
                .as_ref()
                .is_some_and(|driver| !driver.trim().is_empty())
            && self.workload == *workload
            && self.warm_score().is_some()
    }

    pub(super) fn warm_score(&self) -> Option<u64> {
        let batch = self.workload.requested_batch;
        if self.schema_version != 1
            || !self.validation_passed
            || !self.validation_failures.is_empty()
            || self.cold_runs.len() != 1
            || self.warm_runs.len() < 3
            || batch == 0
            || self.device.id.is_empty()
            || self.device.name.is_empty()
        {
            return None;
        }
        let frames = self.cold_runs[0].video_frames;
        if frames == 0
            || self.cold_runs.iter().chain(&self.warm_runs).any(|sample| {
                sample.training_pairs != batch
                    || sample.video_frames != frames
                    || sample.training_step_ms == 0
                    || sample.generation_ms == 0
            })
        {
            return None;
        }
        let mut totals: Vec<_> = self
            .warm_runs
            .iter()
            .map(|run| run.training_step_ms.saturating_add(run.generation_ms))
            .collect();
        totals.sort_unstable();
        Some(totals[totals.len() / 2])
    }
}
