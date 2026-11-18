use crate::{AppError, ErrorCode, Result};
use serde::{Deserialize, Serialize};
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LossConfig {
    pub full: f64,
    pub mouth: f64,
    pub temporal: f64,
    pub temporal_mouth: f64,
    pub perceptual: f64,
    pub mouth_gradient: f64,
    pub temporal_kernel: u32,
    pub temporal_stride: u32,
}
impl Default for LossConfig {
    fn default() -> Self {
        Self {
            full: 1.,
            mouth: 4.,
            temporal: 0.5,
            temporal_mouth: 4.,
            perceptual: 0.01,
            mouth_gradient: 0.5,
            temporal_kernel: 5,
            temporal_stride: 1,
        }
    }
}
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TrainConfig {
    /// Stable application-owned random seed; independent sampler and step domains.
    #[serde(default = "default_training_seed")]
    pub seed: u64,
    /// Queued batches, 0 disables prefetch; bounded to 32.
    #[serde(default = "default_prefetch_capacity")]
    pub prefetch_capacity: u32,
    /// Total epochs, including epochs already completed by a resumed run.
    pub epochs: u32,
    /// Target batch in Auto, exact requested batch in Fixed.
    /// Metrics separately report the batch actually executed.
    pub batch_size: u32,
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub epsilon: f64,
    pub weight_decay: f64,
    pub save_every: u32,
    pub loss: LossConfig,
}
fn default_training_seed() -> u64 {
    42
}
fn default_prefetch_capacity() -> u32 {
    2
}
impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            seed: default_training_seed(),
            prefetch_capacity: default_prefetch_capacity(),
            epochs: 60,
            batch_size: 8,
            lr: 0.001,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            weight_decay: 0.,
            save_every: 5,
            loss: LossConfig::default(),
        }
    }
}
impl TrainConfig {
    pub fn validate(&self) -> Result<()> {
        let invalid = |field: &str| {
            let mut e = AppError::new(
                ErrorCode::InvalidConfig,
                "configuration",
                format!("Invalid {field}"),
            );
            e.actions.push(format!("Choose a valid value for {field}."));
            e
        };
        for (name, value) in [
            ("epochs", self.epochs),
            ("batch_size", self.batch_size),
            ("save_every", self.save_every),
            ("temporal_stride", self.loss.temporal_stride),
        ] {
            if value == 0 {
                return Err(invalid(name));
            }
        }
        if self.batch_size > 1024 || self.prefetch_capacity > 32 {
            return Err(invalid("batch_size <= 1024 and prefetch_capacity <= 32"));
        }
        if self.loss.temporal_kernel == 0 || self.loss.temporal_kernel % 2 == 0 {
            return Err(invalid("temporal_kernel (must be positive and odd)"));
        }
        for (name, value) in [("lr", self.lr), ("epsilon", self.epsilon)] {
            if !value.is_finite() || value <= 0. {
                return Err(invalid(name));
            }
        }
        for (name, value) in [("beta1", self.beta1), ("beta2", self.beta2)] {
            if !value.is_finite() || !(0. ..1.).contains(&value) {
                return Err(invalid(name));
            }
        }
        for (name, value) in [
            ("weight_decay", self.weight_decay),
            ("full", self.loss.full),
            ("mouth", self.loss.mouth),
            ("temporal", self.loss.temporal),
            ("temporal_mouth", self.loss.temporal_mouth),
            ("perceptual", self.loss.perceptual),
            ("mouth_gradient", self.loss.mouth_gradient),
        ] {
            if !value.is_finite() || value < 0. {
                return Err(invalid(name));
            }
        }
        Ok(())
    }
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum BackendKind {
    Flex,
    Cpu,
    Vulkan,
    Cuda,
    Metal,
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceRequest {
    pub backend: BackendKind,
    pub device_id: String,
}
#[derive(Debug, Default, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(
    tag = "policy",
    content = "request",
    rename_all = "snake_case",
    deny_unknown_fields
)]
pub enum DevicePolicy {
    #[default]
    Auto,
    Explicit(DeviceRequest),
}
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeviceDescriptor {
    pub id: String,
    pub name: String,
    pub vendor: Option<String>,
    pub driver: Option<String>,
    pub backend: BackendKind,
    pub memory_bytes: Option<u64>,
}
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    F32,
    F16,
    Bf16,
}
