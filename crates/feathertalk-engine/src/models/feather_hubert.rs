//! Frozen, configuration-driven current FeatherHuBERT inference.
use crate::{
    ops::batch_norm::group_norm,
    verified_assets::{VerifiedAssets, error},
};
use burn::tensor::{
    Device, Tensor, TensorData, activation::gelu, module::conv1d, ops::ConvOptions,
};
use feathertalk_core::*;
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use std::{collections::BTreeMap, path::Path};

const SOURCE: &str = "3b147e66502132bac3ff76b4730c2a83c341c75251d22a28182a5a1b689cfc46";
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HubertConfiguration {
    pub channels: usize,
    pub expansion: usize,
    pub num_blocks: usize,
    pub output_dim: usize,
    pub dropout: f64,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Envelope {
    architecture_version: String,
    source_sha256: String,
    configuration: HubertConfiguration,
}
struct Norm {
    weight: Tensor<1>,
    bias: Tensor<1>,
    groups: usize,
    wide_statistics: bool,
}
impl Norm {
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        if self.wide_statistics {
            // Long CPU reductions otherwise amplify FP32 accumulation drift.
            // This remains Burn math on the host device, without changing the
            // source normalization equation or the GPU execution path.
            use burn::tensor::FloatDType;
            return group_norm(
                x.cast(FloatDType::F64),
                self.weight.clone().cast(FloatDType::F64),
                self.bias.clone().cast(FloatDType::F64),
                self.groups,
                1e-5,
            )
            .cast(FloatDType::F32);
        }
        group_norm(x, self.weight.clone(), self.bias.clone(), self.groups, 1e-5)
    }
}
struct Frontend {
    weight: Tensor<3>,
    norm: Norm,
    stride: usize,
}
struct Block {
    norm: Norm,
    expand: Tensor<3>,
    depthwise: Tensor<3>,
    project: Tensor<3>,
    dilation: usize,
}
pub struct FeatherHubert {
    identity: ModelIdentity,
    config: HubertConfiguration,
    device: Device,
    frontend: Vec<Frontend>,
    blocks: Vec<Block>,
    final_norm: Norm,
    projection: Tensor<3>,
    bias: Tensor<1>,
}
fn groups(channels: usize) -> usize {
    [32, 16, 8, 4, 2, 1]
        .into_iter()
        .find(|n| channels % n == 0)
        .unwrap()
}
struct Parameters {
    values: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
}
impl Parameters {
    fn take<const D: usize>(
        &mut self,
        key: &str,
        shape: [usize; D],
        device: &Device,
    ) -> Result<Tensor<D>> {
        let (actual, values) = self
            .values
            .remove(key)
            .ok_or_else(|| error(format!("Missing parameter {key}")))?;
        if actual != shape {
            return Err(error(format!(
                "Parameter {key}: expected {shape:?}, got {actual:?}"
            )));
        }
        Ok(Tensor::from_data(TensorData::new(values, shape), device))
    }
    fn norm(&mut self, key: &str, channels: usize, device: &Device) -> Result<Norm> {
        Ok(Norm {
            weight: self.take(&format!("{key}.weight"), [channels], device)?,
            bias: self.take(&format!("{key}.bias"), [channels], device)?,
            groups: groups(channels),
            wide_statistics: {
                #[cfg(feature = "flex")]
                {
                    *device == Device::flex()
                }
                #[cfg(not(feature = "flex"))]
                {
                    false
                }
            },
        })
    }
}
impl FeatherHubert {
    pub fn load(assets: &Path, device: &Device) -> Result<Self> {
        let assets = VerifiedAssets::load(assets)?;
        let identity = assets.manifest.identity;
        if identity.kind != "feather_hubert"
            || identity.architecture_version != "feather_hubert-source-v1"
            || identity.source_sha256 != SOURCE
        {
            return Err(error("Expected current FeatherHuBERT identity"));
        }
        let config: Envelope = serde_json::from_slice(
            assets
                .files
                .get("config.json")
                .ok_or_else(|| error("Missing config"))?,
        )
        .map_err(error)?;
        if config.architecture_version != identity.architecture_version
            || config.source_sha256 != identity.source_sha256
        {
            return Err(error("Config identity mismatch"));
        }
        let config = config.configuration;
        if config.channels == 0
            || config.expansion == 0
            || config.num_blocks == 0
            || config.output_dim != 1024
            || !config.dropout.is_finite()
            || !(0.0..1.0).contains(&config.dropout)
            || config.channels.checked_mul(config.expansion).is_none()
        {
            return Err(error("Invalid FeatherHuBERT configuration"));
        }
        let safe = SafeTensors::deserialize(
            assets
                .files
                .get("model.safetensors")
                .ok_or_else(|| error("Missing weights"))?,
        )
        .map_err(error)?;
        let mut values = BTreeMap::new();
        for (key, t) in safe.tensors() {
            if t.dtype() != Dtype::F32 {
                return Err(error(format!("{key}: expected F32")));
            }
            let data: Vec<_> = t
                .data()
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            if data.iter().any(|v| !v.is_finite()) {
                return Err(error(format!("{key}: nonfinite weights")));
            }
            values.insert(key, (t.shape().to_vec(), data));
        }
        // Validate every key and shape on the host before any tensor allocation.
        let mut expected = BTreeMap::new();
        let widths = [
            1,
            64,
            128,
            256,
            384,
            config.channels,
            config.channels,
            config.channels,
        ];
        let kernels = [10, 3, 3, 3, 3, 2, 2];
        let mut norm_shapes = |key: String, c: usize| {
            expected.insert(format!("{key}.weight"), vec![c]);
            expected.insert(format!("{key}.bias"), vec![c]);
        };
        for i in 0..7 {
            norm_shapes(format!("frontend.layers.{i}.norm"), widths[i + 1]);
        }
        for i in 0..config.num_blocks {
            norm_shapes(format!("encoder.{i}.norm"), config.channels);
        }
        norm_shapes("final_norm".into(), config.channels);
        let hidden = config.channels * config.expansion;
        for i in 0..7 {
            expected.insert(
                format!("frontend.layers.{i}.conv.weight"),
                vec![widths[i + 1], widths[i], kernels[i]],
            );
        }
        for i in 0..config.num_blocks {
            expected.insert(
                format!("encoder.{i}.pw_expand.weight"),
                vec![hidden, config.channels, 1],
            );
            expected.insert(format!("encoder.{i}.dw_conv.weight"), vec![hidden, 1, 5]);
            expected.insert(
                format!("encoder.{i}.pw_project.weight"),
                vec![config.channels, hidden, 1],
            );
        }
        expected.insert(
            "proj.weight".into(),
            vec![config.output_dim, config.channels, 1],
        );
        expected.insert("proj.bias".into(), vec![config.output_dim]);
        if values.len() != expected.len() {
            return Err(error("Unexpected parameter count"));
        }
        for (key, shape) in expected {
            if values.get(&key).map(|v| &v.0) != Some(&shape) {
                return Err(error(format!(
                    "Missing or wrong-shaped parameter {key}: expected {shape:?}"
                )));
            }
        }
        let mut p = Parameters { values };
        // All weights are frozen; dropout is deliberately absent in evaluation.
        let device = device.clone().inner();
        let mut frontend = Vec::new();
        for i in 0..7 {
            frontend.push(Frontend {
                weight: p.take(
                    &format!("frontend.layers.{i}.conv.weight"),
                    [widths[i + 1], widths[i], kernels[i]],
                    &device,
                )?,
                norm: p.norm(&format!("frontend.layers.{i}.norm"), widths[i + 1], &device)?,
                stride: if i == 0 { 5 } else { 2 },
            });
        }
        let mut blocks = Vec::new();
        for i in 0..config.num_blocks {
            blocks.push(Block {
                norm: p.norm(&format!("encoder.{i}.norm"), config.channels, &device)?,
                expand: p.take(
                    &format!("encoder.{i}.pw_expand.weight"),
                    [hidden, config.channels, 1],
                    &device,
                )?,
                depthwise: p.take(
                    &format!("encoder.{i}.dw_conv.weight"),
                    [hidden, 1, 5],
                    &device,
                )?,
                project: p.take(
                    &format!("encoder.{i}.pw_project.weight"),
                    [config.channels, hidden, 1],
                    &device,
                )?,
                dilation: 1 << (i % 4),
            });
        }
        let final_norm = p.norm("final_norm", config.channels, &device)?;
        let projection = p.take(
            "proj.weight",
            [config.output_dim, config.channels, 1],
            &device,
        )?;
        let bias = p.take("proj.bias", [config.output_dim], &device)?;
        assert!(p.values.is_empty());
        Ok(Self {
            identity,
            config,
            device,
            frontend,
            blocks,
            final_norm,
            projection,
            bias,
        })
    }
    pub fn identity(&self) -> &ModelIdentity {
        &self.identity
    }
    pub fn configuration(&self) -> &HubertConfiguration {
        &self.config
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn forward(&self, waveform: Tensor<2>) -> Tensor<3> {
        self.forward_observed(waveform, |_, _| {})
    }
    /// Diagnostic taps are emitted one at a time; ordinary inference retains no taps.
    pub fn forward_observed(
        &self,
        waveform: Tensor<2>,
        mut observe: impl FnMut(&str, Tensor<3>),
    ) -> Tensor<3> {
        let [b, n] = waveform.dims();
        assert!(
            n >= 400,
            "FeatherHuBERT requires at least 400 samples at 16 kHz"
        );
        let expected = (n - 80) / 320;
        let mut x = waveform.inner().reshape([b, 1, n]);
        for (i, layer) in self.frontend.iter().enumerate() {
            x = conv1d(
                x,
                layer.weight.clone(),
                None,
                ConvOptions::new([layer.stride], [0], [1], 1),
            );
            observe(&format!("frontend.layers.{i}.conv"), x.clone());
            x = layer.norm.forward(x);
            observe(&format!("frontend.layers.{i}.norm"), x.clone());
            x = gelu(x);
            observe(&format!("frontend.layers.{i}"), x.clone());
        }
        observe("frontend", x.clone());
        for (i, block) in self.blocks.iter().enumerate() {
            let y = conv1d(
                block.norm.forward(x.clone()),
                block.expand.clone(),
                None,
                ConvOptions::new([1], [0], [1], 1),
            );
            let y = gelu(conv1d(
                y,
                block.depthwise.clone(),
                None,
                ConvOptions::new(
                    [1],
                    [2 * block.dilation],
                    [block.dilation],
                    self.config.channels * self.config.expansion,
                ),
            ));
            x = x + conv1d(
                y,
                block.project.clone(),
                None,
                ConvOptions::new([1], [0], [1], 1),
            );
            observe(&format!("encoder.{i}"), x.clone());
        }
        observe("encoder", x.clone());
        x = self.final_norm.forward(x);
        observe("final_norm", x.clone());
        x = conv1d(
            gelu(x),
            self.projection.clone(),
            Some(self.bias.clone()),
            ConvOptions::new([1], [0], [1], 1),
        );
        observe("proj", x.clone());
        let t = x.dims()[2];
        if t > expected {
            x = x.slice([0..b, 0..self.config.output_dim, 0..expected]);
        }
        if t < expected {
            x = Tensor::cat(
                vec![
                    x,
                    Tensor::zeros([b, self.config.output_dim, expected - t], &self.device),
                ],
                2,
            );
        }
        x.swap_dims(1, 2)
    }
}
