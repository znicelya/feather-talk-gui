//! Frozen torchvision VGG19 features[0..=14], consuming unnormalized BGR 0..1.
use crate::verified_assets::{VerifiedAssets, error};
use burn::{
    module::{Module, Param},
    tensor::{
        Device, Tensor, TensorData,
        activation::relu,
        module::{conv2d, max_pool2d},
        ops::{ConvOptions, MaxPoolOptions},
    },
};
use feathertalk_core::Result;
use safetensors::{Dtype, SafeTensors};
use std::{collections::BTreeMap, path::Path};
#[derive(Module, Debug)]
struct FrozenConv {
    weight: Param<Tensor<4>>,
    bias: Param<Tensor<1>>,
}
#[derive(Module, Debug)]
pub struct VggFeatures {
    layers: Vec<FrozenConv>,
}
impl VggFeatures {
    pub fn load(assets: &Path, device: &Device) -> Result<Self> {
        let a = VerifiedAssets::load(assets)?;
        let id = &a.manifest.identity;
        if id.kind != "vgg19"
            || id.architecture_version != "vgg19-source-v1"
            || id.source_sha256
                != "dcbb9e9dad569fff7a846263a77324fc34978fea2bfb039c012d710e1776ae44"
        {
            return Err(error("VGG identity mismatch"));
        }
        let c: serde_json::Value = serde_json::from_slice(
            a.files
                .get("config.json")
                .ok_or_else(|| error("Missing VGG config"))?,
        )
        .map_err(error)?;
        if c["source_sha256"] != id.source_sha256
            || c["architecture_version"] != id.architecture_version
            || c["configuration"]
                != serde_json::json!({"architecture":"torchvision.vgg19.features","color":"BGR","inclusive_last_layer":14,"normalization":"none","range":[0,1]})
        {
            return Err(error("VGG configuration mismatch"));
        }
        let safe = SafeTensors::deserialize(
            a.files
                .get("model.safetensors")
                .ok_or_else(|| error("Missing VGG weights"))?,
        )
        .map_err(error)?;
        let mut tensors: BTreeMap<_, _> = safe.tensors().into_iter().collect();
        if tensors.len() != 14 {
            return Err(error("VGG requires exactly fourteen tensors"));
        }
        fn take<const D: usize>(
            t: &mut BTreeMap<String, safetensors::tensor::TensorView<'_>>,
            key: String,
            shape: [usize; D],
            d: &Device,
        ) -> Result<Tensor<D>> {
            let v = t
                .remove(&key)
                .ok_or_else(|| error(format!("Missing VGG {key}")))?;
            if v.dtype() != Dtype::F32 || v.shape() != shape {
                return Err(error(format!("Invalid VGG tensor {key}")));
            }
            let data: Vec<f32> = v
                .data()
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            if data.iter().any(|v| !v.is_finite()) {
                return Err(error(format!("Nonfinite VGG {key}")));
            }
            Ok(Tensor::from_data(TensorData::new(data, shape), d).set_require_grad(false))
        }
        let mut layers = Vec::new();
        for (key, i, o) in [
            (0, 3, 64),
            (2, 64, 64),
            (5, 64, 128),
            (7, 128, 128),
            (10, 128, 256),
            (12, 256, 256),
            (14, 256, 256),
        ] {
            layers.push(FrozenConv {
                weight: Param::from_tensor(take(
                    &mut tensors,
                    format!("{key}.weight"),
                    [o, i, 3, 3],
                    device,
                )?)
                .set_require_grad(false),
                bias: Param::from_tensor(take(&mut tensors, format!("{key}.bias"), [o], device)?)
                    .set_require_grad(false),
            });
        }
        if !tensors.is_empty() {
            return Err(error("Unexpected VGG tensor keys"));
        }
        Ok(Self { layers })
    }
    pub fn forward(&self, images: Tensor<4>) -> Tensor<4> {
        self.forward_observed(images, |_, _| {})
    }
    pub fn forward_observed(
        &self,
        mut x: Tensor<4>,
        mut tap: impl FnMut(usize, Tensor<4>),
    ) -> Tensor<4> {
        for (j, (layer, index)) in self.layers.iter().zip([0, 2, 5, 7, 10, 12, 14]).enumerate() {
            if j == 2 || j == 4 {
                x = max_pool2d(x, MaxPoolOptions::new([2, 2]).with_stride([2, 2]));
                tap(index - 1, x.clone());
            }
            x = conv2d(
                x,
                layer.weight.val(),
                Some(layer.bias.val()),
                ConvOptions::new([1, 1], [1, 1], [1, 1], 1),
            );
            tap(index, x.clone());
            if j != 6 {
                x = relu(x);
            }
        }
        x
    }
}
