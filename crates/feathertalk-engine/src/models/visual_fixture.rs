//! Developer-only strict import of the canonical synthetic initial state.
use super::feathertalk::FeatherTalk;
use crate::reference::{Reference, error};
use burn::{
    module::{Module, ModuleMapper, ModuleVisitor, Param},
    tensor::{Device, Int, Tensor, TensorData},
};
use feathertalk_core::Result;
use std::{collections::BTreeMap, path::Path};
pub fn source_key(path: &str) -> String {
    let mut key = path.to_owned();
    if key.starts_with("inc.") {
        key = key.replacen("inc.", "inc.inconv.", 1);
    }
    for root in ["down1", "down2", "down3", "down4"] {
        key = key.replace(
            &format!("{root}.blocks."),
            &format!("{root}.maxpool_conv.double_conv."),
        );
    }
    for root in ["up1", "up2", "up3", "up4"] {
        key = key.replace(
            &format!("{root}.blocks."),
            &format!("{root}.conv.double_conv."),
        );
    }
    key = key.replace("fuse_conv.blocks.", "fuse_conv.double_conv.");
    for (from, to) in [
        ("expand", "0"),
        ("expand_bn", "1"),
        ("depthwise", "3"),
        ("depth_bn", "4"),
        ("project", "6"),
        ("project_bn", "7"),
    ] {
        if !key.starts_with("audio_") {
            key = key.replace(&format!(".{from}."), &format!(".conv.{to}."));
        }
    }
    key = key
        .replace(".state.mean", ".running_mean")
        .replace(".state.variance", ".running_var")
        .replace(".state.count", ".num_batches_tracked");
    if key.starts_with("outc.") {
        key = key.replacen("outc.", "outc.conv.", 1);
    }
    key
}
#[derive(Default)]
struct Inventory {
    path: Vec<String>,
    items: BTreeMap<String, (Vec<usize>, bool)>,
}
impl ModuleVisitor for Inventory {
    fn enter_module(&mut self, n: &str, _: &str) {
        self.path.push(n.into());
    }
    fn exit_module(&mut self, _: &str, _: &str) {
        self.path.pop();
    }
    fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
        self.items
            .insert(source_key(&self.path.join(".")), (p.dims().to_vec(), false));
    }
    fn visit_int<const D: usize>(&mut self, p: &Param<Tensor<D, Int>>) {
        self.items
            .insert(source_key(&self.path.join(".")), (p.dims().to_vec(), true));
    }
}
struct Loader {
    path: Vec<String>,
    values: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
}
impl ModuleMapper for Loader {
    fn enter_module(&mut self, n: &str, _: &str) {
        self.path.push(n.into());
    }
    fn exit_module(&mut self, _: &str, _: &str) {
        self.path.pop();
    }
    fn map_float<const D: usize>(&mut self, p: Param<Tensor<D>>) -> Param<Tensor<D>> {
        let key = source_key(&self.path.join("."));
        let (_, data) = self.values.remove(&key).unwrap();
        let value = Tensor::from_data(TensorData::new(data, p.dims()), &p.device());
        Param::initialized(p.id, value).set_require_grad(p.is_require_grad())
    }
    fn map_int<const D: usize>(&mut self, p: Param<Tensor<D, Int>>) -> Param<Tensor<D, Int>> {
        let key = source_key(&self.path.join("."));
        let (_, data) = self.values.remove(&key).unwrap();
        Param::initialized(
            p.id,
            Tensor::<D, Int>::from_data(
                TensorData::new(
                    data.into_iter().map(|n| n as i64).collect::<Vec<_>>(),
                    p.dims(),
                ),
                &p.device(),
            )
            .cast(burn::tensor::IntDType::I64),
        )
    }
}
pub fn load_initial(root: &Path, device: &Device) -> Result<FeatherTalk> {
    let r = Reference::load(root, "character_initial_state")?;
    let bytes = std::fs::read(root.join(&r.case.file)).map_err(error)?;
    let tensors = safetensors::SafeTensors::deserialize(&bytes).map_err(error)?;
    for (key, t) in tensors.tensors() {
        let expected = if key.ends_with("num_batches_tracked") {
            safetensors::Dtype::I64
        } else {
            safetensors::Dtype::F32
        };
        if t.dtype() != expected {
            return Err(error(format!("Invalid dtype for {key}")));
        }
    }
    if r.case.configuration["architecture"] != "FeatherTalk.Model" || r.values.len() != 447 {
        return Err(error("Wrong visual fixture identity/inventory"));
    }
    let m = FeatherTalk::new(device);
    let mut inv = Inventory::default();
    m.visit(&mut inv);
    if inv.items.len() != 447 {
        return Err(error(format!(
            "Module inventory has {} entries: {:?}",
            inv.items.len(),
            inv.items.keys()
        )));
    }
    for (key, (shape, integer)) in inv.items {
        let (actual, data) = r
            .values
            .get(&key)
            .ok_or_else(|| error(format!("Missing {key}")))?;
        let scalar = key.ends_with("residual_gate") || integer;
        if (if scalar {
            actual.is_empty() && shape == [1]
        } else {
            actual == &shape
        }) == false
            || data.iter().any(|n| !n.is_finite())
            || (integer && data != &[0.])
        {
            return Err(error(format!("Invalid shape/value for {key}")));
        }
    }
    let mut loader = Loader {
        path: vec![],
        values: r.values,
    };
    let m = m.map(&mut loader);
    assert!(loader.values.is_empty());
    Ok(m)
}
pub fn load_attention(
    root: &Path,
    device: &Device,
) -> Result<super::attention::SpatialAudioAttention> {
    let r = Reference::load(root, "spatial_audio_attention")?;
    let b = super::visual_parameters::Builder {
        device: device.clone(),
    };
    let model = super::attention::SpatialAudioAttention::new(&b, 8, 12, 8, 2, 5, 2);
    let values: BTreeMap<_, _> = r
        .values
        .into_iter()
        .filter_map(|(k, v)| k.strip_prefix("state.").map(|key| (key.to_owned(), v)))
        .collect();
    validate_values(&model, &values, 15)?;
    let mut loader = Loader {
        path: vec![],
        values,
    };
    let model = model.map(&mut loader);
    if !loader.values.is_empty() {
        return Err(error("Unexpected attention fixture parameters"));
    }
    Ok(model)
}

fn validate_values<M: Module>(
    model: &M,
    values: &BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    count: usize,
) -> Result<()> {
    let mut inventory = Inventory::default();
    model.visit(&mut inventory);
    if inventory.items.len() != count || values.len() != count {
        return Err(error("Fixture inventory mismatch"));
    }
    for (key, (shape, integer)) in inventory.items {
        let (actual, data) = values
            .get(&key)
            .ok_or_else(|| error(format!("Missing {key}")))?;
        let scalar = integer || key.ends_with("residual_gate");
        if !(if scalar {
            actual.is_empty() && shape == [1]
        } else {
            actual == &shape
        }) || data.iter().any(|n| !n.is_finite())
            || (integer && data != &[0.])
        {
            return Err(error(format!("Invalid fixture {key}")));
        }
    }
    Ok(())
}
pub fn block_source_key(path: &str) -> String {
    source_key(&format!("inc.{path}"))
        .strip_prefix("inc.inconv.")
        .unwrap()
        .to_owned()
}
pub fn load_block(root: &Path, device: &Device) -> Result<super::visual_blocks::InvertedResidual> {
    let r = Reference::load(root, "visual_inverted_residual")?;
    let b = super::visual_parameters::Builder {
        device: device.clone(),
    };
    let model = super::visual_blocks::InvertedResidual::new(&b, 6, 24, 1, false);
    let mut values = BTreeMap::new();
    for (key, value) in r.values {
        if let Some(key) = key.strip_prefix("state.") {
            let mut key = key.to_owned();
            for (index, name) in [
                (0, "expand"),
                (1, "expand_bn"),
                (3, "depthwise"),
                (4, "depth_bn"),
                (6, "project"),
                (7, "project_bn"),
            ] {
                key = key.replace(&format!("conv.{index}."), &format!("{name}."));
            }
            values.insert(key, value);
        }
    }
    validate_values(&model, &values, 18)?;
    let mut loader = Loader {
        path: vec![],
        values,
    };
    let model = model.map(&mut loader);
    assert!(loader.values.is_empty());
    Ok(model)
}
