use super::ghost_one::{Conv, GhostOne, MobileOne, Parameters};
use crate::verified_assets::{VerifiedAssets, error};
use burn::tensor::{Device, Tensor, activation::relu, module::avg_pool2d};
use feathertalk_core::Result;
use std::path::Path;
pub struct PfldGhostOne {
    device: Device,
    first: MobileOne,
    second: MobileOne,
    stages: Vec<(String, GhostOne)>,
    seventh: MobileOne,
    eighth: Conv,
    out: Conv,
    pub inactive_parameters: Vec<String>,
    pub accounted_counters: usize,
}
impl PfldGhostOne {
    pub fn load(path: &Path, d: &Device) -> Result<Self> {
        let a = VerifiedAssets::load(path)?;
        let id = &a.manifest.identity;
        if id.kind != "pfld_ghost_one"
            || id.architecture_version != "pfld_ghost_one-source-v1"
            || id.source_sha256
                != "bada866661ad5fa1080a085f51fe9c016c69958c406951afa4afc7840f856de0"
        {
            return Err(error("PFLD identity mismatch"));
        }
        let c: serde_json::Value = serde_json::from_slice(
            a.files
                .get("config.json")
                .ok_or_else(|| error("Missing config"))?,
        )
        .map_err(error)?;
        if c["source_sha256"] != id.source_sha256
            || c["architecture_version"] != id.architecture_version
            || c["configuration"]
                != serde_json::json!({"width_factor":0.5,"input_size":192,"landmark_number":110,"inference_mode":false,"num_conv_branches":6})
        {
            return Err(error("PFLD configuration mismatch"));
        }
        let mut p = Parameters::read(
            a.files
                .get("model.safetensors")
                .ok_or_else(|| error("Missing weights"))?,
        )?;
        let accounted_counters = p.counters;
        let first = MobileOne::load(&mut p, "conv1", 3, 32, 3, 2, 1, false, d)?;
        let second = MobileOne::load(&mut p, "conv2", 32, 32, 3, 1, 32, false, d)?;
        let mut stages = vec![];
        for (name, i, h, o, s) in [
            ("conv3_1", 32, 48, 40, 2),
            ("conv3_2", 40, 60, 40, 1),
            ("conv3_3", 40, 60, 40, 1),
            ("conv4_1", 40, 100, 48, 2),
            ("conv4_2", 48, 120, 48, 1),
            ("conv4_3", 48, 120, 48, 1),
            ("conv5_1", 48, 168, 72, 2),
            ("conv5_2", 72, 252, 72, 1),
            ("conv5_3", 72, 252, 72, 1),
            ("conv5_4", 72, 252, 72, 1),
            ("conv6", 72, 108, 8, 1),
        ] {
            stages.push((name.into(), GhostOne::load(&mut p, name, i, h, o, s, d)?))
        }
        let seventh = MobileOne::load(&mut p, "conv7", 8, 16, 3, 1, 1, false, d)?;
        let eighth = Conv::load(&mut p, "conv8.0", 16, 64, 12, 1, 0, 1, false, d)?;
        let out = Conv::load(&mut p, "conv_out", 256, 220, 1, 1, 0, 1, true, d)?;
        let inactive_parameters = p.finish()?;
        Ok(Self {
            device: d.clone(),
            first,
            second,
            stages,
            seventh,
            eighth,
            out,
            inactive_parameters,
            accounted_counters,
        })
    }
    pub fn device(&self) -> &Device {
        &self.device
    }
    pub fn forward(&self, x: Tensor<4>) -> Tensor<2> {
        self.forward_observed(x, |_, _| {})
    }
    pub fn forward_observed(
        &self,
        x: Tensor<4>,
        mut tap: impl FnMut(&str, Tensor<4>),
    ) -> Tensor<2> {
        let mut x = self.first.forward(x);
        tap("conv1", x.clone());
        x = self.second.forward(x);
        let pool = |x: Tensor<4>| {
            let s = x.dims()[2];
            avg_pool2d(
                x,
                burn::tensor::ops::AvgPoolOptions::new([s, s]).with_stride([s, s]),
            )
        };
        let mut features = vec![pool(x.clone())];
        for (name, stage) in &self.stages {
            x = stage.forward(x);
            if ["conv3_3", "conv4_3", "conv5_4"].contains(&name.as_str()) {
                tap(name, x.clone());
                features.push(pool(x.clone()));
            }
        }
        x = relu(self.eighth.forward(self.seventh.forward(x)));
        tap("conv8", x.clone());
        features.push(x);
        let y = self.out.forward(Tensor::cat(features, 1));
        tap("conv_out", y.clone());
        let b = y.dims()[0];
        y.reshape([b, 220])
    }
}
