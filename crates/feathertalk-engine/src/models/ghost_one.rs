//! Original six-branch MobileOne/GhostOne eval graph (no reparameterization).
use crate::verified_assets::error;
use burn::tensor::{
    Device, Tensor, TensorData, activation::relu, module::conv2d, ops::ConvOptions,
};
use feathertalk_core::Result;
use safetensors::{Dtype, SafeTensors};
use std::collections::BTreeMap;
pub(crate) struct Parameters {
    values: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
    pub counters: usize,
}
impl Parameters {
    pub fn read(bytes: &[u8]) -> Result<Self> {
        let safe = SafeTensors::deserialize(bytes).map_err(error)?;
        let mut values = BTreeMap::new();
        let mut counters = 0;
        for (key, t) in safe.tensors() {
            if key.ends_with(".num_batches_tracked") {
                if t.dtype() != Dtype::I64 || !t.shape().is_empty() {
                    return Err(error(format!("Invalid counter {key}")));
                }
                // Kept in inventory until its corresponding BN is constructed.
                values.insert(key, (vec![], vec![]));
                counters += 1;
                continue;
            }
            if t.dtype() != Dtype::F32 {
                return Err(error(format!("Invalid dtype {key}")));
            }
            let data: Vec<_> = t
                .data()
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            if data.iter().any(|v| !v.is_finite()) {
                return Err(error(format!("Nonfinite {key}")));
            }
            values.insert(key, (t.shape().to_vec(), data));
        }
        Ok(Self { values, counters })
    }
    pub fn take<const D: usize>(
        &mut self,
        key: &str,
        shape: [usize; D],
        device: &Device,
    ) -> Result<Tensor<D>> {
        let (s, v) = self
            .values
            .remove(key)
            .ok_or_else(|| error(format!("Missing parameter {key}")))?;
        if s != shape {
            return Err(error(format!(
                "Wrong shape {key}: {s:?}, expected {shape:?}"
            )));
        }
        Ok(Tensor::from_data(TensorData::new(v, shape), device))
    }
    pub fn finish(mut self) -> Result<Vec<String>> {
        let inactive = [
            ("localization.0.weight", vec![8, 1, 7, 7]),
            ("localization.0.bias", vec![8]),
            ("localization.3.weight", vec![10, 8, 5, 5]),
            ("localization.3.bias", vec![10]),
        ];
        let mut report = vec![];
        for (key, shape) in inactive {
            let (s, _) = self
                .values
                .remove(key)
                .ok_or_else(|| error(format!("Missing inactive parameter {key}")))?;
            if s != shape {
                return Err(error("Inactive localization shape mismatch"));
            }
            report.push(key.into());
        }
        if !self.values.is_empty() {
            return Err(error(format!(
                "Unexpected parameters: {:?}",
                self.values.keys()
            )));
        }
        Ok(report)
    }
}
struct Norm {
    scale: Tensor<1>,
    bias: Tensor<1>,
    mean: Tensor<1>,
    var: Tensor<1>,
}
impl Norm {
    fn load(p: &mut Parameters, key: &str, c: usize, d: &Device) -> Result<Self> {
        let count = p
            .values
            .remove(&format!("{key}.num_batches_tracked"))
            .ok_or_else(|| error("Missing BN counter"))?;
        if !count.0.is_empty() || !count.1.is_empty() {
            return Err(error("Invalid BN counter"));
        }
        Ok(Self {
            scale: p.take(&format!("{key}.weight"), [c], d)?,
            bias: p.take(&format!("{key}.bias"), [c], d)?,
            mean: p.take(&format!("{key}.running_mean"), [c], d)?,
            var: p.take(&format!("{key}.running_var"), [c], d)?,
        })
    }
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let c = x.dims()[1];
        (x - self.mean.clone().reshape([1, c, 1, 1]))
            / (self.var.clone().reshape([1, c, 1, 1]) + 1e-5).sqrt()
            * self.scale.clone().reshape([1, c, 1, 1])
            + self.bias.clone().reshape([1, c, 1, 1])
    }
}
pub(crate) struct Conv {
    weight: Tensor<4>,
    bias: Option<Tensor<1>>,
    stride: usize,
    pad: usize,
    groups: usize,
}
impl Conv {
    pub fn load(
        p: &mut Parameters,
        key: &str,
        i: usize,
        o: usize,
        k: usize,
        stride: usize,
        pad: usize,
        groups: usize,
        bias: bool,
        d: &Device,
    ) -> Result<Self> {
        Ok(Self {
            weight: p.take(&format!("{key}.weight"), [o, i / groups, k, k], d)?,
            bias: if bias {
                Some(p.take(&format!("{key}.bias"), [o], d)?)
            } else {
                None
            },
            stride,
            pad,
            groups,
        })
    }
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        conv2d(
            x,
            self.weight.clone(),
            self.bias.clone(),
            ConvOptions::new([self.stride; 2], [self.pad; 2], [1; 2], self.groups),
        )
    }
}
struct Branch {
    conv: Conv,
    norm: Norm,
}
impl Branch {
    fn load(
        p: &mut Parameters,
        key: &str,
        i: usize,
        o: usize,
        k: usize,
        s: usize,
        g: usize,
        d: &Device,
    ) -> Result<Self> {
        Ok(Self {
            conv: Conv::load(p, &format!("{key}.conv"), i, o, k, s, k / 2, g, false, d)?,
            norm: Norm::load(p, &format!("{key}.bn"), o, d)?,
        })
    }
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        self.norm.forward(self.conv.forward(x))
    }
}
pub(crate) struct MobileOne {
    branches: Vec<Branch>,
    scale: Option<Branch>,
    skip: Option<Norm>,
    linear: bool,
}
impl MobileOne {
    pub fn load(
        p: &mut Parameters,
        key: &str,
        i: usize,
        o: usize,
        k: usize,
        s: usize,
        g: usize,
        linear: bool,
        d: &Device,
    ) -> Result<Self> {
        let scale = if k > 1 {
            Some(Branch::load(
                p,
                &format!("{key}.rbr_scale"),
                i,
                o,
                1,
                s,
                g,
                d,
            )?)
        } else {
            None
        };
        let skip = if i == o && s == 1 {
            Some(Norm::load(p, &format!("{key}.rbr_skip"), i, d)?)
        } else {
            None
        };
        let mut branches = vec![];
        for n in 0..6 {
            branches.push(Branch::load(
                p,
                &format!("{key}.rbr_conv.{n}"),
                i,
                o,
                k,
                s,
                g,
                d,
            )?)
        }
        Ok(Self {
            branches,
            scale,
            skip,
            linear,
        })
    }
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let mut sum = self.scale.as_ref().map(|v| v.forward(x.clone()));
        if let Some(skip) = &self.skip {
            let y = skip.forward(x.clone());
            sum = Some(match sum {
                Some(a) => a + y,
                None => y,
            });
        }
        for branch in &self.branches {
            let y = branch.forward(x.clone());
            sum = Some(match sum {
                Some(a) => a + y,
                None => y,
            });
        }
        let y = sum.unwrap();
        if self.linear { y } else { relu(y) }
    }
}
struct GhostModule {
    primary: MobileOne,
    cheap: MobileOne,
}
impl GhostModule {
    fn load(
        p: &mut Parameters,
        key: &str,
        i: usize,
        o: usize,
        linear: bool,
        d: &Device,
    ) -> Result<Self> {
        let half = o.div_ceil(2);
        Ok(Self {
            primary: MobileOne::load(
                p,
                &format!("{key}.primary_conv"),
                i,
                half,
                1,
                1,
                1,
                linear,
                d,
            )?,
            cheap: MobileOne::load(
                p,
                &format!("{key}.cheap_operation"),
                half,
                half,
                3,
                1,
                half,
                linear,
                d,
            )?,
        })
    }
    fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let x = self.primary.forward(x);
        let y = self.cheap.forward(x.clone());
        Tensor::cat(vec![x, y], 1)
    }
}
pub(crate) struct GhostOne {
    first: GhostModule,
    depth: Option<MobileOne>,
    last: GhostModule,
}
impl GhostOne {
    pub fn load(
        p: &mut Parameters,
        key: &str,
        i: usize,
        h: usize,
        o: usize,
        s: usize,
        d: &Device,
    ) -> Result<Self> {
        Ok(Self {
            first: GhostModule::load(p, &format!("{key}.ghost_conv.0"), i, h, false, d)?,
            depth: if s > 1 {
                Some(MobileOne::load(
                    p,
                    &format!("{key}.ghost_conv.1"),
                    h,
                    h,
                    3,
                    s,
                    h,
                    true,
                    d,
                )?)
            } else {
                None
            },
            last: GhostModule::load(p, &format!("{key}.ghost_conv.2"), h, o, true, d)?,
        })
    }
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let mut x = self.first.forward(x);
        if let Some(depth) = &self.depth {
            x = depth.forward(x)
        }
        self.last.forward(x)
    }
}
