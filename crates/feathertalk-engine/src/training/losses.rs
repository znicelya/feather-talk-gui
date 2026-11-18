//! Source loss reductions; all parts remain on the selected device until sampled.
use crate::models::vgg::VggFeatures;
use burn::tensor::{Tensor, module::avg_pool2d, ops::AvgPoolOptions};
use feathertalk_core::{LossConfig, LossScalars};
pub struct LossParts {
    pub full: Tensor<1>,
    pub mouth: Tensor<1>,
    pub temporal: Tensor<1>,
    pub temporal_mouth: Tensor<1>,
    pub perceptual: Tensor<1>,
    pub mouth_gradient: Tensor<1>,
    total: Tensor<1>,
}
pub struct LossOutput {
    pub total: Tensor<1>,
    pub parts: LossParts,
}
impl LossParts {
    /// Synchronize metrics only when the caller samples a training update.
    pub fn to_scalars(&self) -> LossScalars {
        let scalar = |t: &Tensor<1>| t.clone().into_scalar::<f32>() as f64;
        LossScalars {
            full: scalar(&self.full),
            mouth: scalar(&self.mouth),
            temporal: scalar(&self.temporal),
            temporal_mouth: scalar(&self.temporal_mouth),
            perceptual: scalar(&self.perceptual),
            mouth_gradient: scalar(&self.mouth_gradient),
            total: scalar(&self.total),
        }
    }
}
fn masked(error: Tensor<4>, mask: Tensor<4>, channels: usize) -> Tensor<1> {
    (error * mask.clone()).sum() / (mask.sum().clamp_min(1.) * channels as f32)
}
pub fn compute_loss(
    prediction: Tensor<5>,
    target: Tensor<5>,
    mouth_mask: Tensor<5>,
    vgg: &VggFeatures,
    config: &LossConfig,
) -> LossOutput {
    let [b, t, c, h, w] = prediction.dims();
    assert_eq!(t, 2, "Training requires adjacent frame pairs");
    assert_eq!(c, 3);
    assert_eq!(target.dims(), [b, t, c, h, w]);
    assert_eq!(mouth_mask.dims(), [b, t, 1, h, w]);
    let device = prediction.device();
    let p = prediction.reshape([b * t, c, h, w]);
    let y = target.reshape([b * t, c, h, w]);
    let m = mouth_mask.clone().reshape([b * t, 1, h, w]);
    let diff = (p.clone() - y.clone()).abs();
    let full = diff.clone().mean();
    let mouth = masked(diff, m.clone(), c);
    let (temporal, temporal_mouth) = if config.temporal != 0. || config.temporal_mouth != 0. {
        let k = config.temporal_kernel as usize;
        let blur = |x: Tensor<4>| {
            if k > 1 {
                avg_pool2d(
                    x,
                    AvgPoolOptions::new([k, k])
                        .with_stride([1, 1])
                        .with_padding([k / 2, k / 2])
                        .with_count_include_pad(true),
                )
            } else {
                x
            }
        };
        let delta = |x: Tensor<4>| {
            let x = x.reshape([b, t, c, h, w]);
            x.clone()
                .slice([0..b, 1..2, 0..c, 0..h, 0..w])
                .reshape([b, c, h, w])
                - x.slice([0..b, 0..1, 0..c, 0..h, 0..w])
                    .reshape([b, c, h, w])
        };
        let e = (delta(blur(p.clone())) - delta(blur(y.clone()))).abs();
        let union = mouth_mask.max_dim(1).reshape([b, 1, h, w]);
        (e.clone().mean(), masked(e, union, c))
    } else {
        (Tensor::zeros([1], &device), Tensor::zeros([1], &device))
    };
    let dx = |x: Tensor<4>| {
        x.clone().slice([0..b * t, 0..c, 0..h, 1..w]) - x.slice([0..b * t, 0..c, 0..h, 0..w - 1])
    };
    let dy = |x: Tensor<4>| {
        x.clone().slice([0..b * t, 0..c, 1..h, 0..w]) - x.slice([0..b * t, 0..c, 0..h - 1, 0..w])
    };
    let mx = m.clone().slice([0..b * t, 0..1, 0..h, 1..w])
        * m.clone().slice([0..b * t, 0..1, 0..h, 0..w - 1]);
    let my =
        m.clone().slice([0..b * t, 0..1, 1..h, 0..w]) * m.slice([0..b * t, 0..1, 0..h - 1, 0..w]);
    let mouth_gradient = (((dx(p.clone()) - dx(y.clone())).abs() * mx.clone()).sum()
        + ((dy(p.clone()) - dy(y.clone())).abs() * my.clone()).sum())
        / ((mx.sum() + my.sum()).clamp_min(1.) * c as f32);
    let f = vgg.forward(p)
        - vgg
            .forward(y.set_require_grad(false))
            .set_require_grad(false);
    let perceptual = f.clone().mul(f).mean();
    let total = full.clone() * config.full
        + mouth.clone() * config.mouth
        + temporal.clone() * config.temporal
        + temporal_mouth.clone() * config.temporal_mouth
        + perceptual.clone() * config.perceptual
        + mouth_gradient.clone() * config.mouth_gradient;
    LossOutput {
        total: total.clone(),
        parts: LossParts {
            full,
            mouth,
            temporal,
            temporal_mouth,
            perceptual,
            mouth_gradient,
            total,
        },
    }
}
