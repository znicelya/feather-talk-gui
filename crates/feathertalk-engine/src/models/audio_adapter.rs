use super::{
    visual_parameters::{Builder, Conv1, LayerNorm, Linear},
    visual_state::SourceBatchNorm,
};
use burn::{
    module::Module,
    tensor::{
        Tensor,
        activation::{silu, softmax},
    },
};
#[derive(Module, Debug)]
pub struct TemporalBlock {
    pub depthwise: Conv1,
    pub bn: SourceBatchNorm,
    pub pointwise: Conv1,
    pub point_bn: SourceBatchNorm,
}
impl TemporalBlock {
    fn new(b: &Builder, d: usize) -> Self {
        Self {
            depthwise: Conv1::new(b, 512, 512, 5, d, 512, false),
            bn: b.bn(512),
            pointwise: Conv1::new(b, 512, 512, 1, 1, 1, false),
            point_bn: b.bn(512),
        }
    }
    fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let y = silu(self.bn.temporal(self.depthwise.forward(x.clone())));
        silu(x + self.point_bn.temporal(self.pointwise.forward(y)))
    }
}
#[derive(Module, Debug)]
pub struct AudioAdapter {
    pub norm: LayerNorm,
    pub input_projection: Conv1,
    pub blocks: Vec<TemporalBlock>,
    pub output_projection: Conv1,
}
impl AudioAdapter {
    pub(crate) fn new(b: &Builder) -> Self {
        Self {
            norm: LayerNorm::new(b, 2048),
            input_projection: Conv1::new(b, 2048, 512, 1, 1, 1, true),
            blocks: [1, 2, 4, 8]
                .into_iter()
                .map(|d| TemporalBlock::new(b, d))
                .collect(),
            output_projection: Conv1::new(b, 512, 512, 1, 1, 1, true),
        }
    }
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, t, c] = x.dims();
        assert_eq!([t, c], [40, 1024]);
        let mut x = self
            .input_projection
            .forward(self.norm.forward(x.reshape([b, 20, 2048])).swap_dims(1, 2));
        for block in &self.blocks {
            x = block.forward(x);
        }
        self.output_projection.forward(x).swap_dims(1, 2)
    }
}
#[derive(Module, Debug)]
pub struct CenteredAudioContext {
    pub score: Linear,
}
impl CenteredAudioContext {
    pub(crate) fn new(b: &Builder) -> Self {
        Self {
            score: Linear::new(b, 512, 1),
        }
    }
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, t, c] = x.dims();
        let center = 10.min(t - 1);
        let start = center.saturating_sub(2);
        let end = (center + 3).min(t);
        let local = x.clone().slice([0..b, start..end, 0..c]);
        let scores = softmax(self.score.forward(local.clone()), 1);
        let context = (local * scores).sum_dim(1);
        Tensor::cat(vec![x.slice([0..b, center..center + 1, 0..c]), context], 2)
    }
}
