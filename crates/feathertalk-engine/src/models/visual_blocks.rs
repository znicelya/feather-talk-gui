use super::{
    visual_parameters::{Builder, Conv2},
    visual_state::SourceBatchNorm,
};
use burn::{
    module::Module,
    tensor::{Tensor, activation::relu},
};
#[derive(Module, Debug)]
pub struct InvertedResidual {
    pub expand: Conv2,
    pub expand_bn: SourceBatchNorm,
    pub depthwise: Conv2,
    pub depth_bn: SourceBatchNorm,
    pub project: Conv2,
    pub project_bn: SourceBatchNorm,
    pub residual: bool,
}
impl InvertedResidual {
    pub(crate) fn new(b: &Builder, i: usize, o: usize, s: usize, residual: bool) -> Self {
        Self {
            expand: Conv2::new(b, i, 2 * i, 1, 1, 1, false),
            expand_bn: b.bn(2 * i),
            depthwise: Conv2::new(b, 2 * i, 2 * i, 3, s, 2 * i, false),
            depth_bn: b.bn(2 * i),
            project: Conv2::new(b, 2 * i, o, 1, 1, 1, false),
            project_bn: b.bn(o),
            residual,
        }
    }
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        let y = relu(self.expand_bn.forward(self.expand.forward(x.clone())));
        let y = relu(self.depth_bn.forward(self.depthwise.forward(y)));
        let y = self.project_bn.forward(self.project.forward(y));
        if self.residual { x + y } else { y }
    }
}
#[derive(Module, Debug)]
pub struct DoubleConv {
    pub blocks: Vec<InvertedResidual>,
}
impl DoubleConv {
    pub(crate) fn new(b: &Builder, i: usize, o: usize, s: usize) -> Self {
        Self {
            blocks: vec![
                InvertedResidual::new(b, i, o, s, false),
                InvertedResidual::new(b, o, o, 1, true),
            ],
        }
    }
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        self.blocks[1].forward(self.blocks[0].forward(x))
    }
}
/// Source scale-by-two interpolation followed by Python floor-divided padding/cropping.
pub fn upsample_to_skip(x: Tensor<4>, skip_hw: [usize; 2]) -> Tensor<4> {
    let [_, _, h, w] = x.dims();
    let mut x = crate::ops::resize::resize_bilinear_align_corners(x, [h * 2, w * 2]);
    for (axis, target) in [(2, skip_hw[0]), (3, skip_hw[1])] {
        let dims = x.dims();
        let current = dims[axis];
        if current > target {
            let start = (current - target).div_ceil(2);
            let mut ranges = dims.map(|n| 0..n);
            ranges[axis] = start..start + target;
            x = x.slice(ranges);
        } else if current < target {
            let left = (target - current) / 2;
            let right = target - current - left;
            let mut pieces = Vec::new();
            if left > 0 {
                let mut s = dims;
                s[axis] = left;
                pieces.push(Tensor::zeros(s, &x.device()));
            }
            pieces.push(x.clone());
            if right > 0 {
                let mut s = dims;
                s[axis] = right;
                pieces.push(Tensor::zeros(s, &x.device()));
            }
            x = Tensor::cat(pieces, axis);
        }
    }
    x
}
pub fn up(block: &DoubleConv, x: Tensor<4>, skip: Tensor<4>) -> Tensor<4> {
    let [_, _, h, w] = skip.dims();
    block.forward(Tensor::cat(vec![upsample_to_skip(x, [h, w]), skip], 1))
}
