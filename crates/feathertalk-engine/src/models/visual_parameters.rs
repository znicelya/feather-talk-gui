use super::visual_state::{Buffers, SourceBatchNorm, State};
use burn::{
    module::{Module, Param},
    tensor::{
        Device, Distribution, Tensor,
        module::{conv1d, conv2d},
        ops::ConvOptions,
    },
};
pub(crate) struct Builder {
    pub device: Device,
}
impl Builder {
    pub fn uniform<const D: usize>(&self, shape: [usize; D], fan: usize) -> Param<Tensor<D>> {
        let bound = 1.0 / (fan as f64).sqrt();
        Param::from_tensor(Tensor::random(
            shape,
            Distribution::Uniform(-bound, bound),
            &self.device,
        ))
    }
    pub fn constant<const D: usize>(&self, shape: [usize; D], value: f32) -> Param<Tensor<D>> {
        Param::from_tensor(Tensor::full(shape, value, &self.device))
    }
    pub fn bn(&self, c: usize) -> SourceBatchNorm {
        SourceBatchNorm {
            weight: self.constant([c], 1.),
            bias: self.constant([c], 0.),
            training: Param::from_bool(true),
            state: State::new(Buffers {
                mean: self.constant([c], 0.).set_require_grad(false),
                variance: self.constant([c], 1.).set_require_grad(false),
                count: Param::initialized(
                    burn::module::ParamId::new(),
                    Tensor::<1, burn::tensor::Int>::from_data(
                        burn::tensor::TensorData::new(vec![0i64], [1]),
                        &self.device,
                    )
                    .cast(burn::tensor::IntDType::I64),
                ),
            }),
        }
    }
}
#[derive(Module, Debug)]
pub struct Conv2 {
    pub weight: Param<Tensor<4>>,
    pub bias: Option<Param<Tensor<1>>>,
    pub stride: usize,
    pub padding: usize,
    pub groups: usize,
}
impl Conv2 {
    pub(crate) fn new(
        b: &Builder,
        i: usize,
        o: usize,
        k: usize,
        s: usize,
        g: usize,
        bias: bool,
    ) -> Self {
        let fan = i / g * k * k;
        Self {
            weight: b.uniform([o, i / g, k, k], fan),
            bias: bias.then(|| b.uniform([o], fan)),
            stride: s,
            padding: k / 2,
            groups: g,
        }
    }
    pub fn forward(&self, x: Tensor<4>) -> Tensor<4> {
        conv2d(
            x,
            self.weight.val(),
            self.bias.as_ref().map(|b| b.val()),
            ConvOptions::new([self.stride; 2], [self.padding; 2], [1; 2], self.groups),
        )
    }
}
#[derive(Module, Debug)]
pub struct Conv1 {
    pub weight: Param<Tensor<3>>,
    pub bias: Option<Param<Tensor<1>>>,
    pub dilation: usize,
    pub groups: usize,
}
impl Conv1 {
    pub(crate) fn new(
        b: &Builder,
        i: usize,
        o: usize,
        k: usize,
        d: usize,
        g: usize,
        bias: bool,
    ) -> Self {
        let fan = i / g * k;
        Self {
            weight: b.uniform([o, i / g, k], fan),
            bias: bias.then(|| b.uniform([o], fan)),
            dilation: d,
            groups: g,
        }
    }
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        conv1d(
            x,
            self.weight.val(),
            self.bias.as_ref().map(|b| b.val()),
            ConvOptions::new(
                [1],
                [self.dilation * (self.weight.dims()[2] / 2)],
                [self.dilation],
                self.groups,
            ),
        )
    }
}
#[derive(Module, Debug)]
pub struct Linear {
    pub weight: Param<Tensor<2>>,
    pub bias: Param<Tensor<1>>,
}
impl Linear {
    pub(crate) fn new(b: &Builder, i: usize, o: usize) -> Self {
        Self {
            weight: b.uniform([o, i], i),
            bias: b.uniform([o], i),
        }
    }
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [o, i] = self.weight.dims();
        x.matmul(self.weight.val().reshape([1, o, i]).swap_dims(1, 2))
            + self.bias.val().reshape([1, 1, o])
    }
}
#[derive(Module, Debug)]
pub struct LayerNorm {
    pub weight: Param<Tensor<1>>,
    pub bias: Param<Tensor<1>>,
}
impl LayerNorm {
    pub(crate) fn new(b: &Builder, c: usize) -> Self {
        Self {
            weight: b.constant([c], 1.),
            bias: b.constant([c], 0.),
        }
    }
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let c = x.dims()[2];
        let x = x.clone() - x.mean_dim(2);
        let var = (x.clone() * x.clone()).mean_dim(2);
        x / (var + 1e-5).sqrt() * self.weight.val().reshape([1, 1, c])
            + self.bias.val().reshape([1, 1, c])
    }
}
