//! Eager, independently cloned normalization buffers. No pending running updates.
use burn::{
    module::{
        Content, Flag, Module, ModuleDisplay, ModuleDisplayDefault, ModuleMapper, ModuleVisitor,
        Param,
    },
    tensor::{Device, Int, Tensor},
};
use std::sync::Mutex;

#[derive(Module, Debug)]
pub struct Buffers {
    pub mean: Param<Tensor<1>>,
    pub variance: Param<Tensor<1>>,
    pub count: Param<Tensor<1, Int>>,
}
#[derive(Debug)]
pub struct State(Mutex<Buffers>);
impl Clone for State {
    fn clone(&self) -> Self {
        Self(Mutex::new(self.0.lock().unwrap().clone()))
    }
}
impl ModuleDisplayDefault for State {
    fn content(&self, c: Content) -> Option<Content> {
        Some(c)
    }
}
impl ModuleDisplay for State {}
impl State {
    pub fn new(b: Buffers) -> Self {
        Self(Mutex::new(b))
    }
    pub fn get(&self) -> Buffers {
        self.0.lock().unwrap().clone()
    }
    pub fn update(&self, mean: Tensor<1>, variance: Tensor<1>) {
        let mut b = self.0.lock().unwrap();
        b.mean = Param::initialized(b.mean.id, mean.detach().set_require_grad(false));
        b.variance = Param::initialized(b.variance.id, variance.detach().set_require_grad(false));
        b.count = Param::initialized(b.count.id, b.count.val() + 1);
    }
    fn transform(self, f: impl FnOnce(Buffers) -> Buffers) -> Self {
        let b = f(self.0.into_inner().unwrap()).set_require_grad(false);
        Self::new(b)
    }
}
impl Module for State {
    fn collect_devices(&self, d: Vec<Device>) -> Vec<Device> {
        self.get().collect_devices(d)
    }
    fn fork(self, d: &Device) -> Self {
        self.transform(|b| b.fork(d))
    }
    fn to_device(self, d: &Device) -> Self {
        self.transform(|b| b.to_device(d))
    }
    fn train(self) -> Self {
        self.transform(|b| b.train())
    }
    fn valid(&self) -> Self {
        Self::new(self.get().valid().set_require_grad(false))
    }
    fn materialize(self) -> Self {
        self.transform(|b| b.materialize())
    }
    fn visit<V: ModuleVisitor>(&self, v: &mut V) {
        self.0.lock().unwrap().visit(v);
    }
    fn map<M: ModuleMapper>(self, m: &mut M) -> Self {
        self.transform(|b| b.map(m))
    }
}
#[derive(Module, Debug)]
pub struct SourceBatchNorm {
    pub weight: Param<Tensor<1>>,
    pub bias: Param<Tensor<1>>,
    pub training: Param<Flag>,
    pub state: State,
}
impl SourceBatchNorm {
    pub fn forward(&self, input: Tensor<4>) -> Tensor<4> {
        let [b, c, h, w] = input.dims();
        let training = input.device().is_autodiff() && self.training.is_enabled();
        let x = input.swap_dims(0, 1).reshape([c, b * h * w]);
        let state = self.state.get();
        let (mean, var) = if training {
            assert!(b * h * w > 1);
            // Flex's sequential FP32 reductions lose accuracy over 41,472 pixels.
            // Accumulate statistics in F64 on that same CPU device, preserving
            // autodiff, then use source FP32 normalization and affine arithmetic.
            #[cfg(feature = "flex")]
            let wide = x.device() == Device::flex();
            #[cfg(not(feature = "flex"))]
            let wide = false;
            let stats = if wide {
                x.clone().cast(burn::tensor::FloatDType::F64)
            } else {
                x.clone()
            };
            let mean = if wide {
                stats.clone().mean_dim(1)
            } else {
                stable_mean(stats.clone())
            };
            let centered = stats - mean.clone();
            let squared = centered.clone() * centered;
            let var = if wide {
                squared.mean_dim(1)
            } else {
                stable_mean(squared)
            };
            let mean = mean.cast(burn::tensor::FloatDType::F32);
            let var = var.cast(burn::tensor::FloatDType::F32);
            self.state.update(
                state.mean.val() * 0.9 + mean.clone().reshape([c]).detach() * 0.1,
                state.variance.val() * 0.9
                    + var.clone().reshape([c]).detach()
                        * (0.1 * (b * h * w) as f64 / (b * h * w - 1) as f64),
            );
            (mean, var)
        } else {
            (
                state.mean.val().reshape([c, 1]),
                state.variance.val().reshape([c, 1]),
            )
        };
        (((x - mean) / (var + 1e-5).sqrt()) * self.weight.val().reshape([c, 1])
            + self.bias.val().reshape([c, 1]))
        .reshape([c, b, h, w])
        .swap_dims(0, 1)
    }
    pub fn temporal(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, c, t] = x.dims();
        self.forward(x.reshape([b, c, t, 1])).reshape([b, c, t])
    }
}

// Bound each FP32 reduction rather than accumulating a full image in one lane.
// All reductions and their backward operations execute on the input device.
fn stable_mean(x: Tensor<2>) -> Tensor<2> {
    let [c, n] = x.dims();
    let mut chunk = 1;
    while chunk < 256 && n % (chunk * 2) == 0 {
        chunk *= 2;
    }
    x.reshape([c, n / chunk, chunk])
        .mean_dim(2)
        .mean_dim(1)
        .reshape([c, 1])
}
