use burn::tensor::Tensor;
/// PyTorch-compatible state updates, with biased training normalization and
/// unbiased running variance. All math and gradients stay on the input device.
pub struct BatchNorm {
    pub weight: Tensor<1>,
    pub bias: Tensor<1>,
    pub running_mean: Tensor<1>,
    pub running_var: Tensor<1>,
    pub num_batches_tracked: u64,
    pub momentum: f64,
    pub epsilon: f64,
}
impl BatchNorm {
    pub fn forward(&mut self, input: Tensor<4>, train: bool) -> Tensor<4> {
        let [b, c, h, w] = input.dims();
        let x = input.swap_dims(0, 1).reshape([c, b * h * w]);
        let (mean, var) = if train {
            assert!(b * h * w > 1);
            let mean = x.clone().mean_dim(1);
            let centered = x.clone() - mean.clone();
            let var = centered.clone().mul(centered).mean_dim(1);
            self.running_mean = self.running_mean.clone() * (1.0 - self.momentum)
                + mean.clone().reshape([c]).detach() * self.momentum;
            self.running_var = self.running_var.clone() * (1.0 - self.momentum)
                + var.clone().reshape([c]).detach()
                    * (self.momentum * (b * h * w) as f64 / (b * h * w - 1) as f64);
            self.num_batches_tracked += 1;
            (mean, var)
        } else {
            (
                self.running_mean.clone().reshape([c, 1]),
                self.running_var.clone().reshape([c, 1]),
            )
        };
        (((x - mean) / (var + self.epsilon).sqrt()) * self.weight.clone().reshape([c, 1])
            + self.bias.clone().reshape([c, 1]))
        .reshape([c, b, h, w])
        .swap_dims(0, 1)
    }
}
pub fn group_norm(
    input: Tensor<3>,
    weight: Tensor<1>,
    bias: Tensor<1>,
    groups: usize,
    epsilon: f64,
) -> Tensor<3> {
    let [b, c, l] = input.dims();
    let x = input.reshape([b, groups, c / groups * l]);
    let centered = x.clone() - x.mean_dim(2);
    let var = centered.clone().mul(centered.clone()).mean_dim(2);
    (centered / (var + epsilon).sqrt()).reshape([b, c, l]) * weight.reshape([1, c, 1])
        + bias.reshape([1, c, 1])
}
