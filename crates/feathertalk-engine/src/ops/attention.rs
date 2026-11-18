use burn::tensor::{
    Tensor,
    activation::{sigmoid, softmax},
};
use std::collections::BTreeMap;
/// Spatial audio attention with PyTorch-layout flattened parameters.
pub struct SpatialAudioAttention {
    pub parameters: BTreeMap<String, Tensor<1>>,
    pub heads: usize,
    pub attention_dim: usize,
}
impl SpatialAudioAttention {
    fn p<const D: usize>(&self, name: &str, shape: [usize; D]) -> Tensor<D> {
        self.parameters[name].clone().reshape(shape)
    }
    fn norm(&self, x: Tensor<3>, name: &str) -> Tensor<3> {
        let c = x.dims()[2];
        let centered = x.clone() - x.mean_dim(2);
        let var = centered.clone().mul(centered.clone()).mean_dim(2);
        centered / (var + 1e-5).sqrt() * self.p(&format!("{name}.weight"), [1, 1, c])
            + self.p(&format!("{name}.bias"), [1, 1, c])
    }
    fn linear(&self, x: Tensor<3>, name: &str, out: usize) -> Tensor<3> {
        let c = x.dims()[2];
        x.matmul(
            self.p(&format!("{name}.weight"), [1, out, c])
                .swap_dims(1, 2),
        ) + self.p(&format!("{name}.bias"), [1, 1, out])
    }
    pub fn forward(&self, visual: Tensor<4>, audio: Tensor<3>) -> Tensor<4> {
        let [b, c, h, w] = visual.dims();
        let [_, t, a] = audio.dims();
        let n = self.heads;
        let d = self.attention_dim / n;
        let tokens = visual.reshape([b, c, h * w]).swap_dims(1, 2);
        let audio = audio + self.p("audio_position", [1, t, a]);
        let q = self
            .linear(
                self.norm(tokens.clone(), "visual_norm"),
                "query",
                self.attention_dim,
            )
            .reshape([b, h * w, n, d])
            .swap_dims(1, 2);
        let audio = self.norm(audio, "audio_norm");
        let k = self
            .linear(audio.clone(), "key", self.attention_dim)
            .reshape([b, t, n, d])
            .swap_dims(1, 2);
        let v = self
            .linear(audio, "value", c)
            .reshape([b, t, n, c / n])
            .swap_dims(1, 2);
        let logits =
            q.matmul(k.swap_dims(2, 3)) / (d as f64).sqrt() + self.p("relative_bias", [1, n, 1, t]);
        let context = softmax(logits, 3)
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([b, h * w, c]);
        (tokens + sigmoid(self.p("residual_gate", [1, 1, 1])) * self.linear(context, "out", c))
            .swap_dims(1, 2)
            .reshape([b, c, h, w])
    }
}
