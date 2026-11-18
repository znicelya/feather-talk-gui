use super::visual_parameters::{Builder, LayerNorm, Linear};
use burn::{
    module::{Module, Param},
    tensor::{
        Tensor, TensorData,
        activation::{sigmoid, softmax},
    },
};
#[derive(Module, Debug)]
pub struct SpatialAudioAttention {
    pub visual_norm: LayerNorm,
    pub audio_norm: LayerNorm,
    pub query: Linear,
    pub key: Linear,
    pub value: Linear,
    pub out: Linear,
    pub audio_position: Param<Tensor<2>>,
    pub relative_bias: Param<Tensor<2>>,
    pub residual_gate: Param<Tensor<1>>,
    pub heads: usize,
    pub attention_dim: usize,
}
impl SpatialAudioAttention {
    pub(crate) fn new(
        b: &Builder,
        c: usize,
        a: usize,
        d: usize,
        heads: usize,
        t: usize,
        center: usize,
    ) -> Self {
        assert!(c % heads == 0 && d % heads == 0);
        Self {
            visual_norm: LayerNorm::new(b, c),
            audio_norm: LayerNorm::new(b, a),
            query: Linear::new(b, c, d),
            key: Linear::new(b, a, d),
            value: Linear::new(b, a, c),
            out: Linear::new(b, c, c),
            audio_position: b.constant([t, a], 0.),
            relative_bias: Param::from_tensor(Tensor::from_data(
                TensorData::new(
                    (0..heads * t)
                        .map(|i| -0.12 * (i % t).abs_diff(center) as f32)
                        .collect::<Vec<_>>(),
                    [heads, t],
                ),
                &b.device,
            )),
            residual_gate: b.constant([1], -1.5),
            heads,
            attention_dim: d,
        }
    }
    pub fn forward(&self, visual: Tensor<4>, audio: Tensor<3>) -> Tensor<4> {
        let [b, c, h, w] = visual.dims();
        let [ab, t, a] = audio.dims();
        assert_eq!(ab, b);
        assert!(t <= self.audio_position.dims()[0]);
        let n = self.heads;
        let d = self.attention_dim / n;
        let tokens = visual.reshape([b, c, h * w]).swap_dims(1, 2);
        let audio = self.audio_norm.forward(
            audio
                + self
                    .audio_position
                    .val()
                    .slice([0..t, 0..a])
                    .reshape([1, t, a]),
        );
        let q = self
            .query
            .forward(self.visual_norm.forward(tokens.clone()))
            .reshape([b, h * w, n, d])
            .swap_dims(1, 2);
        let k = self
            .key
            .forward(audio.clone())
            .reshape([b, t, n, d])
            .swap_dims(1, 2);
        let v = self
            .value
            .forward(audio)
            .reshape([b, t, n, c / n])
            .swap_dims(1, 2);
        let logits = q.matmul(k.swap_dims(2, 3)) / (d as f64).sqrt()
            + self
                .relative_bias
                .val()
                .slice([0..n, 0..t])
                .reshape([1, n, 1, t]);
        let context = softmax(logits, 3)
            .matmul(v)
            .swap_dims(1, 2)
            .reshape([b, h * w, c]);
        (tokens + sigmoid(self.residual_gate.val().reshape([1, 1, 1])) * self.out.forward(context))
            .swap_dims(1, 2)
            .reshape([b, c, h, w])
    }
}
