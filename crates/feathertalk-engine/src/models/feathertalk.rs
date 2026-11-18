//! Default trainable FeatherTalk character network.
use super::{
    attention::SpatialAudioAttention,
    audio_adapter::{AudioAdapter, CenteredAudioContext},
    visual_blocks::{DoubleConv, InvertedResidual, up},
    visual_parameters::{Builder, Conv2, Linear},
};
use burn::{
    module::Module,
    tensor::{Device, Tensor, activation::sigmoid},
};
#[derive(Clone, Debug)]
pub struct ImagePyramid {
    pub x1: Tensor<4>,
    pub x2: Tensor<4>,
    pub x3: Tensor<4>,
    pub x4: Tensor<4>,
    pub x5: Tensor<4>,
}
#[derive(Module, Debug)]
pub struct FeatherTalk {
    pub inc: InvertedResidual,
    pub down1: DoubleConv,
    pub down2: DoubleConv,
    pub down3: DoubleConv,
    pub down4: DoubleConv,
    pub audio_model: AudioAdapter,
    pub audio_context: CenteredAudioContext,
    pub audio_to_x5: Linear,
    pub spatial_x3: SpatialAudioAttention,
    pub spatial_x4: SpatialAudioAttention,
    pub spatial_x5: SpatialAudioAttention,
    pub fuse_conv: DoubleConv,
    pub up1: DoubleConv,
    pub up2: DoubleConv,
    pub up3: DoubleConv,
    pub up4: DoubleConv,
    pub outc: Conv2,
}
impl FeatherTalk {
    pub fn new(device: &Device) -> Self {
        let b = &Builder {
            device: device.clone(),
        };
        Self {
            inc: InvertedResidual::new(b, 6, 24, 1, false),
            down1: DoubleConv::new(b, 24, 32, 2),
            down2: DoubleConv::new(b, 32, 64, 2),
            down3: DoubleConv::new(b, 64, 128, 2),
            down4: DoubleConv::new(b, 128, 256, 2),
            audio_model: AudioAdapter::new(b),
            audio_context: CenteredAudioContext::new(b),
            audio_to_x5: Linear::new(b, 1024, 256),
            spatial_x3: SpatialAudioAttention::new(b, 64, 512, 128, 4, 20, 10),
            spatial_x4: SpatialAudioAttention::new(b, 128, 512, 128, 4, 20, 10),
            spatial_x5: SpatialAudioAttention::new(b, 256, 512, 128, 4, 20, 10),
            fuse_conv: DoubleConv::new(b, 512, 256, 1),
            up1: DoubleConv::new(b, 384, 128, 1),
            up2: DoubleConv::new(b, 192, 64, 1),
            up3: DoubleConv::new(b, 96, 32, 1),
            up4: DoubleConv::new(b, 56, 24, 1),
            outc: Conv2::new(b, 24, 3, 1, 1, 1, true),
        }
    }
    pub fn encode_image(&self, image: Tensor<4>) -> ImagePyramid {
        let [_, c, h, w] = image.dims();
        assert_eq!([c, h, w], [6, 144, 144]);
        let x1 = self.inc.forward(image);
        let x2 = self.down1.forward(x1.clone());
        let x3 = self.down2.forward(x2.clone());
        let x4 = self.down3.forward(x3.clone());
        let x5 = self.down4.forward(x4.clone());
        ImagePyramid { x1, x2, x3, x4, x5 }
    }
    pub fn decode_features(&self, f: ImagePyramid, audio: Tensor<3>) -> Tensor<4> {
        let audio = self.audio_model.forward(audio);
        let x3 = self.spatial_x3.forward(f.x3, audio.clone());
        let x4 = self.spatial_x4.forward(f.x4, audio.clone());
        let x5 = self.spatial_x5.forward(f.x5, audio.clone());
        let [b, c, h, w] = x5.dims();
        let context = self
            .audio_to_x5
            .forward(self.audio_context.forward(audio))
            .reshape([b, c, 1, 1])
            .expand([b, c, h, w]);
        let x = self.fuse_conv.forward(Tensor::cat(vec![x5, context], 1));
        let x = up(&self.up1, x, x4);
        let x = up(&self.up2, x, x3);
        let x = up(&self.up3, x, f.x2);
        let x = up(&self.up4, x, f.x1);
        sigmoid(self.outc.forward(x))
    }
    pub fn forward(&self, image: Tensor<4>, audio: Tensor<3>) -> Tensor<4> {
        self.decode_features(self.encode_image(image), audio)
    }
}
