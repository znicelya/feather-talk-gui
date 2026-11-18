use crate::{
    data::{
        audio::AudioFeatures,
        geometry::{audio_window, invalid, prepare_frame},
        image::resize_linear,
        landmarks::{BgrFrame, Landmarks},
    },
    models::feathertalk::FeatherTalk,
};
use burn::{
    module::Module,
    tensor::{Device, Tensor, TensorData},
};
use feathertalk_core::*;
use std::io::{Read, Seek, SeekFrom};

#[derive(Clone)]
pub struct FramePicker {
    count: usize,
    index: usize,
    forward: bool,
}
impl FramePicker {
    pub fn new(count: usize) -> Result<Self> {
        if count < 2 {
            return Err(invalid("At least two resource frames are required"));
        }
        Ok(Self {
            count,
            index: 0,
            forward: true,
        })
    }
    pub fn next(&mut self) -> usize {
        if self.index == self.count - 1 {
            self.forward = false;
        }
        if self.index == 0 {
            self.forward = true;
        }
        if self.forward {
            self.index += 1;
        } else {
            self.index -= 1;
        }
        self.index
    }
    pub fn reset(&mut self) {
        self.index = 0;
        self.forward = true;
    }
}

/// Fresh source crop, entire center replacement, then source LINEAR back-resize.
pub fn compose(frame: &BgrFrame, landmarks: &Landmarks, prediction: &[f32]) -> Result<BgrFrame> {
    prepare_frame(frame, landmarks)?; // shared strict bounds and source crop policy
    if prediction.len() != 3 * 144 * 144 || prediction.iter().any(|v| !v.is_finite()) {
        return Err(invalid("Prediction must be finite BGR CHW 144x144"));
    }
    let x = landmarks.points[1][0] as usize;
    let y = landmarks.points[52][1] as usize;
    let size = landmarks.points[31][0] as usize - x;
    let mut data = Vec::with_capacity(size * size * 3);
    for row in y..y + size {
        data.extend_from_slice(
            &frame.data[row * frame.stride + x * 3..row * frame.stride + (x + size) * 3],
        );
    }
    let mut crop = resize_linear(
        &BgrFrame {
            width: size,
            height: size,
            stride: size * 3,
            data,
        },
        152,
        152,
        true,
    );
    for c in 0..3 {
        for row in 0..144 {
            for col in 0..144 {
                crop.data[((row + 4) * 152 + col + 4) * 3 + c] =
                    (prediction[c * 144 * 144 + row * 144 + col] * 255.).clamp(0., 255.) as u8;
            }
        }
    }
    let crop = resize_linear(&crop, size, size, false);
    let mut output = frame.clone();
    for row in 0..size {
        output.data[(y + row) * frame.stride + x * 3..(y + row) * frame.stride + (x + size) * 3]
            .copy_from_slice(&crop.data[row * crop.stride..(row + 1) * crop.stride]);
    }
    Ok(output)
}

/// Disk-backed canonical resources; only one original frame and one model forward resident.
pub struct FrameRenderer {
    model: FeatherTalk,
    device: Device,
    raw: std::fs::File,
    width: usize,
    height: usize,
    landmarks: Vec<Landmarks>,
    picker: FramePicker,
    next: u64,
}
impl FrameRenderer {
    pub fn new(
        model: FeatherTalk,
        raw: std::fs::File,
        width: usize,
        height: usize,
        landmarks: Vec<Landmarks>,
    ) -> Result<Self> {
        if width == 0 || height == 0 || width > 16384 || height > 16384 {
            return Err(invalid("Unsupported resource canvas"));
        }
        let picker = FramePicker::new(landmarks.len())?;
        let expected = width
            .checked_mul(height)
            .and_then(|n| n.checked_mul(3))
            .and_then(|n| n.checked_mul(landmarks.len()))
            .ok_or_else(|| invalid("Resource size overflow"))?;
        if raw.metadata().map_err(invalid)?.len() != expected as u64 {
            return Err(invalid("Canonical resource count or canvas mismatch"));
        }
        let model = model.valid();
        let device = model.devices()[0].clone();
        Ok(Self {
            model,
            device,
            raw,
            width,
            height,
            landmarks,
            picker,
            next: 0,
        })
    }
    pub fn render(&mut self, frame_index: u64, features: &AudioFeatures) -> Result<BgrFrame> {
        if frame_index != self.next
            || frame_index >= features.video_frames as u64
            || Some(features.values.len()) != features.video_frames.checked_mul(2048)
        {
            return Err(invalid("Renderer requires ordered feature frame indices"));
        }
        let mut picker = self.picker.clone();
        let index = picker.next();
        let len = self.width * self.height * 3;
        self.raw
            .seek(SeekFrom::Start((index * len) as u64))
            .map_err(invalid)?;
        let mut data = vec![0; len];
        self.raw.read_exact(&mut data).map_err(invalid)?;
        let frame = BgrFrame {
            width: self.width,
            height: self.height,
            stride: self.width * 3,
            data,
        };
        let prepared = prepare_frame(&frame, &self.landmarks[index])?;
        let image = [prepared.target, prepared.masked].concat();
        let audio = audio_window(
            &features.values,
            frame_index as usize,
            0,
            features.video_frames,
        )?;
        let prediction = self
            .model
            .forward(
                Tensor::from_data(TensorData::new(image, [1, 6, 144, 144]), &self.device),
                Tensor::from_data(TensorData::new(audio, [1, 40, 1024]), &self.device),
            )
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|e| invalid(format!("Prediction read: {e:?}")))?;
        let result = compose(&frame, &self.landmarks[index], &prediction)?;
        self.picker = picker;
        self.next += 1;
        Ok(result)
    }
}
