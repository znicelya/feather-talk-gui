//! Training geometry. Invalid source crops are rejected rather than padded or clamped.
use super::{
    image::resize_linear,
    landmarks::{BgrFrame, Landmarks},
};
use feathertalk_core::*;
pub(crate) fn invalid(message: impl ToString) -> AppError {
    AppError::new(ErrorCode::InvalidConfig, "dataset", message.to_string())
}
pub struct PreparedFrame {
    pub target: Vec<f32>,
    pub masked: Vec<f32>,
    pub mask: Vec<f32>,
}
pub fn prepare_frame(frame: &BgrFrame, landmarks: &Landmarks) -> Result<PreparedFrame> {
    let x = landmarks.points[1][0] as i64;
    let y = landmarks.points[52][1] as i64;
    let size = landmarks.points[31][0] as i64 - x;
    if size <= 0
        || x < 0
        || y < 0
        || x + size > frame.width as i64
        || y + size > frame.height as i64
        || frame.stride < frame.width.saturating_mul(3)
        || frame.data.len() < frame.height.saturating_mul(frame.stride)
    {
        return Err(invalid(
            "Landmark crop lies outside frame or has invalid dimensions",
        ));
    }
    let size = size as usize;
    let mut crop = Vec::with_capacity(size * size * 3);
    for row in y as usize..y as usize + size {
        let start = row * frame.stride + x as usize * 3;
        crop.extend_from_slice(&frame.data[start..start + size * 3]);
    }
    let crop = resize_linear(
        &BgrFrame {
            width: size,
            height: size,
            stride: size * 3,
            data: crop,
        },
        152,
        152,
        true,
    );
    let mut target = vec![0.; 3 * 144 * 144];
    for c in 0..3 {
        for row in 0..144 {
            for col in 0..144 {
                target[c * 144 * 144 + row * 144 + col] =
                    crop.data[((row + 4) * 152 + col + 4) * 3 + c] as f32 / 255.;
            }
        }
    }
    let mut masked = target.clone();
    for c in 0..3 {
        for row in 4..134 {
            masked[c * 144 * 144 + row * 144 + 4..c * 144 * 144 + row * 144 + 139].fill(0.);
        }
    }
    let scale = 152f32 / size as f32;
    let mut lo = [f32::INFINITY; 2];
    let mut hi = [f32::NEG_INFINITY; 2];
    for p in &landmarks.points[90..110] {
        for a in 0..2 {
            let v = (p[a] as f32 - [x, y][a] as f32) * scale - 4.;
            lo[a] = lo[a].min(v);
            hi[a] = hi[a].max(v);
        }
    }
    let mut bounds = [[0usize; 2]; 2];
    for a in 0..2 {
        // NumPy 2.x preserves f32 scalar arithmetic here, including rounding ties.
        let center = (lo[a] + hi[a]) / 2.;
        let extent = (hi[a] - lo[a] + 3.6) * [1.45, 1.75][a];
        let extent = extent.max([46.8, 32.4][a]);
        let start = (center - extent / 2.).round_ties_even().clamp(0., 143.) as usize;
        let end = (center + extent / 2.)
            .round_ties_even()
            .clamp((start + 1) as f32, 144.) as usize;
        bounds[a] = [start, end];
    }
    let mut mask = vec![0.; 144 * 144];
    for row in bounds[1][0]..bounds[1][1] {
        mask[row * 144 + bounds[0][0]..row * 144 + bounds[0][1]].fill(1.);
    }
    Ok(PreparedFrame {
        target,
        masked,
        mask,
    })
}
pub fn audio_window(features: &[f32], index: usize, start: usize, end: usize) -> Result<Vec<f32>> {
    if features.len() % 2048 != 0 || start > index || index >= end || end > features.len() / 2048 {
        return Err(invalid("Invalid audio window bounds"));
    }
    let mut result = vec![0.; 40 * 1024];
    for offset in 0..20 {
        let frame = index as i128 + offset as i128 - 10;
        if frame >= start as i128 && frame < end as i128 {
            result[offset * 2048..(offset + 1) * 2048]
                .copy_from_slice(&features[frame as usize * 2048..(frame as usize + 1) * 2048]);
        }
    }
    Ok(result)
}
