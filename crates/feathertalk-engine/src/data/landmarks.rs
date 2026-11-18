//! Source-compatible crop geometry and local-to-global landmark conversion.
use crate::verified_assets::{VerifiedAssets, error};
use feathertalk_core::*;
use sha2::{Digest, Sha256};
use std::path::Path;
#[derive(Clone, Debug)]
pub struct BgrFrame {
    pub width: usize,
    pub height: usize,
    pub stride: usize,
    pub data: Vec<u8>,
}
#[derive(Clone, Copy, Debug)]
pub struct Rect {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}
#[derive(Clone, Debug)]
pub struct Detection {
    pub bbox: Rect,
    pub score: f32,
    pub keypoints: [[f32; 2]; 5],
}
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Landmarks {
    pub points: [[i32; 2]; 110],
}
pub fn no_face(frame: &BgrFrame) -> AppError {
    let mut e = AppError::new(
        ErrorCode::NoFace,
        "face-landmarks",
        "No valid face was detected in the frame",
    );
    e.evidence.push(format!(
        "frame={}x{} stride={}",
        frame.width, frame.height, frame.stride
    ));
    e.actions
        .push("Choose a frame with a visible, unobstructed face".into());
    e
}
pub fn crop_face(frame: &BgrFrame, detections: &[Detection]) -> Result<(BgrFrame, [i32; 2])> {
    if frame.width == 0
        || frame.height == 0
        || frame.stride < frame.width.saturating_mul(3)
        || frame.data.len() < frame.stride.saturating_mul(frame.height)
    {
        return Err(AppError::new(
            ErrorCode::InvalidConfig,
            "face-landmarks",
            "Invalid BGR frame layout",
        ));
    }
    for detection in detections {
        let b = detection.bbox;
        if ![b.x, b.y, b.width, b.height]
            .iter()
            .all(|v| v.is_finite() && v.abs() < i32::MAX as f32 / 4.)
            || b.width <= 0.
            || b.height <= 0.
        {
            continue;
        }
        let (x1, y1, x2, y2) = (
            b.x as i64,
            b.y as i64,
            (b.x + b.width) as i64,
            (b.y + b.height) as i64,
        );
        let size = (((x2 - x1).max(y2 - y1)) as f64 * 1.05) as i64;
        if size <= 0 {
            continue;
        }
        let origin = [
            (x1 + x2).div_euclid(2) - size / 2,
            (y1 + y2).div_euclid(2) - size / 2,
        ];
        if origin[0] >= frame.width as i64
            || origin[1] >= frame.height as i64
            || origin[0] + size <= 0
            || origin[1] + size <= 0
        {
            continue;
        }
        let s = size as usize;
        let bytes = s
            .checked_mul(s)
            .and_then(|n| n.checked_mul(3))
            .ok_or_else(|| error("Face crop dimensions overflow"))?;
        let mut data = Vec::new();
        data.try_reserve_exact(bytes).map_err(|_| {
            AppError::new(
                ErrorCode::OutOfMemory,
                "face-landmarks",
                "Unable to allocate padded face crop",
            )
        })?;
        data.resize(bytes, 0);
        for y in 0..s {
            for x in 0..s {
                let sx = origin[0] + x as i64;
                let sy = origin[1] + y as i64;
                if sx >= 0 && sx < frame.width as i64 && sy >= 0 && sy < frame.height as i64 {
                    let src = sy as usize * frame.stride + sx as usize * 3;
                    data[(y * s + x) * 3..(y * s + x) * 3 + 3]
                        .copy_from_slice(&frame.data[src..src + 3]);
                }
            }
        }
        return Ok((
            BgrFrame {
                width: s,
                height: s,
                stride: s * 3,
                data,
            },
            [origin[0] as i32, origin[1] as i32],
        ));
    }
    Err(no_face(frame))
}
pub fn map_landmarks(
    raw: &[f32; 220],
    mean: &[f32; 220],
    crop: &BgrFrame,
    origin: [i32; 2],
) -> Result<Landmarks> {
    let mut points = [[0; 2]; 110];
    for n in 0..110 {
        for axis in 0..2 {
            let v = (raw[n * 2 + axis] + mean[n * 2 + axis])
                * if axis == 0 { crop.width } else { crop.height } as f32;
            if !v.is_finite() || v.abs() > i32::MAX as f32 / 2. {
                return Err(error("Invalid predicted landmark"));
            }
            points[n][axis] = (v as i32)
                .checked_add(origin[axis])
                .ok_or_else(|| error("Landmark overflow"))?;
        }
    }
    Ok(Landmarks { points })
}
pub fn load_mean_face(root: &Path) -> Result<[f32; 220]> {
    let a = VerifiedAssets::load(root)?;
    if a.manifest.identity.kind != "mean_face"
        || a.manifest.identity.source_sha256
            != "40bed2653ae861e9a24b4766c3d085986b7c1245a27373913a9a5b1a6ce069ef"
    {
        return Err(error("Mean face identity mismatch"));
    }
    let config: serde_json::Value = serde_json::from_slice(
        a.files
            .get("config.json")
            .ok_or_else(|| error("Missing mean face config"))?,
    )
    .map_err(error)?;
    if a.manifest.identity.architecture_version != "mean_face-source-v1"
        || config["source_sha256"] != a.manifest.identity.source_sha256
        || config["architecture_version"] != a.manifest.identity.architecture_version
        || config["configuration"] != serde_json::json!({"count":220,"shape":[110,2]})
    {
        return Err(error("Mean face config mismatch"));
    }
    let bytes = a
        .files
        .get("model.safetensors")
        .ok_or_else(|| error("Missing mean face"))?;
    if format!("{:x}", Sha256::digest(bytes))
        != "298f0167a6240ea5c058d24b051b2ba816e07f655d91e781268a7f4682e42e81"
    {
        return Err(error("Untrusted mean face values"));
    }
    let safe = safetensors::SafeTensors::deserialize(bytes).map_err(error)?;
    let t = safe.tensor("mean_face").map_err(error)?;
    if t.shape() != [220] || t.dtype() != safetensors::Dtype::F32 {
        return Err(error("Invalid mean face shape"));
    }
    let values: Vec<_> = t
        .data()
        .chunks_exact(4)
        .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
        .collect();
    values
        .try_into()
        .map_err(|_| error("Invalid mean face length"))
}

pub fn locate_face(
    detector: &crate::models::scrfd::Scrfd,
    landmarks: &crate::models::pfld::PfldGhostOne,
    mean_face: &[f32; 220],
    frame: &BgrFrame,
) -> Result<Landmarks> {
    let mut digest = Sha256::new();
    for value in mean_face {
        digest.update(value.to_le_bytes());
    }
    if format!("{:x}", digest.finalize())
        != "f73ea6c54a8469030f9495307fb5edded2eeeadf448651ef9285293666c89740"
    {
        return Err(error("Untrusted caller mean face values"));
    }
    let detections = detector.detect(frame)?;
    let (crop, origin) = crop_face(frame, &detections)?;
    let input = super::image::resize_linear(&crop, 192, 192, false);
    let mut values = vec![0.; 3 * 192 * 192];
    for y in 0..192 {
        for x in 0..192 {
            for c in 0..3 {
                values[c * 192 * 192 + y * 192 + x] =
                    input.data[y * input.stride + x * 3 + c] as f32 / 255.;
            }
        }
    }
    let tensor = burn::tensor::Tensor::from_data(
        burn::tensor::TensorData::new(values, [1, 3, 192, 192]),
        landmarks.device(),
    );
    let raw: Vec<f32> = landmarks
        .forward(tensor)
        .into_data()
        .try_to_vec()
        .map_err(error)?;
    let raw: [f32; 220] = raw
        .try_into()
        .map_err(|_| error("Unexpected PFLD output"))?;
    map_landmarks(&raw, mean_face, &crop, origin)
}
