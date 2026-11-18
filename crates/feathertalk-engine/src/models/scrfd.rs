use super::generated::scrfd::Model;
use crate::verified_assets::{VerifiedAssets, error};
use burn::module::Module;
use burn::tensor::{Bytes, Device, Tensor};
use burn_store::{BurnpackStore, ModuleSnapshot};
use feathertalk_core::Result;
use std::path::Path;
pub struct Scrfd {
    model: Model,
}
impl Scrfd {
    pub fn load(root: &Path, device: &Device) -> Result<Self> {
        let a = VerifiedAssets::load(root)?;
        if a.manifest.identity.kind != "scrfd_2_5g"
            || a.manifest.identity.architecture_version != "scrfd_2_5g-source-v1"
            || a.manifest.identity.source_sha256
                != "32d20c77b9e2dc1d07e94c2ab9d25bdd5cd05eddbe0b46e7b38e7a1eca22e99a"
        {
            return Err(error("SCRFD identity mismatch"));
        }
        let config: serde_json::Value = serde_json::from_slice(
            a.files
                .get("config.json")
                .ok_or_else(|| error("Missing SCRFD config"))?,
        )
        .map_err(error)?;
        if config["architecture_version"] != a.manifest.identity.architecture_version
            || config["source_sha256"] != a.manifest.identity.source_sha256
            || config["configuration"]
                != serde_json::json!({"graph":"architecture.onnx","input":[1,3,640,640],"opsets":{"":12},"initializer_names":"original ONNX"})
        {
            return Err(error("SCRFD config mismatch"));
        }
        let proof: serde_json::Value = serde_json::from_slice(
            a.files
                .get("generation.json")
                .ok_or_else(|| error("Missing generation provenance"))?,
        )
        .map_err(error)?;
        if proof["source_sha256"] != a.manifest.identity.source_sha256
            || proof["generator"] != "burn-onnx=0.22.0"
            || proof["transform_version"] != "scrfd-fixed640-redundant-ceil-v1"
            || proof["derived_sha256"]
                != "ca6ee50c171e3ee5e37f05939dfea0bf8d864f9fab4562fefd8316bd9a5b67bf"
        {
            return Err(error("SCRFD generation provenance mismatch"));
        }
        if a.manifest.hashes.get(Path::new("architecture.onnx"))
            != Some(&a.manifest.identity.source_sha256)
            || a.manifest
                .hashes
                .get(Path::new("generated.onnx"))
                .map(String::as_str)
                != Some("ca6ee50c171e3ee5e37f05939dfea0bf8d864f9fab4562fefd8316bd9a5b67bf")
        {
            return Err(error("SCRFD graph identity mismatch"));
        }
        let bytes = a
            .files
            .get("model.bpk")
            .ok_or_else(|| error("Missing generated weights"))?;
        let mut model = Model::new(device);
        let mut store = BurnpackStore::from_bytes(Some(Bytes::from_bytes_vec(bytes.clone())));
        let loaded = model.load_from(&mut store).map_err(error)?;
        if !loaded.missing.is_empty()
            || !loaded.unused.is_empty()
            || !loaded.errors.is_empty()
            || !loaded.skipped.is_empty()
        {
            return Err(error("Incomplete SCRFD state"));
        }
        Ok(Self { model })
    }
    pub fn forward(&self, input: Tensor<4>) -> Vec<Tensor<3>> {
        let (a, b, c, d, e, f, g, h, i) = self.model.forward(input);
        vec![a, b, c, d, e, f, g, h, i]
    }
}

impl Scrfd {
    pub fn detect(
        &self,
        frame: &crate::data::landmarks::BgrFrame,
    ) -> Result<Vec<crate::data::landmarks::Detection>> {
        let (values, [h, w, py, px]) = detector_input(frame)?;
        let device = self.model.devices()[0].clone();
        let raw = self.forward(Tensor::from_data(
            burn::tensor::TensorData::new(values, [1, 3, 640, 640]),
            &device,
        ));
        let raw: Vec<Vec<f32>> = raw
            .into_iter()
            .map(|t| t.into_data().try_to_vec().map_err(error))
            .collect::<Result<_>>()?;
        return decode_outputs(&raw, frame.width, frame.height, w, h, px, py);
    }
}
/// The exact host preprocessing used by detect, exposed for source parity checks.
pub fn detector_input(frame: &crate::data::landmarks::BgrFrame) -> Result<(Vec<f32>, [usize; 4])> {
    if frame.width == 0
        || frame.height == 0
        || frame.stride < frame.width.saturating_mul(3)
        || frame.data.len() < frame.height.saturating_mul(frame.stride)
    {
        return Err(error("Invalid detector frame"));
    }
    let ratio = frame.height as f64 / frame.width as f64;
    let (w, h) = if ratio > 1. {
        ((640. / ratio) as usize, 640)
    } else if ratio < 1. {
        (640, (640. * ratio) as usize + 1)
    } else {
        (640, 640)
    };
    if w == 0 || h == 0 || w > 640 || h > 640 {
        return Err(error(
            "Frame aspect ratio cannot be resized to detector input",
        ));
    }
    let resized = crate::data::image::resize_linear(frame, w, h, true);
    let (px, py) = ((640 - w) / 2, (640 - h) / 2);
    let mut values = vec![-127.5 / 128.; 3 * 640 * 640];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                values[c * 640 * 640 + (y + py) * 640 + x + px] =
                    (resized.data[y * resized.stride + x * 3 + 2 - c] as f32 - 127.5) / 128.;
            }
        }
    }
    Ok((values, [h, w, py, px]))
}
pub fn decode_outputs(
    raw: &[Vec<f32>],
    frame_width: usize,
    frame_height: usize,
    w: usize,
    h: usize,
    px: usize,
    py: usize,
) -> Result<Vec<crate::data::landmarks::Detection>> {
    use crate::data::landmarks::*;
    if raw.len() != 9 || w == 0 || h == 0 {
        return Err(error("Invalid SCRFD output geometry"));
    }
    for level in 0..3 {
        let count = (640usize / [8, 16, 32][level]).pow(2) * 2;
        for (group, channels) in [1, 4, 10].into_iter().enumerate() {
            let output = &raw[level + group * 3];
            if output.len() != count * channels || output.iter().any(|v| !v.is_finite()) {
                return Err(error("Invalid SCRFD output tensor"));
            }
        }
    }
    let mut out = vec![];
    for (level, stride) in [8, 16, 32].into_iter().enumerate() {
        let side = 640 / stride;
        for index in 0..side * side * 2 {
            let score = raw[level][index];
            if score <= 0.1 {
                continue;
            }
            let cx = (index / 2 % side * stride) as f32;
            let cy = (index / 2 / side * stride) as f32;
            let dist = &raw[level + 3][index * 4..index * 4 + 4];
            let s = stride as f32;
            let rx = frame_width as f32 / w as f32;
            let ry = frame_height as f32 / h as f32;
            let bbox = Rect {
                x: (cx - dist[0] * s - px as f32) * rx,
                y: (cy - dist[1] * s - py as f32) * ry,
                width: ((cx + dist[2] * s) - (cx - dist[0] * s)) * rx,
                height: ((cy + dist[3] * s) - (cy - dist[1] * s)) * ry,
            };
            let mut keypoints = [[0.; 2]; 5];
            for k in 0..5 {
                keypoints[k] = [
                    (cx + raw[level + 6][index * 10 + k * 2] * s - px as f32) * rx,
                    (cy + raw[level + 6][index * 10 + k * 2 + 1] * s - py as f32) * ry,
                ];
            }
            out.push(Detection {
                bbox,
                score,
                keypoints,
            });
        }
    }
    out.sort_by(|a, b| b.score.total_cmp(&a.score));
    let mut kept: Vec<Detection> = vec![];
    for d in out {
        if kept.iter().all(|k| {
            let a = d.bbox;
            let b = k.bbox;
            let (ax, ay, aw, ah) = (a.x as f64, a.y as f64, a.width as f64, a.height as f64);
            let (bx, by, bw, bh) = (b.x as f64, b.y as f64, b.width as f64, b.height as f64);
            let intersection = ((ax + aw).min(bx + bw) - ax.max(bx)).max(0.)
                * ((ay + ah).min(by + bh) - ay.max(by)).max(0.);
            let area = aw * ah + bw * bh - intersection;
            let overlap = if area <= f64::EPSILON {
                0.
            } else {
                intersection / area
            };
            (overlap as f32) <= 0.5
        }) {
            kept.push(d)
        }
    }
    Ok(kept)
}
