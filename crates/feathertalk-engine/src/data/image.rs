//! Host BGR resizing, prior to device tensor conversion.
//! Arithmetic follows OpenCV 4.10 resize.cpp U8 paths: halfpixel coordinates,
//! 11-bit coefficients, specialized vertical shifts, and f32 AREA tables.
//! X clamps its fraction at borders; Y retains it while clamping source rows.
//! Frozen source fixtures verify this reference build, not every OpenCV platform.
use super::landmarks::BgrFrame;
pub(crate) fn resize_linear(frame: &BgrFrame, width: usize, height: usize, area: bool) -> BgrFrame {
    let rx = 1.0 / (width as f64 / frame.width as f64);
    let ry = 1.0 / (height as f64 / frame.height as f64);
    if (area && rx >= 1. && ry >= 1.) || (!area && rx == 2. && ry == 2.) {
        return resize_area(frame, width, height, rx, ry);
    }
    let axis = |d: usize, src: usize, dst: usize, vertical: bool| {
        let scale = 1.0 / (dst as f64 / src as f64);
        let (base, frac) = if area {
            let b = (d as f64 * scale).floor();
            let f = ((d + 1) as f64 - (b + 1.) / scale) as f32;
            (b as i32, if f <= 0. { 0. } else { f - f.floor() })
        } else {
            let v = ((d as f64 + 0.5) * scale - 0.5) as f32;
            (v.floor() as i32, v - v.floor())
        };
        if vertical {
            (base, frac)
        } else if base < 0 {
            (0, 0.)
        } else if base >= src as i32 - 1 {
            (src as i32 - 1, 0.)
        } else {
            (base, frac)
        }
    };
    let mut data = vec![0; width * height * 3];
    for y in 0..height {
        let (sy, fy) = axis(y, frame.height, height, true);
        let b0 = ((1. - fy) * 2048.).round_ties_even() as i64;
        let b1 = (fy * 2048.).round_ties_even() as i64;
        for x in 0..width {
            let (sx, fx) = axis(x, frame.width, width, false);
            let a0 = ((1. - fx) * 2048.).round_ties_even() as i64;
            let a1 = (fx * 2048.).round_ties_even() as i64;
            for c in 0..3 {
                let at = |xx: i32, yy: i32| {
                    frame.data[yy.clamp(0, frame.height as i32 - 1) as usize * frame.stride
                        + xx.clamp(0, frame.width as i32 - 1) as usize * 3
                        + c] as i64
                };
                let h0 = at(sx, sy) * a0 + at(sx + 1, sy) * a1;
                let h1 = at(sx, sy + 1) * a0 + at(sx + 1, sy + 1) * a1;
                data[(y * width + x) * 3 + c] =
                    ((((b0 * (h0 >> 4)) >> 16) + ((b1 * (h1 >> 4)) >> 16) + 2) >> 2) as u8;
            }
        }
    }
    BgrFrame {
        width,
        height,
        stride: width * 3,
        data,
    }
}

fn resize_area(frame: &BgrFrame, width: usize, height: usize, rx: f64, ry: f64) -> BgrFrame {
    let mut data = vec![0; width * height * 3];
    if (rx - rx.round()).abs() < f64::EPSILON && (ry - ry.round()).abs() < f64::EPSILON {
        let sx = rx as usize;
        let sy = ry as usize;
        let scale = 1. / (sx * sy) as f32;
        for y in 0..height {
            for x in 0..width {
                for c in 0..3 {
                    let mut sum = 0u32;
                    for dy in 0..sy {
                        for dx in 0..sx {
                            sum += frame.data[(y * sy + dy) * frame.stride + (x * sx + dx) * 3 + c]
                                as u32;
                        }
                    }
                    data[(y * width + x) * 3 + c] = if sx == 2 && sy == 2 {
                        ((sum + 2) >> 2) as u8
                    } else {
                        (sum as f32 * scale).round_ties_even().clamp(0., 255.) as u8
                    };
                }
            }
        }
    } else {
        let table = |src: usize, dst: usize, r: f64| {
            let mut entries = vec![];
            for d in 0..dst {
                let l = d as f64 * r;
                let right = l + r;
                let cell = r.min(src as f64 - l);
                let v = (right.floor() as usize).min(src - 1);
                let u = (l.ceil() as usize).min(v);
                let mut weights = vec![];
                if u as f64 - l > 0.001 {
                    weights.push((u - 1, ((u as f64 - l) / cell) as f32))
                }
                for i in u..v {
                    weights.push((i, (1. / cell) as f32));
                }
                if right - v as f64 > 0.001 {
                    weights.push((v, ((right - v as f64).min(1.).min(cell) / cell) as f32))
                }
                entries.push(weights);
            }
            entries
        };
        let xt = table(frame.width, width, rx);
        let yt = table(frame.height, height, ry);
        for y in 0..height {
            let mut sums = vec![0f32; width * 3];
            for &(sy, beta) in &yt[y] {
                for x in 0..width {
                    for c in 0..3 {
                        let mut horizontal = 0f32;
                        for &(sx, alpha) in &xt[x] {
                            horizontal += frame.data[sy * frame.stride + sx * 3 + c] as f32 * alpha;
                        }
                        sums[x * 3 + c] += horizontal * beta;
                    }
                }
            }
            for (i, value) in sums.into_iter().enumerate() {
                data[y * width * 3 + i] = value.round_ties_even().clamp(0., 255.) as u8;
            }
        }
    }
    BgrFrame {
        width,
        height,
        stride: width * 3,
        data,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use safetensors::SafeTensors;
    #[test]
    fn source_linear_preprocessing_is_exact() {
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.artifacts/reference");
        for case in ["left_top", "right_bottom", "negative_center"] {
            let bytes =
                std::fs::read(root.join(format!("landmark_coordinates_{case}.safetensors")))
                    .unwrap();
            let safe = SafeTensors::deserialize(&bytes).unwrap();
            let t = safe.tensor("crop").unwrap();
            let f = BgrFrame {
                width: t.shape()[1],
                height: t.shape()[0],
                stride: t.shape()[1] * 3,
                data: t.data().to_vec(),
            };
            let resized = resize_linear(&f, 192, 192, false);
            let expected: Vec<_> = safe
                .tensor("pfld_input")
                .unwrap()
                .data()
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect();
            let mut differences = 0;
            for y in 0..192 {
                for x in 0..192 {
                    for c in 0..3 {
                        if resized.data[(y * 192 + x) * 3 + c] as f32 / 255.
                            != expected[c * 192 * 192 + y * 192 + x]
                        {
                            differences += 1
                        }
                    }
                }
            }
            assert_eq!(differences, 0, "{case}");
        }
    }

    #[test]
    fn source_area_preprocessing_is_exact() {
        let root =
            std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.artifacts/reference");
        for case in ["portrait", "landscape", "integer_shrink", "rational_shrink"] {
            let bytes = std::fs::read(root.join(format!("detector_preprocess_{case}.safetensors")))
                .unwrap();
            let safe = SafeTensors::deserialize(&bytes).unwrap();
            let t = safe.tensor("bgr").unwrap();
            let frame = BgrFrame {
                width: t.shape()[1],
                height: t.shape()[0],
                stride: t.shape()[1] * 3,
                data: t.data().to_vec(),
            };
            let geo: Vec<_> = safe
                .tensor("geometry")
                .unwrap()
                .data()
                .chunks_exact(8)
                .map(|v| i64::from_le_bytes(v.try_into().unwrap()) as usize)
                .collect();
            let resized = resize_linear(&frame, geo[1], geo[0], true);
            let expected = safe.tensor("resized").unwrap();
            let mut differences = 0;
            for y in 0..geo[0] {
                for x in 0..geo[1] {
                    for c in 0..3 {
                        if resized.data[(y * geo[1] + x) * 3 + c]
                            != expected.data()[((y + geo[2]) * 640 + x + geo[3]) * 3 + c]
                        {
                            differences += 1
                        }
                    }
                }
            }
            assert_eq!(differences, 0, "{case}");
        }
    }
}
