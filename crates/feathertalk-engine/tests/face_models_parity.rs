#[allow(dead_code)]
#[path = "../src/reference.rs"]
mod reference;
use burn::tensor::Device;
use feathertalk_core::*;
use feathertalk_engine::{
    models::{pfld::PfldGhostOne, scrfd::Scrfd},
    *,
};
fn device() -> Device {
    let b = std::env::var("FT_TEST_BACKEND").unwrap_or("flex".into());
    let backend = match b.as_str() {
        "flex" => BackendKind::Flex,
        "vulkan" => BackendKind::Vulkan,
        "cuda" => BackendKind::Cuda,
        _ => panic!("backend"),
    };
    let d = open_device(&DeviceRequest {
        backend,
        device_id: format!("{b}:0"),
    })
    .unwrap();
    eprintln!("Face actual device {:?}", d.descriptor);
    d.device
}
fn root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}
fn compare(r: &reference::Reference, key: &str, shape: &[usize], values: &[f32]) {
    assert_eq!(shape, r.values[key].0);
    let expected = &r.values[key].1;
    assert_eq!(values.len(), expected.len());
    let t = &r.case.tolerance;
    let mut count = 0;
    let mut max = 0f32;
    for (a, b) in values.iter().zip(expected) {
        max = max.max((a - b).abs());
        if !a.is_finite() || (*a as f64 - *b as f64).abs() > t.atol + t.rtol * (*b as f64).abs() {
            count += 1;
        }
    }
    eprintln!("{key} max_abs={max} failures={count}");
    assert_eq!(count, 0, "{key}");
}
#[test]
fn scrfd_levels_and_nms_match() {
    let d = device();
    let r = reference::Reference::load(&root().join(".artifacts/reference"), "scrfd_2_5g").unwrap();
    let m = Scrfd::load(&root().join(".artifacts/assets/scrfd_2_5g"), &d).unwrap();
    for (i, t) in m.forward(r.tensor("input.0", &d)).into_iter().enumerate() {
        compare(
            &r,
            &format!("out{i}"),
            &t.dims(),
            &t.into_data().try_to_vec::<f32>().unwrap(),
        );
    }
}
#[test]
fn pfld_raw_outputs_match() {
    let d = device();
    let r =
        reference::Reference::load(&root().join(".artifacts/reference"), "pfld_ghost_one").unwrap();
    let m = PfldGhostOne::load(&root().join(".artifacts/assets/pfld_ghost_one"), &d).unwrap();
    assert_eq!(m.accounted_counters, 351);
    assert_eq!(m.inactive_parameters.len(), 4);
    let t = m.forward_observed(r.tensor("input.0", &d), |k, t| {
        compare(
            &r,
            &format!("layer.{k}"),
            &t.dims(),
            &t.into_data().try_to_vec::<f32>().unwrap(),
        )
    });
    compare(
        &r,
        "output",
        &t.dims(),
        &t.into_data().try_to_vec::<f32>().unwrap(),
    );
}

#[test]
fn mean_face_and_border_coordinates_match() {
    use feathertalk_engine::data::landmarks::*;
    use safetensors::SafeTensors;
    let mean = load_mean_face(&root().join(".artifacts/assets/mean_face")).unwrap();
    for case in ["left_top", "right_bottom", "negative_center"] {
        let bytes = std::fs::read(root().join(format!(
            ".artifacts/reference/landmark_coordinates_{case}.safetensors"
        )))
        .unwrap();
        let safe = SafeTensors::deserialize(&bytes).unwrap();
        let f = safe.tensor("frame").unwrap();
        let shape = f.shape();
        let frame = BgrFrame {
            width: shape[1],
            height: shape[0],
            stride: shape[1] * 3,
            data: f.data().to_vec(),
        };
        let floats = |key: &str| -> Vec<f32> {
            safe.tensor(key)
                .unwrap()
                .data()
                .chunks_exact(4)
                .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                .collect()
        };
        let boxes = floats("boxes_xywh");
        let detections: Vec<_> = boxes
            .chunks_exact(4)
            .map(|b| Detection {
                bbox: Rect {
                    x: b[0],
                    y: b[1],
                    width: b[2],
                    height: b[3],
                },
                score: 0.9,
                keypoints: [[0.; 2]; 5],
            })
            .collect();
        let (crop, origin) = crop_face(&frame, &detections).unwrap();
        assert_eq!(crop.data, safe.tensor("crop").unwrap().data());
        let raw: [f32; 220] = floats("raw").try_into().unwrap();
        let actual = map_landmarks(&raw, &mean, &crop, origin).unwrap();
        let expected: Vec<_> = safe
            .tensor("points")
            .unwrap()
            .data()
            .chunks_exact(4)
            .map(|v| i32::from_le_bytes(v.try_into().unwrap()))
            .collect();
        assert_eq!(
            actual.points.iter().flatten().copied().collect::<Vec<_>>(),
            expected
        );
    }
}
#[test]
fn no_face_is_actionable() {
    use feathertalk_engine::data::landmarks::*;
    let frame = BgrFrame {
        width: 2,
        height: 2,
        stride: 6,
        data: vec![0; 12],
    };
    let e = crop_face(&frame, &[]).unwrap_err();
    assert_eq!(e.code, ErrorCode::NoFace);
    assert!(!e.actions.is_empty());
    assert!(!e.evidence.is_empty());
    assert_eq!(e.stage, "face-landmarks");
}

#[test]
fn stride_anchor_threshold_and_nms_rules() {
    use feathertalk_engine::models::scrfd::decode_outputs;
    let mut raw = vec![];
    for channels in [1, 4, 10] {
        for stride in [8usize, 16, 32] {
            raw.push(vec![0.; (640 / stride).pow(2) * 2 * channels]);
        }
    }
    // Duplicated consecutive anchors: score tie keeps the first offset prediction.
    raw[0][2] = 0.9;
    raw[0][3] = 0.9;
    raw[0][4] = 0.1;
    for i in [2, 3, 4] {
        raw[3][i * 4..i * 4 + 4].fill(1.);
    }
    raw[6][2 * 10] = 0.5;
    raw[6][3 * 10] = 1.;
    raw[1][2] = 0.8;
    raw[4][8..12].fill(0.25);
    raw[2][2] = 0.7;
    raw[5][8..12].fill(0.25);
    let kept = decode_outputs(&raw, 640, 640, 640, 640, 0, 0).unwrap();
    assert_eq!(kept.len(), 3);
    assert_eq!(kept[0].bbox.x, 0.);
    assert_eq!(kept[0].keypoints[0][0], 12.);
    assert_eq!(kept[1].bbox.x, 12.);
    assert_eq!(kept[2].bbox.x, 24.);
    // Equal IoU .5 survives; a larger overlap suppresses.
    raw[0].fill(0.);
    raw[1].fill(0.);
    raw[2].fill(0.);
    raw[0][0] = 0.9;
    raw[0][1] = 0.8;
    raw[3][0..4].copy_from_slice(&[0., 0., 2., 2.]);
    raw[3][4..8].copy_from_slice(&[0., 0., 1., 2.]);
    assert_eq!(
        decode_outputs(&raw, 640, 640, 640, 640, 0, 0)
            .unwrap()
            .len(),
        2
    );
    raw[3][6] = 1.1;
    assert_eq!(
        decode_outputs(&raw, 640, 640, 640, 640, 0, 0)
            .unwrap()
            .len(),
        1
    );
}
#[test]
fn mean_face_tampering_is_rejected() {
    use feathertalk_engine::data::landmarks::load_mean_face;
    use sha2::{Digest, Sha256};
    let source = root().join(".artifacts/assets/mean_face");
    let temp = std::env::temp_dir().join(format!("feathertalk-mean-{}", std::process::id()));
    std::fs::create_dir_all(&temp).unwrap();
    for name in ["manifest.json", "config.json", "model.safetensors"] {
        std::fs::copy(source.join(name), temp.join(name)).unwrap();
    }
    let mut bytes = std::fs::read(temp.join("model.safetensors")).unwrap();
    let last = bytes.len() - 4;
    bytes[last..].copy_from_slice(&0.9f32.to_le_bytes());
    std::fs::write(temp.join("model.safetensors"), &bytes).unwrap();
    let mut manifest: AssetManifest =
        serde_json::from_slice(&std::fs::read(temp.join("manifest.json")).unwrap()).unwrap();
    let hash = format!("{:x}", Sha256::digest(&bytes));
    manifest.identity.weights_sha256 = hash.clone();
    manifest.hashes.insert("model.safetensors".into(), hash);
    std::fs::write(
        temp.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    assert!(load_mean_face(&temp).is_err());
    std::fs::remove_dir_all(temp).unwrap();
}

#[test]
fn malformed_generated_weights_are_rejected() {
    use sha2::{Digest, Sha256};
    let source = root().join(".artifacts/assets/scrfd_2_5g");
    let temp = std::env::temp_dir().join(format!("feathertalk-scrfd-{}", std::process::id()));
    std::fs::create_dir_all(&temp).unwrap();
    let mut manifest: AssetManifest =
        serde_json::from_slice(&std::fs::read(source.join("manifest.json")).unwrap()).unwrap();
    for name in &manifest.relative_files {
        std::fs::copy(source.join(name), temp.join(name)).unwrap();
    }
    let bytes = b"invalid Burnpack";
    std::fs::write(temp.join("model.bpk"), bytes).unwrap();
    let hash = format!("{:x}", Sha256::digest(bytes));
    manifest.identity.weights_sha256 = hash.clone();
    manifest.hashes.insert("model.bpk".into(), hash);
    manifest
        .sizes
        .insert("model.bpk".into(), bytes.len() as u64);
    std::fs::write(
        temp.join("manifest.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    let e = Scrfd::load(&temp, &Device::flex())
        .err()
        .expect("malformed container must fail");
    assert_eq!(e.code, ErrorCode::ResourceCorrupt);
    std::fs::remove_dir_all(temp).unwrap();
}
