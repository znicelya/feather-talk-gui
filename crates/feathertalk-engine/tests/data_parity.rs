use feathertalk_engine::data::{
    geometry::*,
    landmarks::{BgrFrame, Landmarks},
};
use safetensors::SafeTensors;
fn fixture(name: &str) -> Vec<u8> {
    std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("../../.artifacts/reference/{name}.safetensors")),
    )
    .unwrap()
}
fn floats(t: safetensors::tensor::TensorView<'_>) -> Vec<f32> {
    t.data()
        .chunks_exact(4)
        .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
        .collect()
}
#[test]
fn geometry_matches_opencv() {
    let bytes = fixture("character_training");
    let s = SafeTensors::deserialize(&bytes).unwrap();
    let frames = s.tensor("pair_frames_bgr").unwrap();
    let lm = s.tensor("pair_landmarks").unwrap();
    let points: Vec<i32> = lm
        .data()
        .chunks_exact(4)
        .map(|x| i32::from_le_bytes(x.try_into().unwrap()))
        .collect();
    let expected = floats(s.tensor("labels").unwrap());
    let masks = floats(s.tensor("masks").unwrap());
    for i in 0..2 {
        let f = BgrFrame {
            width: 240,
            height: 220,
            stride: 720,
            data: frames.data()[i * 220 * 720..(i + 1) * 220 * 720].to_vec(),
        };
        let landmarks = Landmarks {
            points: std::array::from_fn(|n| [points[i * 220 + n * 2], points[i * 220 + n * 2 + 1]]),
        };
        let inner = prepare_frame(&f, &landmarks).unwrap();
        assert_eq!(
            inner.target,
            expected[i * 3 * 144 * 144..(i + 1) * 3 * 144 * 144]
        );
        assert_eq!(inner.mask, masks[i * 144 * 144..(i + 1) * 144 * 144]);
        let image = floats(s.tensor("image").unwrap());
        assert_eq!(
            inner.masked,
            image[(i * 6 + 3) * 144 * 144..(i * 6 + 6) * 144 * 144]
        );
    }
}
#[test]
fn audio_edges_match() {
    let features: Vec<f32> = (0..24 * 2048).map(|n| n as f32 / 1024.).collect();
    for (case, index, start, end) in [
        ("audio_first", 0, 0, 24),
        ("audio_last", 23, 0, 24),
        ("audio_clip_first", 8, 8, 14),
        ("audio_clip_last", 13, 8, 14),
    ] {
        let bytes = fixture(case);
        let s = SafeTensors::deserialize(&bytes).unwrap();
        assert_eq!(
            audio_window(&features, index, start, end).unwrap(),
            floats(s.tensor("tokens").unwrap())
        );
    }
}
#[test]
fn production_detector_blob_matches_source() {
    for case in ["portrait", "landscape", "integer_shrink", "rational_shrink"] {
        let bytes = fixture(&format!("detector_preprocess_{case}"));
        let s = SafeTensors::deserialize(&bytes).unwrap();
        let t = s.tensor("bgr").unwrap();
        let frame = BgrFrame {
            width: t.shape()[1],
            height: t.shape()[0],
            stride: t.shape()[1] * 3,
            data: t.data().to_vec(),
        };
        let (blob, geometry) = feathertalk_engine::models::scrfd::detector_input(&frame).unwrap();
        let expected: Vec<usize> = s
            .tensor("geometry")
            .unwrap()
            .data()
            .chunks_exact(8)
            .map(|b| i64::from_le_bytes(b.try_into().unwrap()) as usize)
            .collect();
        assert_eq!(geometry.as_slice(), expected);
        assert_eq!(blob, floats(s.tensor("blob").unwrap()));
    }
}
#[test]
fn geometry_odd_sizes_and_area_enlargement() {
    use sha2::{Digest, Sha256};
    let manifest: serde_json::Value =
        serde_json::from_str(include_str!("../../../tests/reference/task8-manifest.json")).unwrap();
    for size in [71, 100, 151, 153, 159, 161, 213] {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
            "../../.artifacts/task8-oracles/geometry_{size}.safetensors"
        ));
        let bytes = std::fs::read(path).unwrap();
        assert_eq!(
            format!("{:x}", Sha256::digest(&bytes)),
            manifest["files"][format!("geometry_{size}.safetensors")]
                .as_str()
                .unwrap()
        );
        let s = SafeTensors::deserialize(&bytes).unwrap();
        let t = s.tensor("frame").unwrap();
        let frame = BgrFrame {
            width: t.shape()[1],
            height: t.shape()[0],
            stride: t.shape()[1] * 3,
            data: t.data().to_vec(),
        };
        let lm = s.tensor("landmarks").unwrap();
        let points: Vec<i32> = lm
            .data()
            .chunks_exact(4)
            .map(|b| i32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let lm = Landmarks {
            points: std::array::from_fn(|i| [points[i * 2], points[i * 2 + 1]]),
        };
        let actual = prepare_frame(&frame, &lm).unwrap();
        for (name, values) in [
            ("target", actual.target),
            ("masked", actual.masked),
            ("mask", actual.mask),
        ] {
            let expected = floats(s.tensor(name).unwrap());
            let differences = values.iter().zip(&expected).filter(|(a, b)| a != b).count();
            assert_eq!(differences, 0, "size={size} {name}");
        }
    }
}
#[test]
fn invalid_source_crop_is_rejected_without_padding() {
    let frame = BgrFrame {
        width: 20,
        height: 20,
        stride: 60,
        data: vec![0; 1200],
    };
    let mut lm = Landmarks {
        points: [[0; 2]; 110],
    };
    lm.points[1][0] = -2;
    lm.points[31][0] = 10;
    assert!(prepare_frame(&frame, &lm).is_err());
    lm.points[1][0] = 15;
    lm.points[31][0] = 25;
    assert!(prepare_frame(&frame, &lm).is_err());
}
