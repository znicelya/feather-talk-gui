#[allow(dead_code)]
#[path = "../src/reference.rs"]
mod reference;
use burn::tensor::Device;
use feathertalk_core::*;
use feathertalk_engine::{data::audio::*, models::feather_hubert::FeatherHubert, *};
use std::path::PathBuf;

fn roots() -> (PathBuf, PathBuf) {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    (
        std::env::var_os("FT_ASSETS_DIR")
            .map(PathBuf::from)
            .unwrap_or(root.join(".artifacts/assets"))
            .join("feather_hubert"),
        std::env::var_os("FT_REFERENCE_DIR")
            .map(PathBuf::from)
            .unwrap_or(root.join(".artifacts/reference")),
    )
}
fn device() -> Device {
    let backend = std::env::var("FT_TEST_BACKEND").unwrap_or("flex".into());
    let kind = match backend.as_str() {
        "flex" => BackendKind::Flex,
        "vulkan" => BackendKind::Vulkan,
        "cuda" => BackendKind::Cuda,
        _ => panic!("unknown backend"),
    };
    let d = open_device(&DeviceRequest {
        backend: kind,
        device_id: format!("{backend}:0"),
    })
    .unwrap();
    eprintln!("HuBERT actual device: {:?}", d.descriptor);
    d.device
}
fn compare(r: &reference::Reference, key: &str, shape: &[usize], actual: &[f32]) {
    assert_eq!(shape, r.values[key].0, "{key} shape");
    let expected = &r.values[key].1;
    assert_eq!(actual.len(), expected.len());
    let t = r
        .case
        .tensor_tolerances
        .get(key)
        .unwrap_or(&r.case.tolerance);
    let failures: Vec<_> = actual
        .iter()
        .zip(expected)
        .enumerate()
        .filter(|(_, (a, b))| {
            !a.is_finite() || (**a as f64 - **b as f64).abs() > t.atol + t.rtol * (**b as f64).abs()
        })
        .take(5)
        .collect();
    eprintln!(
        "{key}: max_abs={} failures={failures:?}",
        actual
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    );
    assert!(failures.is_empty(), "{key}: {failures:?}");
}
#[test]
fn current_checkpoint_layers_and_extraction_match() {
    let (assets, references) = roots();
    let d = device();
    let model = FeatherHubert::load(&assets, &d).unwrap();
    assert_eq!(model.configuration().channels, 256);
    assert_eq!(model.configuration().num_blocks, 8);
    let r = reference::Reference::load(&references, "feather_hubert").unwrap();
    let expected_taps: std::collections::BTreeSet<_> = [
        "layer.frontend.layers.0",
        "layer.frontend",
        "layer.encoder.0",
        "layer.encoder",
        "layer.final_norm",
        "layer.proj",
    ]
    .into_iter()
    .map(str::to_string)
    .collect();
    assert_eq!(
        r.values
            .keys()
            .filter(|k| k.starts_with("layer."))
            .cloned()
            .collect::<std::collections::BTreeSet<_>>(),
        expected_taps
    );
    let mut seen = std::collections::BTreeSet::new();
    let out = model.forward_observed(r.tensor("input.0", &d), |key, tensor| {
        let key = format!("layer.{key}");
        if r.values.contains_key(&key) {
            assert!(seen.insert(key.clone()), "Duplicate tap {key}");
            let shape = tensor.dims();
            compare(
                &r,
                &key,
                &shape,
                &tensor.into_data().try_to_vec::<f32>().unwrap(),
            );
        }
    });
    assert_eq!(seen, expected_taps);
    compare(
        &r,
        "output",
        &out.dims(),
        &out.into_data().try_to_vec::<f32>().unwrap(),
    );
    for suffix in [
        "399", "400", "719", "720", "1040", "320000", "320080", "320400", "silence", "stereo",
    ] {
        eprintln!("extraction {suffix}");
        let r =
            reference::Reference::load(&references, &format!("audio_extract_{suffix}")).unwrap();
        let speech = &r.values["speech"];
        let mono: Vec<_> = if suffix == "stereo" {
            speech.1.chunks_exact(2).map(|v| v[0]).collect()
        } else {
            speech.1.clone()
        };
        if suffix == "399" {
            let e = extract_audio_features(&model, &mono).err().unwrap();
            assert!(!e.actions.is_empty());
            continue;
        }
        if mono.len() <= 1040 {
            use burn::tensor::{Tensor, TensorData};
            let raw = model.forward(Tensor::<2>::from_data(
                TensorData::new(normalize_waveform(&mono).unwrap(), [1, mono.len()]),
                &d,
            ));
            assert_eq!(raw.dims(), [1, (mono.len() - 80) / 320, 1024]);
            compare(
                &r,
                "hidden",
                &[raw.dims()[1], 1024],
                &raw.into_data().try_to_vec::<f32>().unwrap(),
            );
        }
        let f = extract_audio_features(&model, &mono).unwrap();
        assert_eq!(f.identity, *model.identity());
        assert_eq!(f.values.len(), f.video_frames * 2 * 1024);
        assert!(f.values.iter().all(|v| v.is_finite()));
        let key = if r.values.contains_key("even") {
            "even"
        } else {
            "hidden"
        };
        compare(&r, key, &[f.video_frames * 2, 1024], &f.values);
    }
}
#[test]
fn invalid_normalization_and_assets_fail() {
    for input in [&[][..], &[f32::NAN], &[f32::INFINITY]] {
        assert!(normalize_waveform(input).is_err());
    }
    assert_eq!(normalize_waveform(&[3.0; 720]).unwrap(), vec![0.0; 720]);
    let normalized = normalize_waveform(&[0.0, 1.0, 2.0]).unwrap();
    let scale = (2.0f64 / 3.0 + 1e-7).sqrt();
    assert_eq!(
        normalized,
        vec![(-1.0 / scale) as f32, 0.0, (1.0 / scale) as f32]
    );
    assert!(FeatherHubert::load(std::path::Path::new("missing-hubert"), &Device::flex()).is_err());
}

#[test]
fn damaged_assets_and_parameter_inventories_are_rejected() {
    use sha2::{Digest, Sha256};
    let (assets, _) = roots();
    let temp =
        std::env::temp_dir().join(format!("feathertalk-hubert-assets-{}", std::process::id()));
    std::fs::create_dir_all(&temp).unwrap();
    for key in ["manifest.json", "config.json", "model.safetensors"] {
        std::fs::copy(assets.join(key), temp.join(key)).unwrap();
    }
    let original_config = std::fs::read(temp.join("config.json")).unwrap();
    std::fs::write(temp.join("config.json"), b"{}").unwrap();
    assert!(FeatherHubert::load(&temp, &Device::flex()).is_err());
    std::fs::write(temp.join("config.json"), original_config).unwrap();
    let bytes = std::fs::read(temp.join("model.safetensors")).unwrap();
    let safe = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    for mode in ["missing", "shape", "extra"] {
        let mut entries = safe.tensors();
        if mode == "missing" {
            entries.retain(|(k, _)| k != "proj.bias");
        }
        if mode == "shape" {
            let t = entries.iter_mut().find(|(k, _)| k == "proj.bias").unwrap();
            t.1 = safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                vec![512, 2],
                t.1.data(),
            )
            .unwrap();
        }
        if mode == "extra" {
            entries.push((
                "unexpected.weight".into(),
                safe.tensor("proj.bias").unwrap(),
            ));
        }
        let changed = safetensors::tensor::serialize(entries, &None).unwrap();
        std::fs::write(temp.join("model.safetensors"), &changed).unwrap();
        let mut manifest: AssetManifest =
            serde_json::from_slice(&std::fs::read(assets.join("manifest.json")).unwrap()).unwrap();
        let hash = format!("{:x}", Sha256::digest(&changed));
        manifest
            .hashes
            .insert("model.safetensors".into(), hash.clone());
        manifest
            .sizes
            .insert("model.safetensors".into(), changed.len() as u64);
        manifest.identity.weights_sha256 = hash;
        std::fs::write(
            temp.join("manifest.json"),
            serde_json::to_vec(&manifest).unwrap(),
        )
        .unwrap();
        let error = FeatherHubert::load(&temp, &Device::flex()).err().unwrap();
        assert!(error.message.contains("parameter"), "{mode}: {error}");
    }
    std::fs::remove_dir_all(temp).unwrap();
}
