use burn::tensor::{Device, Tensor};
use feathertalk_core::LossConfig;
use feathertalk_engine::{models::vgg::VggFeatures, training::losses::compute_loss};
#[allow(dead_code)]
#[path = "../src/reference.rs"]
mod reference;
fn root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.artifacts/reference")
}
fn device() -> Device {
    use feathertalk_core::*;
    let backend = std::env::var("FT_TEST_BACKEND").unwrap_or("flex".into());
    let kind = match backend.as_str() {
        "flex" => BackendKind::Flex,
        "vulkan" => BackendKind::Vulkan,
        "cuda" => BackendKind::Cuda,
        _ => panic!("backend"),
    };
    let d = feathertalk_engine::open_device(&DeviceRequest {
        backend: kind,
        device_id: format!("{backend}:0"),
    })
    .unwrap();
    eprintln!("Loss actual device: {:?}", d.descriptor);
    d.device.autodiff()
}
fn vgg(d: &Device) -> VggFeatures {
    VggFeatures::load(&root().join("../assets/vgg19"), d).unwrap()
}
fn compare<const D: usize>(r: &reference::Reference, key: &str, x: Tensor<D>) {
    let a = x.to_data().try_to_vec::<f32>().unwrap();
    let e = &r.values[key].1;
    assert_eq!(a.len(), e.len());
    let t = &r.case.tolerance;
    let bad = a
        .iter()
        .zip(e)
        .filter(|(a, b)| {
            !a.is_finite() || (**a as f64 - **b as f64).abs() > t.atol + t.rtol * (**b as f64).abs()
        })
        .count();
    let max = a
        .iter()
        .zip(e)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    eprintln!("{key}: max_abs={max} violations={bad}");
    assert_eq!(bad, 0, "{key}");
}
#[test]
fn all_loss_terms_match_reference() {
    let d = device();
    let v = vgg(&d);
    let r = reference::Reference::load(&root(), "loss_individual_terms").unwrap();
    for (source, index) in [
        ("full", 0),
        ("mouth", 1),
        ("temp", 2),
        ("temp_mouth", 3),
        ("percep", 4),
        ("mouth_grad", 5),
    ] {
        let p = r.tensor::<5>("prediction", &d).require_grad();
        let o = compute_loss(
            p.clone(),
            r.tensor("labels", &d),
            r.tensor("masks", &d),
            &v,
            &LossConfig::default(),
        );
        let term = [
            o.parts.full,
            o.parts.mouth,
            o.parts.temporal,
            o.parts.temporal_mouth,
            o.parts.perceptual,
            o.parts.mouth_gradient,
        ]
        .into_iter()
        .nth(index)
        .unwrap();
        compare(&r, &format!("loss.{source}"), term.clone());
        let g = term.backward();
        compare(&r, &format!("gradient.{source}"), p.grad(&g).unwrap());
    }
}
#[test]
fn zero_mask_is_finite() {
    case("loss_empty", LossConfig::default());
}
#[test]
fn temporal_weights_zero_disable_temporal_path() {
    case(
        "loss_disabled",
        LossConfig {
            temporal: 0.,
            temporal_mouth: 0.,
            ..Default::default()
        },
    );
}
fn case(name: &str, c: LossConfig) {
    let d = device();
    let v = vgg(&d);
    let r = reference::Reference::load(&root(), name).unwrap();
    let p = r.tensor::<5>("prediction", &d).require_grad();
    let o = compute_loss(
        p.clone(),
        r.tensor("labels", &d),
        r.tensor("masks", &d),
        &v,
        &c,
    );
    for (key, x) in [
        ("full", o.parts.full),
        ("mouth", o.parts.mouth),
        ("temp", o.parts.temporal),
        ("temp_mouth", o.parts.temporal_mouth),
        ("percep", o.parts.perceptual),
        ("mouth_grad", o.parts.mouth_gradient),
    ] {
        compare(&r, &format!("loss.{key}"), x);
    }
    compare(&r, "total", o.total.clone());
    let g = o.total.backward();
    compare(&r, "gradient", p.grad(&g).unwrap());
}
#[test]
fn vgg_frozen_but_input_gradient_flows() {
    use burn::{
        module::{Module, ModuleVisitor, Param},
        optim::{AdamConfig, GradientsParams},
    };
    struct FrozenWeights(Vec<Vec<f32>>);
    impl ModuleVisitor for FrozenWeights {
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
            assert!(!p.is_require_grad());
            self.0.push(p.val().to_data().try_to_vec::<f32>().unwrap());
        }
    }
    let d = device();
    let v = vgg(&d);
    let mut before = FrozenWeights(vec![]);
    v.visit(&mut before);
    assert_eq!(before.0.len(), 14);
    let r = reference::Reference::load(&root(), "loss_individual_terms").unwrap();
    let p = r.tensor::<5>("prediction", &d).require_grad();
    let o = compute_loss(
        p.clone(),
        r.tensor("labels", &d),
        r.tensor("masks", &d),
        &v,
        &LossConfig::default(),
    );
    let g = o.total.backward();
    assert!(
        p.grad(&g)
            .unwrap()
            .to_data()
            .try_to_vec::<f32>()
            .unwrap()
            .iter()
            .any(|g| g.abs() > 0.)
    );
    let params = GradientsParams::from_grads(g, &v);
    assert_eq!(params.len(), 0);
    let mut optimizer = AdamConfig::new().with_epsilon(1e-8).init();
    let v = optimizer.step(0.001, v, params);
    let mut after = FrozenWeights(vec![]);
    v.visit(&mut after);
    assert_eq!(before.0, after.0);
}

#[test]
fn vgg_feature_layers_match_reference() {
    let d = device();
    let v = vgg(&d);
    let r = reference::Reference::load(&root(), "vgg19").unwrap();
    let out = v.forward_observed(r.tensor("input.0", &d), |index, x| {
        if [0, 4, 9, 14].contains(&index) {
            compare(&r, &format!("layer.{index}"), x);
        }
    });
    compare(&r, "output", out);
}
#[test]
fn complete_loss_prediction_gradient_matches_original_upstream() {
    let d = device();
    let v = vgg(&d);
    let r = reference::Reference::load(&root(), "loss_individual_terms").unwrap();
    let upstream = reference::Reference::load(&root(), "character_output_gradient").unwrap();
    let p = r.tensor::<5>("prediction", &d).require_grad();
    let o = compute_loss(
        p.clone(),
        r.tensor("labels", &d),
        r.tensor("masks", &d),
        &v,
        &LossConfig::default(),
    );
    let g = o.total.backward();
    compare(
        &upstream,
        "prediction_gradient",
        p.grad(&g).unwrap().reshape([2, 3, 144, 144]),
    );
}
#[cfg(feature = "dev-visual-fixtures")]
#[test]
fn complete_character_optimizer_step_keeps_vgg_frozen() {
    use burn::{
        module::{Module, ModuleVisitor, Param},
        optim::{AdamConfig, GradientsParams},
    };
    struct Weights(Vec<Vec<f32>>);
    impl ModuleVisitor for Weights {
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
            assert!(!p.is_require_grad());
            self.0.push(p.val().to_data().try_to_vec::<f32>().unwrap());
        }
    }
    let d = device();
    let v = vgg(&d);
    let mut before = Weights(vec![]);
    v.visit(&mut before);
    assert_eq!(before.0.len(), 14);
    let m = feathertalk_engine::models::visual_fixture::load_initial(&root(), &d).unwrap();
    let r = reference::Reference::load(&root(), "character_training").unwrap();
    let image = r.tensor::<4>("image", &d).require_grad();
    let prediction = m.forward(image.clone(), r.tensor("audio", &d));
    let labels = r.tensor::<5>("labels", &d);
    let masks = r.tensor::<5>("masks", &d);
    let o = compute_loss(
        prediction.clone().reshape([1, 2, 3, 144, 144]),
        labels,
        masks,
        &v,
        &LossConfig::default(),
    );
    let mut g = o.total.backward();
    assert!(
        image
            .grad(&g)
            .unwrap()
            .to_data()
            .try_to_vec::<f32>()
            .unwrap()
            .iter()
            .any(|g| g.abs() > 0.)
    );
    let previous_data = m.outc.weight.val().to_data();
    let previous = previous_data.try_to_vec::<f32>().unwrap();
    let params = GradientsParams::from_module(&mut g, &m);
    let frozen_params = GradientsParams::from_module(&mut g, &v);
    assert_eq!(params.len(), 252);
    assert_eq!(frozen_params.len(), 0);
    struct FiniteParameters<'a>(Option<&'a GradientsParams>);
    impl ModuleVisitor for FiniteParameters<'_> {
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
            if p.is_require_grad() {
                let value = match self.0 {
                    Some(grads) => grads.get::<D>(p.id).expect("character gradient"),
                    None => p.val(),
                };
                assert!(
                    value
                        .to_data()
                        .try_to_vec::<f32>()
                        .unwrap()
                        .iter()
                        .all(|v| v.is_finite())
                );
            }
        }
    }
    m.visit(&mut FiniteParameters(Some(&params)));
    let outc_grad = params
        .get::<4>(m.outc.weight.id)
        .expect("output weight gradient");
    assert!(outc_grad.abs().sum().into_scalar::<f32>() > 0.);
    let mut optimizer = AdamConfig::new().with_epsilon(1e-8).init();
    let m = optimizer.step(0.001, m, params);
    m.visit(&mut FiniteParameters(None));
    let after_data = m.outc.weight.val().to_data();
    let after = after_data.try_to_vec::<f32>().unwrap();
    eprintln!(
        "Adam max parameter delta: {}",
        previous
            .iter()
            .zip(&after)
            .map(|(a, b)| (a - b).abs())
            .fold(0f32, f32::max)
    );
    assert_ne!(previous, after);
    let v = optimizer.step(0.001, v, frozen_params);
    let mut after = Weights(vec![]);
    v.visit(&mut after);
    assert_eq!(before.0, after.0);
}
#[test]
fn fractional_neighbor_masks_union_clamp_and_batch_order() {
    use burn::tensor::TensorData;
    let d = device();
    let v = vgg(&d);
    let (b, t, c, h, w) = (2, 2, 3, 8, 8);
    let mut p = vec![0.; b * t * c * h * w];
    let mut mask = vec![0.; b * t * h * w];
    mask[0] = 0.25;
    mask[1] = 0.5;
    mask[h * w + 1] = 0.75;
    mask[h * w + 2] = 0.125;
    for n in 0..b * t {
        for channel in 0..c {
            for y in 0..h {
                for x in 0..w {
                    p[((n * c + channel) * h + y) * w + x] = 0.1 * (n + 1) as f32 + 0.01 * x as f32;
                }
            }
        }
    }
    let config = LossConfig {
        temporal_kernel: 1,
        temporal: 0.,
        ..Default::default()
    };
    let out = compute_loss(
        Tensor::from_data(TensorData::new(p.clone(), [b, t, c, h, w]), &d),
        Tensor::zeros([b, t, c, h, w], &d),
        Tensor::from_data(TensorData::new(mask.clone(), [b, t, 1, h, w]), &d),
        &v,
        &config,
    );
    let mut mouth = 0f32;
    let mut edge = 0f32;
    let mut edge_count = 0f32;
    let mut temporal = 0f32;
    let mut temporal_mouth = 0f32;
    let mut union_count = 0f32;
    for n in 0..b * t {
        for y in 0..h {
            for x in 0..w {
                let m = mask[(n * h + y) * w + x];
                for channel in 0..c {
                    let i = ((n * c + channel) * h + y) * w + x;
                    mouth += p[i].abs() * m;
                    if x + 1 < w {
                        edge += (p[i + 1] - p[i]).abs() * m * mask[(n * h + y) * w + x + 1];
                    }
                    if y + 1 < h {
                        edge += (p[i + w] - p[i]).abs() * m * mask[(n * h + y + 1) * w + x];
                    }
                }
                if x + 1 < w {
                    edge_count += m * mask[(n * h + y) * w + x + 1];
                }
                if y + 1 < h {
                    edge_count += m * mask[(n * h + y + 1) * w + x];
                }
            }
        }
    }
    for batch in 0..b {
        for y in 0..h {
            for x in 0..w {
                let u = mask[((batch * t) * h + y) * w + x]
                    .max(mask[((batch * t + 1) * h + y) * w + x]);
                union_count += u;
                for channel in 0..c {
                    let i = (((batch * t) * c + channel) * h + y) * w + x;
                    let delta = (p[i + c * h * w] - p[i]).abs();
                    temporal += delta;
                    temporal_mouth += delta * u;
                }
            }
        }
    }
    let s = out.parts.to_scalars();
    let close = |a: f64, b: f32| assert!((a - b as f64).abs() < 2e-6, "{a} != {b}");
    close(s.full, p.iter().sum::<f32>() / p.len() as f32);
    close(
        s.mouth,
        mouth / (mask.iter().sum::<f32>().max(1.) * c as f32),
    );
    close(s.temporal, temporal / (b * c * h * w) as f32);
    close(
        s.temporal_mouth,
        temporal_mouth / (union_count.max(1.) * c as f32),
    );
    close(s.mouth_gradient, edge / (edge_count.max(1.) * c as f32));
    assert!(
        s.temporal > 0.,
        "one zero coefficient still computes both temporal parts"
    );
    assert!(s.mouth_gradient > 0.);
}
#[test]
fn strict_vgg_loader_rejects_foreign_nonfinite_and_wrong_dtype() {
    use sha2::{Digest, Sha256};
    let d = device();
    let assets = root().join("../assets/vgg19");
    let temp = std::env::temp_dir().join(format!("feathertalk-vgg-{}", std::process::id()));
    std::fs::create_dir_all(&temp).unwrap();
    let manifest: serde_json::Value =
        serde_json::from_slice(&std::fs::read(assets.join("manifest.json")).unwrap()).unwrap();
    let original = std::fs::read(assets.join("model.safetensors")).unwrap();
    std::fs::copy(assets.join("config.json"), temp.join("config.json")).unwrap();
    for mode in ["key", "nonfinite", "dtype"] {
        let mut bytes = original.clone();
        match mode {
            "key" => {
                let at = bytes.windows(9).position(|s| s == b"14.weight").unwrap();
                bytes[at] = b'2';
            }
            "dtype" => {
                let at = bytes.windows(3).position(|s| s == b"F32").unwrap();
                bytes[at] = b'I';
            }
            _ => {
                let safe = safetensors::SafeTensors::deserialize(&bytes).unwrap();
                let view = safe.tensor("14.weight").unwrap();
                let at = view.data().as_ptr() as usize - bytes.as_ptr() as usize;
                bytes[at..at + 4].copy_from_slice(&f32::NAN.to_le_bytes());
            }
        }
        let mut m = manifest.clone();
        let hash = format!("{:x}", Sha256::digest(&bytes));
        m["hashes"]["model.safetensors"] = hash.clone().into();
        m["identity"]["weights_sha256"] = hash.into();
        std::fs::write(temp.join("model.safetensors"), bytes).unwrap();
        std::fs::write(temp.join("manifest.json"), serde_json::to_vec(&m).unwrap()).unwrap();
        assert!(VggFeatures::load(&temp, &d).is_err(), "{mode}");
    }
    for name in ["manifest.json", "config.json", "model.safetensors"] {
        std::fs::remove_file(temp.join(name)).unwrap();
    }
    std::fs::remove_dir(temp).unwrap();
}
