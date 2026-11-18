use burn::{module::Module, tensor::Tensor};
use feathertalk_engine::models::feathertalk::FeatherTalk;
#[test]
fn fresh_model_forward_and_split() {
    let d = device();
    let m = FeatherTalk::new(&d).valid();
    let image = Tensor::zeros([1, 6, 144, 144], &d);
    let audio = Tensor::zeros([1, 40, 1024], &d);
    let full = m.forward(image.clone(), audio.clone());
    assert_eq!(full.dims(), [1, 3, 144, 144]);
    let split = m.decode_features(m.encode_image(image), audio);
    assert_eq!(full.to_data(), split.to_data());
    use burn_store::{BurnpackStore, ModuleSnapshot};
    let mut store = BurnpackStore::from_bytes(None);
    m.save_into(&mut store).unwrap();
    let mut restored = FeatherTalk::new(&d).valid();
    restored
        .load_from(&mut BurnpackStore::from_bytes(Some(
            store.get_bytes().unwrap(),
        )))
        .unwrap();
    let restored_output = restored.forward(
        Tensor::zeros([1, 6, 144, 144], &d),
        Tensor::zeros([1, 40, 1024], &d),
    );
    assert_eq!(full.to_data(), restored_output.to_data());
}
#[cfg(feature = "dev-visual-fixtures")]
#[allow(dead_code)]
#[path = "../src/reference.rs"]
mod reference;
fn device() -> burn::tensor::Device {
    use feathertalk_core::*;
    use feathertalk_engine::open_device;
    let backend = std::env::var("FT_TEST_BACKEND").unwrap_or("flex".into());
    let kind = match backend.as_str() {
        "flex" => BackendKind::Flex,
        "vulkan" => BackendKind::Vulkan,
        "cuda" => BackendKind::Cuda,
        _ => panic!("backend"),
    };
    let d = open_device(&DeviceRequest {
        backend: kind,
        device_id: format!("{backend}:0"),
    })
    .unwrap();
    eprintln!("Visual actual device: {:?}", d.descriptor);
    d.device
}
#[cfg(feature = "dev-visual-fixtures")]
fn root() -> std::path::PathBuf {
    std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.artifacts/reference")
}
#[cfg(feature = "dev-visual-fixtures")]
fn compare<const D: usize>(r: &reference::Reference, key: &str, x: Tensor<D>) {
    let actual = x.to_data().try_to_vec::<f32>().unwrap();
    let (shape, expected) = &r.values[key];
    assert_eq!(actual.len(), expected.len(), "{key} {shape:?}");
    let tol = r
        .case
        .tensor_tolerances
        .get(key)
        .unwrap_or(&r.case.tolerance);
    let errors = actual
        .iter()
        .zip(expected)
        .filter(|(a, b)| {
            !a.is_finite()
                || (**a as f64 - **b as f64).abs() > tol.atol + tol.rtol * (**b as f64).abs()
        })
        .count();
    let max = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("{key}: max_abs={max} violations={errors}");
    assert_eq!(errors, 0, "{key}");
}
#[cfg(feature = "dev-visual-fixtures")]
#[test]
fn canonical_eval_layers() {
    use burn::tensor::activation::sigmoid;
    use feathertalk_engine::models::{visual_blocks::up, visual_fixture::load_initial};
    let d = device();
    let m = load_initial(&root(), &d).unwrap().valid();
    let r = reference::Reference::load(&root(), "character_eval").unwrap();
    let image = r.tensor("input.0", &d);
    let audio = r.tensor("input.1", &d);
    let f = m.encode_image(image.clone());
    compare(&r, "layer.inc", f.x1.clone());
    compare(&r, "layer.down2", f.x3.clone());
    compare(&r, "layer.down4", f.x5.clone());
    let a = m.audio_model.forward(audio.clone());
    compare(&r, "layer.audio_model", a.clone());
    let x3 = m.spatial_x3.forward(f.x3.clone(), a.clone());
    compare(&r, "layer.spatial_x3", x3.clone());
    let x4 = m.spatial_x4.forward(f.x4.clone(), a.clone());
    let x5 = m.spatial_x5.forward(f.x5.clone(), a.clone());
    let [b, c, h, w] = x5.dims();
    let ctx = m
        .audio_to_x5
        .forward(m.audio_context.forward(a))
        .reshape([b, c, 1, 1])
        .expand([b, c, h, w]);
    let x = m.fuse_conv.forward(Tensor::cat(vec![x5, ctx], 1));
    compare(&r, "layer.fuse_conv", x.clone());
    let x = up(&m.up1, x, x4);
    let x = up(&m.up2, x, x3);
    let x = up(&m.up3, x, f.x2.clone());
    let x = up(&m.up4, x, f.x1.clone());
    compare(&r, "layer.up4", x.clone());
    let logits = m.outc.forward(x);
    compare(&r, "layer.outc", logits.clone());
    compare(&r, "output", sigmoid(logits));
    compare(&r, "output", m.decode_features(f, audio.clone()));
    compare(&r, "output", m.forward(image, audio));
}
#[test]
fn odd_crop_uses_top_left_and_padding_bottom_right() {
    use feathertalk_engine::models::visual_blocks::upsample_to_skip;
    let d = device();
    let x = Tensor::from_floats([[[[1., 2.], [3., 4.]]]], &d);
    let doubled = feathertalk_engine::ops::resize::resize_bilinear_align_corners(x.clone(), [4, 4]);
    assert_eq!(
        upsample_to_skip(x.clone(), [3, 3]).to_data(),
        doubled.clone().slice([0..1, 0..1, 1..4, 1..4]).to_data()
    );
    let padded = upsample_to_skip(x, [5, 5]);
    assert_eq!(
        padded.clone().slice([0..1, 0..1, 0..4, 0..4]).to_data(),
        doubled.to_data()
    );
    assert_eq!(
        padded
            .slice([0..1, 0..1, 4..5, 0..5])
            .sum()
            .into_scalar::<f32>(),
        0.
    );
}

#[cfg(feature = "dev-visual-fixtures")]
#[test]
fn canonical_training_vjp_and_all_bn_buffers() {
    use burn::{
        module::{ModuleVisitor, Param},
        tensor::{Gradients, Int},
    };
    use feathertalk_engine::models::visual_fixture::{load_initial, source_key};
    let d = device().autodiff();
    let m = load_initial(&root(), &d).unwrap();
    let r = reference::Reference::load(&root(), "character_training").unwrap();
    let upstream = reference::Reference::load(&root(), "character_output_gradient").unwrap();
    let image = r.tensor("image", &d).require_grad();
    let audio = r.tensor("audio", &d).require_grad();
    let output = m.forward(image.clone(), audio.clone());
    compare(&r, "prediction", output.clone());
    let grads = (output * upstream.tensor::<4>("prediction_gradient", &d))
        .sum()
        .backward();
    compare(&r, "image_gradient", image.grad(&grads).unwrap());
    compare(&r, "audio_gradient", audio.grad(&grads).unwrap());
    struct Check<'a> {
        path: Vec<String>,
        r: &'a reference::Reference,
        g: &'a Gradients,
        gradient_count: usize,
        state_count: usize,
    }
    impl ModuleVisitor for Check<'_> {
        fn enter_module(&mut self, n: &str, _: &str) {
            self.path.push(n.into());
        }
        fn exit_module(&mut self, _: &str, _: &str) {
            self.path.pop();
        }
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
            let key = source_key(&self.path.join("."));
            let grad = format!("gradient.{key}");
            if self.r.values.contains_key(&grad) {
                compare(self.r, &grad, p.grad(self.g).expect("registered gradient"));
                self.gradient_count += 1;
            }
            let state = format!("updated_state.{key}");
            if self.r.values.contains_key(&state) {
                assert!(!p.is_require_grad());
                compare(self.r, &state, p.val());
                self.state_count += 1;
            }
        }
        fn visit_int<const D: usize>(&mut self, p: &Param<Tensor<D, Int>>) {
            let key = format!("updated_state.{}", source_key(&self.path.join(".")));
            assert_eq!(
                p.val().to_data().try_to_vec::<i64>().unwrap(),
                vec![self.r.values[&key].1[0] as i64]
            );
            self.state_count += 1;
        }
    }
    let mut check = Check {
        path: vec![],
        r: &r,
        g: &grads,
        gradient_count: 0,
        state_count: 0,
    };
    m.visit(&mut check);
    assert_eq!(check.gradient_count, 5);
    assert_eq!(check.state_count, 195);
}
#[test]
fn bn_independent_clone_latest_snapshot_and_optimizer() {
    use burn::{
        optim::{AdamConfig, GradientsParams},
        tensor::TensorData,
    };
    use burn_store::{BurnpackStore, ModuleSnapshot};
    let d = device().autodiff();
    let m = FeatherTalk::new(&d);
    let bn = m.inc.expand_bn;
    let original = bn.clone();
    let x = Tensor::from_data(
        TensorData::new(
            (0..48).map(|i| i as f32 * 0.01).collect::<Vec<_>>(),
            [2, 12, 2, 1],
        ),
        &d,
    );
    let y = bn.forward(x.clone());
    let state = bn.state.get();
    assert_eq!(
        state.count.val().to_data().try_to_vec::<i64>().unwrap(),
        [1]
    );
    assert_eq!(
        original
            .state
            .get()
            .count
            .val()
            .to_data()
            .try_to_vec::<i64>()
            .unwrap(),
        [0]
    );
    let mut store = BurnpackStore::from_bytes(None);
    bn.save_into(&mut store).unwrap();
    let bytes = store.get_bytes().unwrap();
    let mut restored = original.clone();
    restored
        .load_from(&mut BurnpackStore::from_bytes(Some(bytes)))
        .unwrap();
    assert_eq!(
        restored.state.get().mean.val().to_data(),
        state.mean.val().to_data()
    );
    assert_eq!(
        restored.state.get().variance.val().to_data(),
        state.variance.val().to_data()
    );
    assert_eq!(
        restored.state.get().count.val().to_data(),
        state.count.val().to_data()
    );
    assert_eq!(restored.weight.id, bn.weight.id);
    let eval = restored.valid();
    let before = eval.state.get();
    eval.forward(x.clone().without_autodiff());
    assert_eq!(
        eval.state.get().count.val().to_data(),
        before.count.val().to_data()
    );
    let before_weight = bn.weight.val().to_data();
    let grad = (y * y_weights(&d)).sum().backward();
    assert!(bn.weight.grad(&grad).is_some());
    assert!(bn.bias.grad(&grad).is_some());
    assert!(state.mean.grad(&grad).is_none());
    let mut opt = AdamConfig::new().with_epsilon(1e-8).init();
    let gradients = GradientsParams::from_grads(grad, &bn);
    let updated = opt.step(0.001, bn, gradients);
    assert_ne!(updated.weight.val().to_data(), before_weight);
    let next = restored.forward(x.clone());
    assert_eq!(
        restored
            .state
            .get()
            .count
            .val()
            .to_data()
            .try_to_vec::<i64>()
            .unwrap(),
        [2]
    );
    assert_eq!(
        original
            .state
            .get()
            .count
            .val()
            .to_data()
            .try_to_vec::<i64>()
            .unwrap(),
        [0]
    );
    assert_eq!(next.dims(), [2, 12, 2, 1]);
}
fn y_weights(d: &burn::tensor::Device) -> Tensor<4> {
    Tensor::from_data(
        burn::tensor::TensorData::new(
            (0..48).map(|i| (i * i) as f32 * 0.01).collect::<Vec<_>>(),
            [2, 12, 2, 1],
        ),
        d,
    )
}

#[cfg(feature = "dev-visual-fixtures")]
#[test]
fn isolated_attention_all_parameters_and_gradients() {
    use burn::{
        module::{ModuleVisitor, Param},
        tensor::Gradients,
    };
    let d = device().autodiff();
    let m = feathertalk_engine::models::visual_fixture::load_attention(&root(), &d).unwrap();
    let r = reference::Reference::load(&root(), "spatial_audio_attention").unwrap();
    let v = r.tensor("input.0", &d).require_grad();
    let a = r.tensor("input.1", &d).require_grad();
    let y = m.forward(v.clone(), a.clone());
    compare(&r, "output", y.clone());
    let grads = (y * r.tensor::<4>("upstream_gradient", &d))
        .sum()
        .backward();
    compare(&r, "input_gradient.0", v.grad(&grads).unwrap());
    compare(&r, "input_gradient.1", a.grad(&grads).unwrap());
    struct Check<'a> {
        path: Vec<String>,
        r: &'a reference::Reference,
        g: &'a Gradients,
        count: usize,
    }
    impl ModuleVisitor for Check<'_> {
        fn enter_module(&mut self, n: &str, _: &str) {
            self.path.push(n.into());
        }
        fn exit_module(&mut self, _: &str, _: &str) {
            self.path.pop();
        }
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
            let key = format!("gradient.{}", self.path.join("."));
            compare(self.r, &key, p.grad(self.g).unwrap());
            self.count += 1;
        }
    }
    let mut check = Check {
        path: vec![],
        r: &r,
        g: &grads,
        count: 0,
    };
    m.visit(&mut check);
    assert_eq!(check.count, 15);
}
#[test]
fn fresh_network_all_parameters_optimize_and_checkpoint() {
    use burn::{
        module::{ModuleVisitor, Param},
        optim::{AdamConfig, GradientsParams},
        tensor::{Distribution, Gradients},
    };
    use burn_store::{BurnpackStore, ModuleSnapshot};
    let d = device().autodiff();
    let m = FeatherTalk::new(&d);
    let pristine = m.clone();
    let image = Tensor::random([1, 6, 144, 144], Distribution::Uniform(-1., 1.), &d);
    let audio = Tensor::random([1, 40, 1024], Distribution::Uniform(-1., 1.), &d);
    let output = m.forward(image.clone(), audio.clone());
    let target = Tensor::random([1, 3, 144, 144], Distribution::Uniform(0., 1.), &d);
    let error = output - target;
    let grads = (error.clone() * error).mean().backward();
    struct Check<'a> {
        grads: &'a Gradients,
        count: usize,
    }
    impl ModuleVisitor for Check<'_> {
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
            if p.is_require_grad() {
                let gradient = p
                    .grad(self.grads)
                    .expect("every fresh parameter participates in graph");
                assert!(
                    gradient
                        .to_data()
                        .try_to_vec::<f32>()
                        .unwrap()
                        .iter()
                        .all(|v| v.is_finite())
                );
                self.count += 1;
            }
        }
    }
    let mut check = Check {
        grads: &grads,
        count: 0,
    };
    m.visit(&mut check);
    assert_eq!(check.count, 252);
    let before = m.outc.weight.val().to_data();
    let gradients = GradientsParams::from_grads(grads, &m);
    let mut optimizer = AdamConfig::new().with_epsilon(1e-8).init();
    let m = optimizer.step(0.001, m, gradients);
    assert_ne!(m.outc.weight.val().to_data(), before);
    assert_eq!(
        m.inc.expand_bn.state.get().count.val().into_scalar::<i64>(),
        1
    );
    assert_eq!(
        pristine
            .inc
            .expand_bn
            .state
            .get()
            .count
            .val()
            .into_scalar::<i64>(),
        0
    );
    let mut store = BurnpackStore::from_bytes(None);
    m.save_into(&mut store).unwrap();
    let bytes = store.get_bytes().unwrap();
    let mut restored = FeatherTalk::new(&d);
    restored
        .load_from(&mut BurnpackStore::from_bytes(Some(bytes)))
        .unwrap();
    assert_eq!(m.outc.weight.id, restored.outc.weight.id);
    let current = m.collect(None, None, false);
    let loaded = restored.collect(None, None, false);
    assert_eq!(current.len(), 447);
    assert_eq!(loaded.len(), 447);
    for (a, b) in current.iter().zip(&loaded) {
        assert_eq!(a.name, b.name);
        assert_eq!(a.dtype, b.dtype);
        assert_eq!(a.shape, b.shape);
        assert_eq!(a.param_id, b.param_id);
        assert_eq!(
            a.to_bytes().unwrap(),
            b.to_bytes().unwrap(),
            "checkpoint {}",
            a.name
        );
    }

    let eval = m.valid().forward(
        image.clone().without_autodiff(),
        audio.clone().without_autodiff(),
    );
    let eval_loaded = restored.valid().forward(
        image.clone().without_autodiff(),
        audio.clone().without_autodiff(),
    );
    assert_eq!(eval.to_data(), eval_loaded.to_data());
    let next = m.forward(image.clone(), audio.clone());
    let next_loaded = restored.decode_features(restored.encode_image(image), audio);
    assert_eq!(next.to_data(), next_loaded.to_data());
    assert_eq!(
        restored
            .inc
            .expand_bn
            .state
            .get()
            .count
            .val()
            .into_scalar::<i64>(),
        2
    );
    assert_eq!(
        pristine
            .inc
            .expand_bn
            .state
            .get()
            .count
            .val()
            .into_scalar::<i64>(),
        0
    );
}
#[cfg(feature = "dev-visual-fixtures")]
#[test]
fn isolated_visual_block_gradients_and_bn_state() {
    use burn::{
        module::{ModuleVisitor, Param},
        tensor::{Gradients, Int},
    };
    use feathertalk_engine::models::visual_fixture::{block_source_key, load_block};
    let d = device().autodiff();
    let m = load_block(&root(), &d).unwrap();
    let r = reference::Reference::load(&root(), "visual_inverted_residual").unwrap();
    let x = r.tensor("input", &d).require_grad();
    let y = m.forward(x.clone());
    compare(&r, "output", y.clone());
    let grads = (y * r.tensor::<4>("upstream_gradient", &d))
        .sum()
        .backward();
    compare(&r, "input_gradient", x.grad(&grads).unwrap());
    struct Check<'a> {
        path: Vec<String>,
        r: &'a reference::Reference,
        g: &'a Gradients,
        grads: usize,
        states: usize,
    }
    impl ModuleVisitor for Check<'_> {
        fn enter_module(&mut self, n: &str, _: &str) {
            self.path.push(n.into());
        }
        fn exit_module(&mut self, _: &str, _: &str) {
            self.path.pop();
        }
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
            let key = block_source_key(&self.path.join("."));
            if p.is_require_grad() {
                compare(self.r, &format!("gradient.{key}"), p.grad(self.g).unwrap());
                self.grads += 1;
            } else {
                compare(self.r, &format!("updated_state.{key}"), p.val());
                self.states += 1;
            }
        }
        fn visit_int<const D: usize>(&mut self, p: &Param<Tensor<D, Int>>) {
            assert_eq!(p.val().into_scalar::<i64>(), 1);
            self.states += 1;
        }
    }
    let mut check = Check {
        path: vec![],
        r: &r,
        g: &grads,
        grads: 0,
        states: 0,
    };
    m.visit(&mut check);
    assert_eq!(check.grads, 9);
    assert_eq!(check.states, 9);
    for (key, (_, values)) in &r.values {
        if key == "input_gradient" || key.starts_with("gradient.") {
            let t = &r.case.tolerance;
            for factor in [0., 1.01] {
                assert!(
                    values
                        .iter()
                        .any(|&v| ((v as f64) * factor - v as f64).abs()
                            > t.atol + t.rtol * (v as f64).abs()),
                    "mutation {key} {factor}"
                );
            }
        }
    }
}
#[cfg(feature = "dev-visual-fixtures")]
#[test]
fn calibrated_gradient_zero_mutations_are_rejected() {
    let r = reference::Reference::load(&root(), "character_training").unwrap();
    assert_eq!(r.case.tensor_tolerances.len(), 4);
    for (key, t) in &r.case.tensor_tolerances {
        assert!(
            r.values[key]
                .1
                .iter()
                .any(|&v| (v as f64).abs() > t.atol + t.rtol * (v as f64).abs()),
            "zero gradient {key}"
        );
    }
}
#[cfg(feature = "dev-visual-fixtures")]
#[test]
fn fixed_state_loader_rejects_foreign_keys_and_nonfinite_values() {
    use sha2::{Digest, Sha256};
    let d = device();
    let temp =
        std::env::temp_dir().join(format!("feathertalk-visual-fixture-{}", std::process::id()));
    std::fs::create_dir_all(&temp).unwrap();
    let manifest = std::fs::read(root().join("manifest.json")).unwrap();
    let original = std::fs::read(root().join("character_initial_state.safetensors")).unwrap();
    for mode in ["key", "nonfinite"] {
        let mut bytes = original.clone();
        if mode == "key" {
            let needle = b"audio_model.norm.bias";
            let at = bytes
                .windows(needle.len())
                .position(|s| s == needle)
                .unwrap();
            bytes[at + needle.len() - 1] = b'z';
        } else {
            let tensors = safetensors::SafeTensors::deserialize(&bytes).unwrap();
            let tensor = tensors.tensor("audio_model.norm.weight").unwrap();
            let at = tensor.data().as_ptr() as usize - bytes.as_ptr() as usize;
            bytes[at..at + 4].copy_from_slice(&f32::INFINITY.to_le_bytes());
        }
        let mut metadata: serde_json::Value = serde_json::from_slice(&manifest).unwrap();
        metadata["cases"]["character_initial_state"]["sha256"] =
            format!("{:x}", Sha256::digest(&bytes)).into();
        std::fs::write(
            temp.join("manifest.json"),
            serde_json::to_vec(&metadata).unwrap(),
        )
        .unwrap();
        std::fs::write(temp.join("character_initial_state.safetensors"), bytes).unwrap();
        assert!(
            feathertalk_engine::models::visual_fixture::load_initial(&temp, &d).is_err(),
            "{mode}"
        );
    }
    std::fs::remove_file(temp.join("manifest.json")).unwrap();
    std::fs::remove_file(temp.join("character_initial_state.safetensors")).unwrap();
    std::fs::remove_dir(temp).unwrap();
}
