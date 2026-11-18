//! Real FeatherTalk/Adam state protocol tests, not six-loss training acceptance.
use super::checkpoint::*;
use super::state::{TrainingState, invalid};
use crate::data::prepared::{file_hash, reopen_generation};
use burn::{
    module::{Module, ModuleVisitor, Param},
    optim::GradientsParams,
    tensor::Tensor,
};
use feathertalk_core::*;
use std::{
    fs,
    path::{Path, PathBuf},
};

pub(super) fn fixture() -> (PathBuf, crate::data::prepared::PreparedProject) {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
        "../../.artifacts/task9b-projects/{}",
        ProjectId::new()
    ));
    fs::create_dir_all(root.join("cache/generation/frames")).unwrap();
    fs::create_dir(root.join("checkpoints")).unwrap();
    let root = fs::canonicalize(root).unwrap();
    fs::write(
        root.join("source.mkv"),
        b"owned state protocol source fixture",
    )
    .unwrap();
    let source = SourceVideo {
        path: "source.mkv".into(),
        sha256: file_hash(&root.join("source.mkv")).unwrap(),
        size_bytes: fs::metadata(root.join("source.mkv")).unwrap().len(),
    };
    let p = ProjectManifest {
        schema_version: 1,
        id: ProjectId::new(),
        name: "state protocol".into(),
        source_video: source.clone(),
        preprocessing: None,
        model_identity: None,
        checkpoints: vec![],
        metrics_path: "metrics.jsonl".into(),
    };
    fs::write(root.join("project.json"), serde_json::to_vec(&p).unwrap()).unwrap();
    let prep = PreprocessingIdentity {
        version: "fixture-v1".into(),
        source_sha256: source.sha256,
        config_sha256: "fixture-inputs".into(),
        cache_directory: "cache/generation".into(),
        component_identities: vec![],
    };
    let generation = root.join("cache/generation");
    let mut files = std::collections::BTreeMap::new();
    for (name, bytes) in [
        ("video.mkv", vec![1]),
        ("audio.wav", vec![2]),
        ("features.f32", vec![0; 2 * 2048 * 4]),
        ("frames/0.f32", vec![0; 4 * 144 * 144 * 4]),
        ("frames/1.f32", vec![0; 4 * 144 * 144 * 4]),
    ] {
        fs::write(generation.join(name), &bytes).unwrap();
        files.insert(name, digest(&bytes));
    }
    let m = serde_json::json!({"project_id":p.id,"identity":"owned-fixed-data-v1","input_identity":"fixture-inputs","preprocessing":prep,"frames":[
        {"file":"frames/0.f32","sha256":files["frames/0.f32"],"landmarks":[]},
        {"file":"frames/1.f32","sha256":files["frames/1.f32"],"landmarks":[]}],"feature_frames":2,"files":files});
    fs::write(
        generation.join("manifest.json"),
        serde_json::to_vec(&m).unwrap(),
    )
    .unwrap();
    let prepared = reopen_generation(
        &root,
        &p,
        Path::new("cache/generation"),
        &file_hash(&generation.join("manifest.json")).unwrap(),
    )
    .unwrap();
    (root, prepared)
}
pub(super) fn device() -> crate::ExecutionDevice {
    let backend = match std::env::var("FT_TEST_BACKEND").as_deref() {
        Ok("cuda") => BackendKind::Cuda,
        Ok("vulkan") => BackendKind::Vulkan,
        _ => BackendKind::Flex,
    };
    crate::open_device(&DeviceRequest {
        backend,
        device_id: format!(
            "{}:0",
            match backend {
                BackendKind::Cuda => "cuda",
                BackendKind::Vulkan => "vulkan",
                _ => "flex",
            }
        ),
    })
    .unwrap()
}
fn update(state: &mut TrainingState) {
    struct Objective(Option<Tensor<1>>);
    impl ModuleVisitor for Objective {
        fn visit_float<const D: usize>(&mut self, p: &Param<Tensor<D>>) {
            if p.is_require_grad() {
                // Includes genuine zero gradients for biases/constants; all IDs participate.
                let term = (p.val().clone() * p.val()).mean();
                self.0 = Some(match self.0.take() {
                    Some(sum) => sum + term,
                    None => term,
                });
            }
        }
    }
    let mut objective = Objective(None);
    state.model.visit(&mut objective);
    let gradients = GradientsParams::from_grads(objective.0.unwrap().backward(), &state.model);
    state.apply_gradients(gradients).unwrap();
    state.metadata.global_step += 1;
}

#[test]
fn complete_state_atomic_failures_and_strict_inventory() {
    let device = device();
    let (root, prepared) = fixture();
    let request = TrainRequest {
        batch_mode: BatchMode::Auto,
        project_id: prepared.project_id,
        config: TrainConfig::default(),
        device: DevicePolicy::Auto,
        resume: None,
    };
    let mut state = TrainingState::fresh(&request, &prepared, &device).unwrap();
    state.metadata.batch_mode = BatchMode::Fixed;
    state.metadata.actual_batch = 1;
    assert!(MaterializedSnapshot::capture(&state).is_err());
    state.metadata.batch_mode = BatchMode::Auto;
    state.metadata.actual_batch = request.config.batch_size;
    state.metadata.config.epsilon = 1e-5;
    assert!(state.optimizer_bytes().is_err());
    state.metadata.config.epsilon = request.config.epsilon;
    let pristine = MaterializedSnapshot::capture(&state).unwrap();
    let mut restored = pristine.restore(&root, &device).unwrap();
    update(&mut state);
    update(&mut restored);
    super::checkpoint_validation::equal_pack(
        &model_bytes(&state.model).unwrap(),
        &model_bytes(&restored.model).unwrap(),
    )
    .unwrap();
    let d = device.device.clone().autodiff();
    let bn = &state.model.inc.expand_bn;
    let _ = bn.forward(Tensor::ones([2, bn.weight.val().dims()[0], 2, 2], &d));
    assert_eq!(bn.state.get().count.val().into_scalar::<i64>(), 1);
    let snapshot = MaterializedSnapshot::capture(&state).unwrap();
    let owned_model_bytes = model_bytes(&state.model).unwrap();
    assert!(
        super::checkpoint_validation::validate_model(&owned_model_bytes[..100], &owned_model_bytes)
            .is_err()
    );
    let old = save_checkpoint(&state, &root.join("checkpoints/first")).unwrap();
    let loaded = load_checkpoint(&root, &old, &device).unwrap();
    assert_eq!(loaded.metadata, state.metadata);
    assert_eq!(
        loaded
            .model
            .inc
            .expand_bn
            .state
            .get()
            .count
            .val()
            .into_scalar::<i64>(),
        1
    );
    assert_eq!(
        loaded.model.inc.expand_bn.weight.id,
        state.model.inc.expand_bn.weight.id
    );
    let prediction = |model: &crate::models::feathertalk::FeatherTalk| {
        let model = model.valid();
        let d = model.devices()[0].clone();
        model
            .forward(
                Tensor::zeros([1, 6, 144, 144], &d),
                Tensor::zeros([1, 40, 1024], &d),
            )
            .to_data()
    };
    assert_eq!(prediction(&loaded.model), prediction(&state.model));
    let mut loaded = loaded;
    update(&mut state);
    update(&mut loaded);
    super::checkpoint_validation::equal_pack(
        &model_bytes(&state.model).unwrap(),
        &model_bytes(&loaded.model).unwrap(),
    )
    .unwrap();
    super::checkpoint_validation::equal_pack(
        &state.optimizer_bytes().unwrap(),
        &loaded.optimizer_bytes().unwrap(),
    )
    .unwrap();
    // Owned bytes remain unchanged after source mutation.
    let independent = snapshot.restore(&root, &device).unwrap();
    assert_eq!(independent.metadata.global_step, 1);
    super::checkpoint_validation::equal_pack(
        &owned_model_bytes,
        &model_bytes(&independent.model).unwrap(),
    )
    .unwrap();
    for phase in ["staged", "published", "index"] {
        assert!(
            save_with_hook(
                &state,
                &root.join(format!("checkpoints/fail-{phase}")),
                |p| if p == phase {
                    Err(invalid("injected interruption"))
                } else {
                    Ok(())
                }
            )
            .is_err()
        );
        assert_eq!(read_project(&root).unwrap().checkpoints.len(), 1);
        assert!(load_checkpoint(&root, &old, &device).is_ok());
    }
    let mut wrong = old.clone();
    wrong.project_id = ProjectId::new();
    assert!(load_checkpoint(&root, &wrong, &device).is_err());
    let mut different = prepared.clone();
    different.identity = "foreign".into();
    assert!(state.validate_prepared(&different).is_err());
    let records =
        super::checkpoint_validation::tensors(&model_bytes(&state.model).unwrap()).unwrap();
    let bytes = state.optimizer_bytes().unwrap();
    let r = burn_pack::Reader::from_bytes(burn_pack::Bytes::from_bytes_vec(bytes.clone())).unwrap();
    let scalars = r.scalars().clone();
    let paths = r.metadata().clone();
    let tensors = r.into_tensors().unwrap();
    for mutation in [
        "missing-moment",
        "missing-time",
        "wrong-time",
        "foreign-id",
        "wrong-shape",
    ] {
        let mut writer = Vec::new();
        for (i, t) in tensors.iter().enumerate() {
            if mutation == "missing-moment" && i == 0 {
                continue;
            }
            let mut t = t.clone();
            if mutation == "foreign-id" && i == 0 {
                t.param_id = Some(0);
            }
            if mutation == "wrong-shape" && i == 0 {
                t.shape = if t.shape.num_dims() == 1 {
                    [t.shape.num_elements(), 1].into()
                } else {
                    [t.shape.num_elements()].into()
                };
            }
            writer.push(t);
        }
        let mut writer = burn_pack::Writer::new(writer);
        let first_time = scalars
            .keys()
            .find(|k| k.ends_with("momentum.time"))
            .unwrap();
        for (k, v) in &scalars {
            if mutation == "missing-time" && k == first_time {
                continue;
            }
            writer = writer.with_scalar(
                k,
                if mutation == "wrong-time" && k == first_time {
                    burn_pack::Scalar::from(99u64)
                } else {
                    *v
                },
            );
        }
        for (k, v) in &paths {
            writer = writer.with_metadata(k, v);
        }
        let malformed = writer.into_bytes().unwrap();
        assert!(
            super::checkpoint_validation::validate_adam(&malformed, &records, 2).is_err(),
            "{mutation}"
        );
        if mutation == "missing-time" {
            let private = root.join("tampered-private");
            MaterializedSnapshot::capture(&state)
                .unwrap()
                .write_private(&private)
                .unwrap();
            fs::write(private.join("adam.bpk"), &*malformed).unwrap();
            let manifest_path = private.join("manifest.json");
            let mut manifest: serde_json::Value =
                serde_json::from_slice(&fs::read(&manifest_path).unwrap()).unwrap();
            manifest["optimizer_sha256"] = digest(&malformed).into();
            manifest["optimizer_size"] = (malformed.len() as u64).into();
            fs::write(manifest_path, serde_json::to_vec(&manifest).unwrap()).unwrap();
            assert!(
                MaterializedSnapshot::read_private(&private)
                    .unwrap()
                    .restore(&root, &device)
                    .is_err()
            );
        }
    }
    let result = save_with_hook(&state, &root.join("checkpoints/unrelated"), |phase| {
        if phase == "published" {
            let mut p = read_project(&root)?;
            p.name = "new user name".into();
            fs::write(root.join("project.json"), serde_json::to_vec(&p).unwrap()).unwrap();
        }
        Ok(())
    })
    .unwrap();
    assert_eq!(read_project(&root).unwrap().name, "new user name");
    assert!(load_checkpoint(&root, &result, &device).is_ok());
    if device.descriptor.backend != BackendKind::Flex {
        let cpu = crate::open_device(&DeviceRequest {
            backend: BackendKind::Flex,
            device_id: "flex:0".into(),
        })
        .unwrap();
        let mut on_cpu = load_checkpoint(&root, &old, &cpu).unwrap();
        update(&mut on_cpu);
        let cpu_ref = save_checkpoint(&on_cpu, &root.join("checkpoints/cpu-portable")).unwrap();
        let mut on_gpu = load_checkpoint(&root, &cpu_ref, &device).unwrap();
        update(&mut on_gpu);
        MaterializedSnapshot::capture(&on_gpu).unwrap(); // verifies all moments/parameters finite.
    }
    fs::write(root.join("source.mkv"), b"changed").unwrap();
    assert!(load_checkpoint(&root, &old, &device).is_err());
}

#[test]
fn batch_norm_lifecycle_is_independent_and_nontrainable() {
    let d = device().device.autodiff();
    let model = crate::models::feathertalk::FeatherTalk::new(&d);
    let bn = model.inc.expand_bn;
    let channels = bn.weight.val().dims()[0];
    let original_ids = (
        bn.state.get().mean.id,
        bn.state.get().variance.id,
        bn.state.get().count.id,
    );
    let valid = bn.valid();
    let training = valid.train();
    let forked = training.clone().fork(&d).to_device(&d);
    let _ = forked.forward(Tensor::ones([2, channels, 2, 2], &d));
    assert_eq!(forked.state.get().count.val().into_scalar::<i64>(), 1);
    assert_eq!(training.state.get().count.val().into_scalar::<i64>(), 0);
    assert_eq!(bn.state.get().count.val().into_scalar::<i64>(), 0);
    for b in [bn.state.get(), training.state.get(), forked.state.get()] {
        assert_eq!((b.mean.id, b.variance.id, b.count.id), original_ids);
        assert!(!b.mean.is_require_grad());
        assert!(!b.variance.is_require_grad());
        assert_eq!(b.count.val().dtype(), burn::tensor::DType::I64);
    }
}

#[test]
fn learning_rate_validation_preserves_restorable_state() {
    let device = device();
    let (root, prepared) = fixture();
    let request = TrainRequest {
        batch_mode: BatchMode::Auto,
        project_id: prepared.project_id,
        config: TrainConfig::default(),
        device: DevicePolicy::Auto,
        resume: None,
    };
    let mut state = TrainingState::fresh(&request, &prepared, &device).unwrap();
    let before_metadata = state.metadata.clone();
    let before_optimizer = state.optimizer_bytes().unwrap();
    let before_model = model_bytes(&state.model).unwrap();
    for rate in [1e100, 1e-100, 0., -0.001, f64::INFINITY, f64::NAN] {
        assert!(
            state.set_learning_rate(rate).is_err(),
            "accepted rate {rate}"
        );
        assert_eq!(state.metadata, before_metadata);
        assert_eq!(state.optimizer_bytes().unwrap(), before_optimizer);
    }
    super::checkpoint_validation::equal_pack(&before_model, &model_bytes(&state.model).unwrap())
        .unwrap();
    // A rejected setter must leave even a newly saved product checkpoint loadable.
    let reference = save_checkpoint(&state, &root.join("checkpoints/rejected-rate")).unwrap();
    let loaded = load_checkpoint(&root, &reference, &device).unwrap();
    assert_eq!(loaded.metadata, before_metadata);
    super::checkpoint_validation::equal_pack(&before_model, &model_bytes(&loaded.model).unwrap())
        .unwrap();
    for rate in [1e100, 1e-100] {
        let mut invalid_request = request.clone();
        invalid_request.config.lr = rate;
        assert!(TrainingState::fresh(&invalid_request, &prepared, &device).is_err());
        // Internal reconstruction cannot bypass the capture boundary's numeric validation.
        let mut metadata = before_metadata.clone();
        metadata.config.lr = rate;
        let invalid_state = TrainingState::from_parts(
            state.model.clone(),
            super::state::adam(&request.config).unwrap(),
            metadata.clone(),
            root.clone(),
        );
        assert!(MaterializedSnapshot::capture(&invalid_state).is_err());
        assert!(super::checkpoint::validate_metadata(&root, &metadata).is_err());
    }
    // Positive F32 subnormals are supported; values rounding to zero are not.
    let subnormal = f32::from_bits(1) as f64;
    state.set_learning_rate(subnormal).unwrap();
    let snapshot = MaterializedSnapshot::capture(&state).unwrap();
    let restored = snapshot.restore(&root, &device).unwrap();
    assert_eq!(restored.metadata.config.lr, subnormal);
    state.set_learning_rate(0.002).unwrap();
    let snapshot = MaterializedSnapshot::capture(&state).unwrap();
    let restored = snapshot.restore(&root, &device).unwrap();
    assert_eq!(restored.metadata, state.metadata);
    super::checkpoint_validation::equal_pack(&before_model, &model_bytes(&restored.model).unwrap())
        .unwrap();
    assert_eq!(restored.optimizer_bytes().unwrap(), before_optimizer);
}
