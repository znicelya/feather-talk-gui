mod training_fixture;
use feathertalk_core::*;
use feathertalk_engine::{
    models::vgg::VggFeatures,
    training::{batch_probe::*, checkpoint::*, state::TrainingState, trainer::complete_step},
};
use std::{fs, path::Path};
#[test]
fn actual_bound_candidate() {
    let (root, prepared) = training_fixture::fixture();
    let device = training_fixture::device();
    let request = TrainRequest {
        project_id: prepared.project_id,
        batch_mode: BatchMode::Auto,
        config: TrainConfig {
            epochs: 1,
            batch_size: 1,
            ..Default::default()
        },
        device: DevicePolicy::Auto,
        resume: None,
    };
    let mut state = TrainingState::fresh(&request, &prepared, &device).unwrap();
    let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.artifacts/assets/vgg19");
    let vgg = VggFeatures::load(&assets, &device.device.clone().autodiff()).unwrap();
    state.metadata.vgg_identity = Some(vgg.identity().clone());
    let original = state.metadata.clone();
    let selected = probe_batch_size(
        &state,
        &prepared,
        &device,
        &ProbeContext {
            runtime: &TrainingRuntime {
                probe_runner: env!("CARGO_BIN_EXE_feathertalk-probe").into(),
            },
            vgg_assets: &assets,
            cancel: &Cancellation::new(),
        },
    )
    .unwrap();
    assert_eq!(selected.candidate, 1);
    assert_eq!(selected.effective_pairs, 1);
    assert_eq!(state.metadata, original);
    assert!(!fs::read_dir(&root).unwrap().any(|e| {
        e.unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".probe-")
    }));
    eprintln!(
        "Actual identity-bound candidate succeeded on {:?}",
        device.descriptor
    );
}
fn equal(a: &[u8], b: &[u8]) {
    let a = burn_pack::Reader::from_bytes(burn_pack::Bytes::from_bytes_vec(a.to_vec())).unwrap();
    let b = burn_pack::Reader::from_bytes(burn_pack::Bytes::from_bytes_vec(b.to_vec())).unwrap();
    assert_eq!(a.scalars(), b.scalars());
    assert_eq!(a.metadata(), b.metadata());
    let a = a.into_tensors().unwrap();
    let b = b.into_tensors().unwrap();
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b.iter()) {
        assert_eq!(a.name, b.name);
        assert_eq!(a.param_id, b.param_id);
        assert_eq!(a.shape, b.shape);
        assert_eq!(a.dtype, b.dtype);
        assert_eq!(a.to_bytes().unwrap(), b.to_bytes().unwrap());
    }
}
fn compare(a: &TrainingState, b: &TrainingState, root: &Path) {
    assert_eq!(a.metadata, b.metadata);
    equal(&a.optimizer_bytes().unwrap(), &b.optimizer_bytes().unwrap());
    for (state, name) in [(a, "a"), (b, "b")] {
        MaterializedSnapshot::capture(state)
            .unwrap()
            .write_private(&root.join(name))
            .unwrap();
    }
    equal(
        &fs::read(root.join("a/model.bpk")).unwrap(),
        &fs::read(root.join("b/model.bpk")).unwrap(),
    );
    fs::remove_dir_all(root.join("a")).unwrap();
    fs::remove_dir_all(root.join("b")).unwrap();
}
#[test]
fn actual_candidate_preserves_state_and_next_full_update() {
    let (root, prepared) = training_fixture::fixture();
    let device = training_fixture::device();
    eprintln!("ACTUAL PROBE DEVICE {:?}", device.descriptor);
    let mut request = TrainRequest {
        project_id: prepared.project_id,
        batch_mode: BatchMode::Fixed,
        config: TrainConfig {
            epochs: 2,
            batch_size: 2,
            ..Default::default()
        },
        device: DevicePolicy::Auto,
        resume: None,
    };
    let mut state = TrainingState::fresh(&request, &prepared, &device).unwrap();
    let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.artifacts/assets/vgg19");
    let cancel = Cancellation::new();
    let missing = TrainingRuntime {
        probe_runner: root.join("does-not-exist.exe"),
    };
    assert!(
        probe_batch_size(
            &state,
            &prepared,
            &device,
            &ProbeContext {
                runtime: &missing,
                vgg_assets: &assets,
                cancel: &cancel
            }
        )
        .unwrap()
        .attempts
        .is_empty()
    );
    request.batch_mode = BatchMode::Auto;
    state.metadata.batch_mode = BatchMode::Auto;
    let mut control = MaterializedSnapshot::capture(&state)
        .unwrap()
        .restore(&root, &device)
        .unwrap();
    let runtime = TrainingRuntime {
        probe_runner: env!("CARGO_BIN_EXE_feathertalk-probe").into(),
    };
    let selected = probe_batch_size(
        &state,
        &prepared,
        &device,
        &ProbeContext {
            runtime: &runtime,
            vgg_assets: &assets,
            cancel: &cancel,
        },
    )
    .unwrap();
    assert_eq!(selected.candidate, 2);
    assert_eq!(selected.effective_pairs, 1);
    assert_eq!(selected.attempts, vec![2]);
    compare(&state, &control, &root);
    assert!(!fs::read_dir(&root).unwrap().any(|e| {
        e.unwrap()
            .file_name()
            .to_string_lossy()
            .starts_with(".probe-")
    }));
    let vgg = VggFeatures::load(&assets, &device.device.clone().autodiff()).unwrap();
    let a = complete_step(&mut state, &prepared, &vgg, &device, &cancel).unwrap();
    let b = complete_step(&mut control, &prepared, &vgg, &device, &cancel).unwrap();
    assert_eq!(a.pairs, b.pairs);
    assert_eq!(a.references, b.references);
    assert_eq!(a.seed, b.seed);
    compare(&state, &control, &root);
    eprintln!("Actual full-step child + exact parent continuation passed");
}
