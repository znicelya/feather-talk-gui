use feathertalk_engine::training::state::SeedPolicy;

#[allow(dead_code)]
mod training_fixture;

#[cfg(feature = "flex")]
#[test]
fn committed_metadata_lookup_skips_device_restore_but_full_load_checks_weights() {
    use feathertalk_core::*;
    use feathertalk_engine::training::{
        checkpoint::{load_checkpoint, read_committed_metadata, save_checkpoint},
        state::TrainingState,
    };
    let (root, prepared) = training_fixture::fixture();
    let device = training_fixture::device();
    let state = TrainingState::fresh(
        &TrainRequest {
            project_id: prepared.project_id,
            batch_mode: BatchMode::Fixed,
            config: TrainConfig::default(),
            device: DevicePolicy::Auto,
            resume: None,
        },
        &prepared,
        &device,
    )
    .unwrap();
    let directory = root.join("checkpoints/metadata-fixture");
    let reference = save_checkpoint(&state, &directory).unwrap();
    assert_eq!(
        read_committed_metadata(&root, &reference).unwrap(),
        state.metadata
    );
    std::fs::write(directory.join("model.bpk"), b"damaged checkpoint weights").unwrap();
    assert_eq!(
        read_committed_metadata(&root, &reference).unwrap(),
        state.metadata
    );
    assert!(load_checkpoint(&root, &reference, &device).is_err());
    std::fs::write(directory.join("manifest.json"), b"damaged manifest").unwrap();
    assert!(read_committed_metadata(&root, &reference).is_err());
}

#[test]
fn seed_policy_is_versioned_and_domain_separated() {
    let policy = SeedPolicy::new(42);
    assert_eq!(policy.version, 1);
    assert_eq!(policy.step_seed(7), SeedPolicy::new(42).step_seed(7));
    assert_ne!(policy.step_seed(7), policy.step_seed(8));
    assert_ne!(policy.initialization_seed(), policy.step_seed(0));
}

#[test]
fn seed_policy_v1_persisted_replay_golden_vectors() {
    let policy = SeedPolicy::new(0);
    assert_eq!(policy.initialization_seed(), 15818207200996579459);
    assert_eq!(policy.step_seed(0), 6173558540527887718);
    assert_eq!(policy.step_seed(1), 9627746045979743105);
    assert_eq!(policy.step_seed(7), 3738356562821197916);
    assert_eq!(policy.step_seed(18446744073709551615), 3236206427424833754);
    let policy = SeedPolicy::new(42);
    assert_eq!(policy.initialization_seed(), 14677853989945954610);
    assert_eq!(policy.step_seed(0), 10450192142375125991);
    assert_eq!(policy.step_seed(1), 8430440234041518190);
    assert_eq!(policy.step_seed(7), 648826432181782192);
    assert_eq!(policy.step_seed(18446744073709551615), 18445903523277676708);
    let policy = SeedPolicy::new(18446744073709551615);
    assert_eq!(policy.initialization_seed(), 5905685506207859730);
    assert_eq!(policy.step_seed(0), 8075582923876958131);
    assert_eq!(policy.step_seed(1), 16360067611579612329);
    assert_eq!(policy.step_seed(7), 17571055357269856679);
    assert_eq!(policy.step_seed(18446744073709551615), 16110727864703306287);
}
