use feathertalk_engine::training::state::SeedPolicy;

#[test]
fn seed_policy_is_versioned_and_domain_separated() {
    let policy = SeedPolicy::new(42);
    assert_eq!(policy.version, 1);
    assert_eq!(policy.step_seed(7), SeedPolicy::new(42).step_seed(7));
    assert_ne!(policy.step_seed(7), policy.step_seed(8));
    assert_ne!(policy.initialization_seed(), policy.step_seed(0));
}
