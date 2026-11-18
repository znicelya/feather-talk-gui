use feathertalk_core::*;
#[test]
fn training_configuration_defaults_and_bounds() {
    let mut wire = serde_json::to_value(TrainConfig::default()).unwrap();
    wire.as_object_mut().unwrap().remove("seed");
    wire.as_object_mut().unwrap().remove("prefetch_capacity");
    let mut config: TrainConfig = serde_json::from_value(wire).unwrap();
    assert_eq!(config.seed, 42);
    assert_eq!(config.prefetch_capacity, 2);
    for capacity in [0, 1, 32] {
        config.prefetch_capacity = capacity;
        config.validate().unwrap();
    }
    config.prefetch_capacity = 33;
    assert!(config.validate().is_err());
    config = TrainConfig::default();
    config.batch_size = 1025;
    assert!(config.validate().is_err());
}
#[test]
fn fixed_batch_configuration_roundtrip_preserves_target() {
    let wire = serde_json::json!({"project_id":ProjectId::new(),"config":TrainConfig::default(),"device":{"policy":"auto"},"resume":null});
    let mut request: TrainRequest = serde_json::from_value(wire).unwrap();
    assert_eq!(request.batch_mode, BatchMode::Auto);
    request.batch_mode = BatchMode::Fixed;
    request.config.batch_size = 7;
    request.config.seed = 1234567890123;
    request.config.prefetch_capacity = 0;
    let restored: TrainRequest =
        serde_json::from_slice(&serde_json::to_vec(&request).unwrap()).unwrap();
    assert_eq!(restored, request);
    assert_eq!(restored.config.batch_size, 7);
}

#[test]
fn compatibility_defaults_only_new_fields() {
    let mut config = serde_json::to_value(TrainConfig::default()).unwrap();
    config.as_object_mut().unwrap().remove("lr");
    assert!(serde_json::from_value::<TrainConfig>(config).is_err());
    let wire = serde_json::json!({"project_id":ProjectId::new(),"config":TrainConfig::default(),"device":{"policy":"auto"},"resume":null,"batch_mode":"unknown"});
    assert!(serde_json::from_value::<TrainRequest>(wire).is_err());
    assert_eq!(serde_json::to_value(BatchMode::Fixed).unwrap(), "fixed");
    assert_eq!(serde_json::to_value(BatchMode::Auto).unwrap(), "auto");
}
