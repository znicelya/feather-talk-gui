//! Shared assertion invoked by training_resume's real full-objective execution.
//! Keeping it in that execution avoids repeating expensive CPU VGG backward work.
pub fn assert_full_objective_decreased(
    first: &feathertalk_core::LossScalars,
    second: &feathertalk_core::LossScalars,
) {
    first.validate().unwrap();
    second.validate().unwrap();
    assert!(
        second.total < first.total,
        "full objective did not decrease: {} -> {}",
        first.total,
        second.total
    );
}
#[cfg(test)]
#[allow(dead_code)]
#[path = "training_fixture.rs"]
mod fixture;
#[test]
fn restored_prefetch_and_direct_next_batch_match() {
    use feathertalk_engine::data::sampler::PairSampler;
    let (_, prepared) = fixture::fixture();
    let mut original = PairSampler::new(&prepared, 42).unwrap();
    original.next_batch(1).unwrap().unwrap();
    original.advance_epoch().unwrap();
    let state = original.snapshot();
    let mut direct = PairSampler::restore(&prepared, state.clone()).unwrap();
    let mut prefetch = PairSampler::restore(&prepared, state)
        .unwrap()
        .prefetch(1, 2, feathertalk_core::Cancellation::new())
        .unwrap();
    assert_eq!(
        direct.next_batch(1).unwrap(),
        prefetch.next_batch().unwrap()
    );
    assert_eq!(direct.snapshot(), prefetch.snapshot());
}
