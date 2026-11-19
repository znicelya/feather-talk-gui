use feathertalk_core::{BackendKind, DeviceDescriptor, DevicePolicy, DeviceRequest, Precision};
use feathertalk_engine::benchmark::{
    BenchmarkReport, BenchmarkSample, WorkloadIdentity, select_backend,
};

fn report(backend: BackendKind, warm_ms: u64) -> BenchmarkReport {
    let device = DeviceDescriptor {
        id: format!("{backend:?}:0"),
        name: "Physical test adapter".into(),
        vendor: Some("NVIDIA".into()),
        driver: Some("test-driver-1".into()),
        backend,
        memory_bytes: None,
    };
    let sample = |training_step_ms| BenchmarkSample {
        training_step_ms,
        training_pairs: 4,
        generation_ms: training_step_ms * 2,
        video_frames: 10,
        ..Default::default()
    };
    BenchmarkReport {
        schema_version: 1,
        workload: WorkloadIdentity {
            engine_sha256: "a".repeat(64),
            source_sha256: "b".repeat(64),
            audio_sha256: "c".repeat(64),
            checkpoint_sha256: "d".repeat(64),
            prepared_sha256: "e".repeat(64),
            assets_sha256: "f".repeat(64),
            ffmpeg_sha256: "0".repeat(64),
            ffprobe_sha256: "1".repeat(64),
            reference_sha256: "2".repeat(64),
            os: "windows".into(),
            architecture: "x86_64".into(),
            precision: Precision::F32,
            requested_batch: 4,
            prefetch_capacity: 2,
            sample_loss: true,
        },
        device,
        validation_passed: true,
        cold_runs: vec![sample(1_000_000)],
        warm_runs: vec![sample(warm_ms), sample(warm_ms + 1), sample(warm_ms * 6)],
        ..Default::default()
    }
}

#[test]
fn selects_only_validated_median_warm_results_and_keeps_manual_choice() {
    let mut cuda = report(BackendKind::Cuda, 1);
    cuda.validation_passed = false;
    let vulkan = report(BackendKind::Vulkan, 12);
    let mut cpu = report(BackendKind::Flex, 20);
    cpu.cold_runs[0].training_step_ms = 1;
    assert_eq!(
        select_backend(&DevicePolicy::Auto, &[cuda.clone(), cpu, vulkan])
            .unwrap()
            .backend,
        BackendKind::Vulkan
    );
    let explicit = DeviceRequest {
        backend: BackendKind::Cuda,
        device_id: cuda.device.id.clone(),
    };
    assert_eq!(
        select_backend(&DevicePolicy::Explicit(explicit.clone()), &[]).unwrap(),
        explicit
    );
}

#[test]
fn refuses_to_rank_different_workloads_tail_batches_or_unfinished_reports() {
    let vulkan = report(BackendKind::Vulkan, 10);
    let mut cuda = report(BackendKind::Cuda, 1);
    cuda.workload.requested_batch = 8;
    assert!(select_backend(&DevicePolicy::Auto, &[vulkan.clone(), cuda.clone()]).is_err());
    cuda.workload = vulkan.workload.clone();
    cuda.warm_runs[0].training_pairs = 2;
    assert_eq!(
        select_backend(&DevicePolicy::Auto, &[vulkan.clone(), cuda.clone()])
            .unwrap()
            .backend,
        BackendKind::Vulkan
    );
    cuda.warm_runs.pop();
    assert!(select_backend(&DevicePolicy::Auto, &[cuda]).is_err());
}

#[test]
fn stale_or_unknown_driver_and_changed_model_cannot_reuse_cached_report() {
    let report = report(BackendKind::Cuda, 3);
    assert!(report.reusable_for(&report.workload, &report.device));
    let mut driver_update = report.device.clone();
    driver_update.driver = Some("test-driver-2".into());
    assert!(!report.reusable_for(&report.workload, &driver_update));
    driver_update.driver = None;
    assert!(!report.reusable_for(&report.workload, &driver_update));
    let mut no_driver = report.clone();
    no_driver.device = driver_update;
    assert!(!no_driver.reusable_for(&report.workload, &no_driver.device));
    let mut next_checkpoint = report.workload.clone();
    next_checkpoint.checkpoint_sha256 = "3".repeat(64);
    assert!(!report.reusable_for(&next_checkpoint, &report.device));
}
