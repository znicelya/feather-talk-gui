use feathertalk_core::*;
use feathertalk_engine::*;
use std::{path::PathBuf, sync::OnceLock};
fn report() -> &'static OperatorReport {
    static REPORT: OnceLock<OperatorReport> = OnceLock::new();
    REPORT.get_or_init(|| {
        let request = DeviceRequest {
            backend: BackendKind::Flex,
            device_id: "flex:0".into(),
        };
        let device = open_device(&request).unwrap();
        let root = std::env::var_os("FT_REFERENCE_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|| {
                PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.artifacts/reference")
            });
        let report = run_operator_checks(&device, &root).unwrap();
        assert_eq!(report.device.id, request.device_id);
        report
    })
}
fn check(prefixes: &[&str]) {
    let checks: Vec<_> = report()
        .checks
        .iter()
        .filter(|c| prefixes.iter().any(|p| c.name.starts_with(p)))
        .collect();
    assert!(!checks.is_empty());
    assert!(checks.iter().all(|c| c.passed), "{checks:#?}");
}
#[test]
fn resize_forward_and_backward_match() {
    check(&["resize"]);
}
#[test]
fn normalization_train_eval_match() {
    check(&["batch_norm", "group_norm"]);
}
#[test]
fn grouped_dilated_conv_gradients_match() {
    check(&["depthwise", "group_dilated"]);
}
#[test]
fn avg_pool_padding_and_attention_match() {
    check(&["avg_pool", "spatial_audio"]);
}

#[test]
fn resize_singleton_and_identity_have_correct_gradients() {
    use burn::tensor::{Device, Tensor};
    use feathertalk_engine::ops::resize::resize_bilinear_align_corners;
    let d = Device::flex().autodiff();
    let x = Tensor::<4>::from_data([[[[1.0, 2.0], [3.0, 4.0]]]], &d).require_grad();
    let singleton = resize_bilinear_align_corners(x.clone(), [1, 1]);
    assert_eq!(
        singleton.clone().into_data().try_to_vec::<f32>().unwrap(),
        vec![1.0]
    );
    let g = singleton.sum().backward();
    assert_eq!(
        x.grad(&g).unwrap().into_data().try_to_vec::<f32>().unwrap(),
        vec![1.0, 0.0, 0.0, 0.0]
    );
    let identity = resize_bilinear_align_corners(x.clone(), [2, 2]);
    assert_eq!(identity.into_data(), x.into_data());
}
#[test]
fn invalid_device_and_missing_reference_are_errors() {
    assert!(
        open_device(&DeviceRequest {
            backend: BackendKind::Flex,
            device_id: "flex:999".into()
        })
        .is_err()
    );
    let d = open_device(&DeviceRequest {
        backend: BackendKind::Flex,
        device_id: "flex:0".into(),
    })
    .unwrap();
    assert!(run_operator_checks(&d, std::path::Path::new("definitely-missing-reference")).is_err());
}

#[test]
fn cpu_alias_and_explicit_policy_resolve_without_fallback() {
    let request = DeviceRequest {
        backend: BackendKind::Cpu,
        device_id: "cpu:0".into(),
    };
    let cpu = open_device(&request).unwrap();
    assert_eq!(cpu.device, burn::tensor::Device::flex());
    let descriptors = probe_devices().unwrap();
    assert_eq!(
        resolve_device(&DevicePolicy::Explicit(request.clone()), &descriptors).unwrap(),
        request
    );
    assert_eq!(
        resolve_device(&DevicePolicy::Auto, &descriptors)
            .unwrap()
            .device_id,
        "flex:0"
    );
    let absent = DeviceRequest {
        backend: BackendKind::Cuda,
        device_id: "cuda:999".into(),
    };
    assert!(resolve_device(&DevicePolicy::Explicit(absent), &descriptors).is_err());
}
