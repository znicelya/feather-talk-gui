use super::BenchmarkReport;
use feathertalk_core::*;

/// Compare reports collected for the same workload. Callers reading a cache
/// must also call `reusable_for` against current device and workload identities.
pub fn select_backend(policy: &DevicePolicy, reports: &[BenchmarkReport]) -> Result<DeviceRequest> {
    if let DevicePolicy::Explicit(request) = policy {
        return Ok(request.clone());
    }
    let no_candidate = || {
        AppError::new(
            ErrorCode::UnsupportedBackend,
            "benchmark-policy",
            "No validated same-batch benchmark candidate",
        )
    };
    let first = reports.first().ok_or_else(no_candidate)?;
    if reports
        .iter()
        .any(|report| report.workload != first.workload)
    {
        return Err(AppError::new(
            ErrorCode::InvalidConfig,
            "benchmark-policy",
            "Benchmark candidates must use an identical workload and engine build",
        ));
    }
    reports
        .iter()
        .filter_map(|report| report.warm_score().map(|score| (score, report)))
        .min_by_key(|(score, report)| (*score, &report.device.id))
        .map(|(_, report)| DeviceRequest {
            backend: report.device.backend,
            device_id: report.device.id.clone(),
        })
        .ok_or_else(no_candidate)
}
