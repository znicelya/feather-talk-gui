use burn::tensor::{Device, Tensor};
use feathertalk_core::*;
pub struct ExecutionDevice {
    pub device: Device,
    pub descriptor: DeviceDescriptor,
}
fn available(backend: BackendKind) -> Vec<Device> {
    match backend {
        #[cfg(feature = "flex")]
        BackendKind::Flex | BackendKind::Cpu => vec![Device::flex()],
        #[cfg(feature = "vulkan")]
        BackendKind::Vulkan => Device::enumerate(burn::tensor::DeviceType::Vulkan).into_vec(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => Device::enumerate(burn::tensor::DeviceType::Cuda).into_vec(),
        #[cfg(feature = "metal")]
        BackendKind::Metal => Device::enumerate(burn::tensor::DeviceType::Metal).into_vec(),
        _ => vec![],
    }
}
fn descriptor(backend: BackendKind, index: usize, device: &Device) -> DeviceDescriptor {
    let label = format!("{backend:?}").to_lowercase();
    let name = if matches!(backend, BackendKind::Flex | BackendKind::Cpu) {
        "Host CPU (Burn Flex)".into()
    } else {
        #[cfg(any(feature = "vulkan", feature = "cuda", feature = "metal"))]
        {
            device
                .identity()
                .map(|i| format!("{i:?}"))
                .unwrap_or_else(|| format!("{device:?}"))
        }
        #[cfg(not(any(feature = "vulkan", feature = "cuda", feature = "metal")))]
        {
            format!("{device:?}")
        }
    };
    DeviceDescriptor {
        id: format!("{label}:{index}"),
        name,
        vendor: None,
        driver: None,
        backend,
        memory_bytes: None,
    }
}
pub fn probe_devices() -> Result<Vec<DeviceDescriptor>> {
    let mut result = vec![];
    for backend in [
        BackendKind::Flex,
        BackendKind::Cpu,
        BackendKind::Vulkan,
        BackendKind::Cuda,
        BackendKind::Metal,
    ] {
        for (i, d) in available(backend).iter().enumerate() {
            result.push(descriptor(backend, i, d));
        }
    }
    Ok(result)
}
pub fn resolve_device(
    policy: &DevicePolicy,
    descriptors: &[DeviceDescriptor],
) -> Result<DeviceRequest> {
    let d = match policy {
        DevicePolicy::Explicit(r) => descriptors
            .iter()
            .find(|d| d.id == r.device_id && d.backend == r.backend),
        DevicePolicy::Auto => descriptors
            .iter()
            .find(|d| d.backend == BackendKind::Flex)
            .or_else(|| descriptors.first()),
    }
    .ok_or_else(|| {
        AppError::new(
            ErrorCode::UnsupportedBackend,
            "device",
            "Requested device is unavailable",
        )
    })?;
    Ok(DeviceRequest {
        backend: d.backend,
        device_id: d.id.clone(),
    })
}
pub fn open_device(request: &DeviceRequest) -> Result<ExecutionDevice> {
    for (i, device) in available(request.backend).into_iter().enumerate() {
        let descriptor = descriptor(request.backend, i, &device);
        if descriptor.id == request.device_id {
            let values = (Tensor::<1>::ones([1], &device) * 2.0)
                .into_data()
                .try_to_vec::<f32>()
                .map_err(|e| AppError::new(ErrorCode::DeviceInit, "device", format!("{e:?}")))?;
            if values != [2.0] {
                return Err(AppError::new(
                    ErrorCode::DeviceInit,
                    "device",
                    "Execution smoke check failed",
                ));
            }
            return Ok(ExecutionDevice { device, descriptor });
        }
    }
    Err(AppError::new(
        ErrorCode::UnsupportedBackend,
        "device",
        "Requested backend/device is not available",
    ))
}
