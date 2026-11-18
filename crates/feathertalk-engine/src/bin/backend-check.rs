use feathertalk_core::*;
use feathertalk_engine::*;
fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |name: &str| args.windows(2).find(|p| p[0] == name).map(|p| p[1].clone());
    let backend = match arg("--backend").as_deref().unwrap_or("flex") {
        "flex" => BackendKind::Flex,
        "cpu" => BackendKind::Cpu,
        "vulkan" => BackendKind::Vulkan,
        "cuda" => BackendKind::Cuda,
        "metal" => BackendKind::Metal,
        _ => return Err("Unknown backend".into()),
    };
    let request = DeviceRequest {
        backend,
        device_id: arg("--device")
            .unwrap_or_else(|| format!("{}:0", format!("{backend:?}").to_lowercase())),
    };
    let device = open_device(&request)?;
    eprintln!("Executing on {}", device.descriptor.name);
    let report = run_operator_checks(
        &device,
        std::path::Path::new(&arg("--reference").unwrap_or(".artifacts/reference".into())),
    )?;
    let json = serde_json::to_string_pretty(&report)?;
    if let Some(output) = arg("--output") {
        let path = std::path::Path::new(&output);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, &json)?;
    } else {
        println!("{json}");
    }
    if !report.checks.iter().all(|c| c.passed) {
        return Err("Operator parity failed; see report".into());
    }
    Ok(())
}
