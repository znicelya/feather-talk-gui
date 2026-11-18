//! Reverify explicit runtime inputs before accepting control requests.
use feathertalk_core::*;
use std::{collections::HashSet, fs};
pub(crate) fn devices(c: &WorkerContext) -> Result<Vec<DeviceDescriptor>> {
    let devices = feathertalk_engine::probe_devices()?
        .into_iter()
        .filter(|d| match c.target.backend.as_str() {
            "cpu" => matches!(d.backend, BackendKind::Cpu | BackendKind::Flex),
            "cuda" => d.backend == BackendKind::Cuda,
            "vulkan" => d.backend == BackendKind::Vulkan,
            "metal" => d.backend == BackendKind::Metal,
            _ => false,
        })
        .collect::<Vec<_>>();
    if devices.is_empty() {
        return Err(AppError::new(
            ErrorCode::UnsupportedBackend,
            "context",
            "No device is available for this worker/probe target",
        ));
    }
    Ok(devices)
}
pub(crate) fn validate_context(c: &WorkerContext) -> Result<()> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    if c.protocol_version != PROTOCOL_VERSION || c.projects.len() != 1 {
        return Err(protocol_error(
            "Context requires protocol v1 and one explicit project",
        ));
    }
    for b in &c.projects {
        if !b.directory.is_absolute()
            || fs::canonicalize(&b.directory).map_err(protocol_error)? != b.directory
        {
            return Err(protocol_error("Project root must be canonical"));
        }
        let project: ProjectManifest = serde_json::from_slice(
            &fs::read(b.directory.join("project.json")).map_err(protocol_error)?,
        )
        .map_err(protocol_error)?;
        if project.id != b.project_id {
            return Err(protocol_error("Bootstrap project ID differs from manifest"));
        }
    }
    let backend_ok = match c.target.backend.as_str() {
        "cpu" => cfg!(feature = "flex"),
        "cuda" => cfg!(feature = "cuda"),
        "vulkan" => cfg!(feature = "vulkan"),
        "metal" => cfg!(feature = "metal"),
        _ => false,
    };
    if c.target.os != std::env::consts::OS
        || c.target.architecture != std::env::consts::ARCH
        || !backend_ok
    {
        return Err(AppError::new(
            ErrorCode::UnsupportedBackend,
            "context",
            "Worker build does not support the requested target/backend",
        ));
    }
    let mut verified = HashSet::new();
    for component in &c.components {
        if component.pin.id.is_empty() || component.pin.version.is_empty() {
            return Err(protocol_error("Missing resource pin"));
        }
        if component.target.as_ref().is_some_and(|t| {
            t.os != c.target.os
                || t.architecture != c.target.architecture
                || (t.backend != "any" && t.backend != c.target.backend)
        }) {
            return Err(protocol_error(
                "Component target differs from worker target",
            ));
        }

        if !component.root.is_absolute()
            || fs::canonicalize(&component.root).map_err(protocol_error)? != component.root
        {
            return Err(protocol_error("Component root must be canonical"));
        }
        if component.files.is_empty() {
            return Err(protocol_error("Empty component inventory"));
        }
        for file in &component.files {
            let path = fs::canonicalize(&file.path).map_err(protocol_error)?;
            if !file.path.is_absolute() || !path.starts_with(&component.root) {
                return Err(protocol_error("Component file escapes root"));
            }
            let mut f = fs::File::open(&path).map_err(protocol_error)?;
            let mut hash = Sha256::new();
            let mut buf = [0; 65536];
            loop {
                let n = f.read(&mut buf).map_err(protocol_error)?;
                if n == 0 {
                    break;
                }
                hash.update(&buf[..n]);
            }
            if format!("{:x}", hash.finalize()) != file.sha256 {
                return Err(AppError::new(
                    ErrorCode::ResourceCorrupt,
                    "context",
                    "Runtime component hash mismatch",
                ));
            }
            verified.insert(path);
        }
    }
    let probe = fs::canonicalize(&c.probe_runner).map_err(protocol_error)?;
    if !c.components.iter().any(|component| {
        component.target.as_ref() == Some(&c.target)
            && component
                .files
                .iter()
                .any(|f| fs::canonicalize(&f.path).ok().as_ref() == Some(&probe))
    }) {
        return Err(protocol_error(
            "Probe is not bound to a matching target/backend component",
        ));
    }
    if let Some(root) = &c.cuda_root {
        if c.target.backend != "cuda"
            || !root.is_absolute()
            || !verified.contains(
                &fs::canonicalize(root.join("nvvm/libdevice/libdevice.10.bc"))
                    .map_err(protocol_error)?,
            )
        {
            return Err(protocol_error("CUDA runtime is outside verified inventory"));
        }
    } else if c.target.backend == "cuda" {
        return Err(protocol_error(
            "CUDA worker requires explicit verified libdevice root",
        ));
    }
    for path in [
        &c.ffmpeg,
        &c.ffprobe,
        &c.probe_runner,
        &c.assets.join("manifest.json"),
    ] {
        if !path.is_absolute()
            || !verified.contains(&fs::canonicalize(path).map_err(protocol_error)?)
        {
            return Err(protocol_error("Runtime path is outside verified inventory"));
        }
    }
    Ok(())
}

pub(crate) fn normalize_generation(request: &mut GenerateRequest) -> Result<()> {
    if !request.audio_path.is_absolute() || !request.output_path.is_absolute() {
        return Err(protocol_error(
            "Generation audio and destination must be absolute",
        ));
    }
    request.audio_path = fs::canonicalize(&request.audio_path).map_err(protocol_error)?;
    if !request.audio_path.is_file() {
        return Err(protocol_error("Audio must be a regular file"));
    }
    let name = request
        .output_path
        .file_name()
        .ok_or_else(|| protocol_error("Missing output filename"))?;
    let parent = request
        .output_path
        .parent()
        .ok_or_else(|| protocol_error("Missing output parent"))?;
    request.output_path = fs::canonicalize(parent).map_err(protocol_error)?.join(name);
    if fs::symlink_metadata(&request.output_path)
        .is_ok_and(|m| !m.is_file() || m.file_type().is_symlink())
    {
        return Err(protocol_error("Output must be a regular file destination"));
    }
    Ok(())
}
