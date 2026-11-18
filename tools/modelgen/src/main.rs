use burn_onnx::{LoadStrategy, ModelGen};
use std::{error::Error, path::PathBuf};
fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let (mut input, mut source, mut assets) = (None, None, None);
    while let Some(flag) = args.next() {
        let value = PathBuf::from(args.next().ok_or("Missing flag value")?);
        match flag.as_str() {
            "--input" => input = Some(value),
            "--source-output" => source = Some(value),
            "--assets-output" => assets = Some(value),
            _ => return Err(format!("Unknown flag {flag}").into()),
        }
    }
    let input = input.ok_or("--input required")?;
    let source = source.ok_or("--source-output required")?;
    let assets = assets.ok_or("--assets-output required")?;
    let staging = assets.join("modelgen-staging");
    std::fs::create_dir_all(&staging)?;
    let staged_input = staging.join("scrfd.onnx");
    let python = std::env::var("FT_MODELGEN_PYTHON").unwrap_or_else(|_| "python".into());
    let helper = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("prepare_scrfd.py");
    let status = std::process::Command::new(&python)
        .arg(helper)
        .arg(&input)
        .arg(&staged_input)
        .status()?;
    if !status.success() {
        return Err("SCRFD graph proof/equivalence failed".into());
    }
    let original = assets.join("architecture.onnx");
    if !original.exists() || std::fs::canonicalize(&input)? != std::fs::canonicalize(&original)? {
        std::fs::copy(&input, &original)?;
    }

    ModelGen::new()
        .input(staged_input.to_str().ok_or("Non-UTF8 input")?)
        .out_dir(staging.to_str().ok_or("Non-UTF8 output")?)
        .partition(false)
        .load_strategy(LoadStrategy::File)
        .run_from_cli();
    std::fs::create_dir_all(source.parent().ok_or("Source parent required")?)?;
    let code = std::fs::read_to_string(staging.join("scrfd.rs"))?;
    let start = code
        .find("impl Default for Model")
        .ok_or("Missing generated Default boundary")?;
    let end = code[start..]
        .find("impl Model {\n    #[allow(unused_variables)]")
        .ok_or("Missing generated new boundary")?
        + start;
    let code = format!("{}{}", &code[..start], &code[end..]);
    let code = code
        .lines()
        .skip(1)
        .collect::<Vec<_>>()
        .join("\n")
        .replace("use burn::tensor::Bytes;\n", "")
        .replace("use burn_store::BurnpackStore;\n", "")
        .replace("use burn_store::ModuleSnapshot;\n", "");
    std::fs::write(
        &source,
        format!(
            "// Generated with burn-onnx 0.22.0 from proven equivalent fixed-640 SCRFD.\n{code}"
        ),
    )?;
    let status = std::process::Command::new("rustfmt")
        .arg("--edition")
        .arg("2024")
        .arg(&source)
        .status()?;
    if !status.success() {
        return Err("Generated source formatting failed".into());
    }
    // Frozen inference artifacts do not need randomly generated training ParamIds.
    // Re-encode through the official container API with stable path-derived IDs.
    use sha2::{Digest, Sha256};
    let reader = burn_pack::Reader::from_file(staging.join("scrfd.bpk"))?;
    let metadata = reader.metadata().clone();
    let scalars = reader.scalars().clone();
    let mut ids = std::collections::BTreeSet::new();
    let mut entries = Vec::new();
    let mut payload_digest = Sha256::new();
    for tensor in reader.into_tensors()? {
        let (name, dtype, shape, old_id, bytes) = tensor.into_parts()?;
        payload_digest.update(name.as_bytes());
        payload_digest.update(format!("{dtype:?}{shape:?}").as_bytes());
        payload_digest.update(&bytes[..]);
        let digest = Sha256::digest(name.as_bytes());
        let id = u64::from_le_bytes(digest[..8].try_into()?);
        if !ids.insert(id) {
            return Err("Generated tensor ParamId collision".into());
        }
        entries.push(burn_pack::Tensor::new(
            name,
            dtype,
            shape,
            old_id.map(|_| id),
            bytes,
        ));
    }
    println!(
        "SCRFD tensor name/shape/dtype/value digest: {:x}",
        payload_digest.finalize()
    );
    let mut writer = burn_pack::Writer::new(entries).overwrite(true);
    for (key, value) in metadata {
        writer = writer.with_metadata(&key, &value);
    }
    for (key, value) in scalars {
        writer = writer.with_scalar(&key, value);
    }
    writer.write_to_file(assets.join("model.bpk"))?;
    let helper = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("finalize_assets.py");
    let status = std::process::Command::new(&python)
        .arg(helper)
        .arg(&assets)
        .status()?;
    if !status.success() {
        return Err("Asset finalization failed".into());
    }
    Ok(())
}
