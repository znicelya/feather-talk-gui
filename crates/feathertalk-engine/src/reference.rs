use burn::tensor::{Device, Tensor, TensorData};
use feathertalk_core::*;
use safetensors::{Dtype, SafeTensors};
use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, path::Path};
#[derive(Deserialize)]
pub struct Case {
    pub file: String,
    pub sha256: String,
    pub configuration: serde_json::Value,
    pub tolerance: Tolerance,
    #[serde(default)]
    pub tensor_tolerances: BTreeMap<String, Tolerance>,
}
#[derive(Deserialize)]
pub struct Tolerance {
    pub atol: f64,
    pub rtol: f64,
}
#[derive(Deserialize)]
pub struct Manifest {
    pub cases: BTreeMap<String, Case>,
}
pub struct Reference {
    pub case: Case,
    pub values: BTreeMap<String, (Vec<usize>, Vec<f32>)>,
}
pub fn error(message: impl ToString) -> AppError {
    AppError::new(
        ErrorCode::ResourceCorrupt,
        "operator-reference",
        message.to_string(),
    )
}
impl Reference {
    pub fn load(root: &Path, name: &str) -> Result<Self> {
        let mut manifest: Manifest =
            serde_json::from_slice(&std::fs::read(root.join("manifest.json")).map_err(error)?)
                .map_err(error)?;
        let case = manifest
            .cases
            .remove(name)
            .ok_or_else(|| error(format!("Missing case {name}")))?;
        let bytes = std::fs::read(root.join(&case.file)).map_err(error)?;
        if format!("{:x}", Sha256::digest(&bytes)) != case.sha256 {
            return Err(error("Reference SHA256 mismatch"));
        }
        let tensors = SafeTensors::deserialize(&bytes).map_err(error)?;
        let mut values = BTreeMap::new();
        for (name, t) in tensors.tensors() {
            let data = match t.dtype() {
                Dtype::F32 => t
                    .data()
                    .chunks_exact(4)
                    .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
                    .collect(),
                Dtype::I64 => t
                    .data()
                    .chunks_exact(8)
                    .map(|v| i64::from_le_bytes(v.try_into().unwrap()) as f32)
                    .collect(),
                _ => return Err(error("Unsupported reference dtype")),
            };
            values.insert(name, (t.shape().to_vec(), data));
        }
        Ok(Self { case, values })
    }
    pub fn tensor<const D: usize>(&self, name: &str, device: &Device) -> Tensor<D> {
        let (shape, data) = &self.values[name];
        let shape: Vec<usize> = if shape.is_empty() {
            vec![1]
        } else {
            shape.clone()
        };
        assert_eq!(shape.len(), D, "{name} rank");
        Tensor::from_data(TensorData::new(data.clone(), shape), device)
    }
    pub fn flat(&self, name: &str, device: &Device) -> Tensor<1> {
        let data = &self.values[name].1;
        Tensor::from_data(TensorData::new(data.clone(), [data.len()]), device)
    }
}
