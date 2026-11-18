//! Validate every tensor before Burn's mutating loader can run.
use super::state::invalid;
use burn::tensor::DType;
use burn_pack::{Reader, Tensor};
use feathertalk_core::Result;
use std::collections::{BTreeMap, BTreeSet};

pub(crate) fn tensors(bytes: &[u8]) -> Result<Vec<Tensor>> {
    Reader::from_bytes(burn_pack::Bytes::from_bytes_vec(bytes.to_vec()))
        .map_err(invalid)?
        .into_tensors()
        .map_err(invalid)
}
pub(crate) fn validate_model(bytes: &[u8], template: &[u8]) -> Result<Vec<Tensor>> {
    let actual = tensors(bytes)?;
    let expected = tensors(template)?;
    if actual.len() != 447 || expected.len() != actual.len() {
        return Err(invalid("Incomplete model inventory"));
    }
    let mut ids = BTreeSet::new();
    for (a, e) in actual.iter().zip(&expected) {
        if a.name != e.name
            || a.shape != e.shape
            || a.dtype != e.dtype
            || a.param_id.is_none()
            || !ids.insert(a.param_id.unwrap())
        {
            return Err(invalid(format!("Invalid model record {}", a.name)));
        }
        let data = a.to_bytes().map_err(invalid)?;
        match a.dtype {
            DType::F32 => {
                if data
                    .chunks_exact(4)
                    .any(|v| !f32::from_le_bytes(v.try_into().unwrap()).is_finite())
                {
                    return Err(invalid("Nonfinite model state"));
                }
            }
            DType::I64 => {
                if data
                    .chunks_exact(8)
                    .any(|v| i64::from_le_bytes(v.try_into().unwrap()) < 0)
                {
                    return Err(invalid("Negative BN counter"));
                }
            }
            _ => return Err(invalid("Unsupported model dtype")),
        }
    }
    Ok(actual)
}
pub(crate) fn validate_adam(bytes: &[u8], model: &[Tensor], step: u64) -> Result<()> {
    let reader =
        Reader::from_bytes(burn_pack::Bytes::from_bytes_vec(bytes.to_vec())).map_err(invalid)?;
    let scalars = reader.scalars().clone();
    let paths = reader.metadata().clone();
    let moments = reader.into_tensors().map_err(invalid)?;
    let trainable: BTreeMap<_, _> = model
        .iter()
        .filter(|p| !p.name.contains(".state."))
        .map(|p| (p.param_id.unwrap(), p))
        .collect();
    if trainable.len() != 252 {
        return Err(invalid("Trainable model inventory mismatch"));
    }
    if step == 0 {
        if !moments.is_empty() || !scalars.is_empty() || !paths.is_empty() {
            return Err(invalid("Pristine checkpoint has optimizer history"));
        }
        return Ok(());
    }
    if moments.len() != 504 {
        return Err(invalid("Incomplete Adam moment inventory"));
    }
    let mut expected_names = BTreeSet::new();
    let mut expected_scalars = BTreeSet::new();
    for (id, p) in &trainable {
        for suffix in ["momentum.moment_1", "momentum.moment_2"] {
            expected_names.insert(format!("{id}.{suffix}"));
        }
        for (suffix, value) in [
            ("__rank", p.shape.num_dims() as u64),
            ("momentum.time", step),
        ] {
            let key = format!("{id}.{suffix}");
            if scalars.get(&key).and_then(|s| u64::try_from(*s).ok()) != Some(value) {
                return Err(invalid(format!("Invalid Adam scalar {key}")));
            }
            expected_scalars.insert(key);
        }
    }
    if scalars.keys().cloned().collect::<BTreeSet<_>>() != expected_scalars {
        return Err(invalid("Foreign Adam scalar"));
    }
    for (id, path) in paths {
        let id: u64 = id.parse().map_err(invalid)?;
        if trainable.get(&id).is_none_or(|p| p.name != path) {
            return Err(invalid("Foreign Adam parameter path"));
        }
    }
    for m in moments {
        let p = m
            .param_id
            .and_then(|id| trainable.get(&id))
            .ok_or_else(|| invalid("Foreign Adam ParamId"))?;
        if !expected_names.remove(&m.name)
            || !m.name.starts_with(&format!("{}.", p.param_id.unwrap()))
            || m.shape != p.shape
            || m.dtype != DType::F32
        {
            return Err(invalid("Invalid Adam moment shape/name/dtype"));
        }
        if m.to_bytes()
            .map_err(invalid)?
            .chunks_exact(4)
            .any(|v| !f32::from_le_bytes(v.try_into().unwrap()).is_finite())
        {
            return Err(invalid("Nonfinite Adam moment"));
        }
    }
    Ok(())
}
pub(crate) fn equal_pack(a: &[u8], b: &[u8]) -> Result<()> {
    let ar = Reader::from_bytes(burn_pack::Bytes::from_bytes_vec(a.to_vec())).map_err(invalid)?;
    let br = Reader::from_bytes(burn_pack::Bytes::from_bytes_vec(b.to_vec())).map_err(invalid)?;
    if ar.scalars() != br.scalars() || ar.metadata() != br.metadata() {
        return Err(invalid("Loaded scalar state differs"));
    }
    let aa = ar.into_tensors().map_err(invalid)?;
    let bb = br.into_tensors().map_err(invalid)?;
    if aa.len() != bb.len() {
        return Err(invalid("Loaded inventory differs"));
    }
    for (a, b) in aa.iter().zip(&bb) {
        if a.name != b.name
            || a.shape != b.shape
            || a.dtype != b.dtype
            || a.param_id != b.param_id
            || a.to_bytes().map_err(invalid)? != b.to_bytes().map_err(invalid)?
        {
            return Err(invalid("Loaded tensor state differs"));
        }
    }
    Ok(())
}
