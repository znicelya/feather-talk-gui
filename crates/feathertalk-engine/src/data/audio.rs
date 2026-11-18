use crate::models::feather_hubert::FeatherHubert;
use burn::tensor::{Tensor, TensorData};
use feathertalk_core::*;

pub struct AudioFeatures {
    pub identity: ModelIdentity,
    pub video_frames: usize,
    pub values: Vec<f32>,
}
fn invalid(message: &str) -> AppError {
    let mut e = AppError::new(ErrorCode::NoAudio, "audio-features", message);
    e.actions
        .push("Provide finite mono audio at 16 kHz with at least 400 samples (25 ms).".into());
    e
}
pub fn normalize_waveform(samples: &[f32]) -> Result<Vec<f32>> {
    if samples.is_empty() || samples.iter().any(|v| !v.is_finite()) {
        return Err(invalid("Audio is empty or contains nonfinite samples"));
    }
    let mean = samples.iter().map(|&v| v as f64).sum::<f64>() / samples.len() as f64;
    let variance = samples
        .iter()
        .map(|&v| (v as f64 - mean).powi(2))
        .sum::<f64>()
        / samples.len() as f64;
    let scale = (variance + 1e-7).sqrt();
    Ok(samples
        .iter()
        .map(|&v| ((v as f64 - mean) / scale) as f32)
        .collect())
}
pub fn extract_audio_features(model: &FeatherHubert, samples_16k: &[f32]) -> Result<AudioFeatures> {
    extract_audio_features_cancellable(model, samples_16k, &Cancellation::new())
}
/// Cancellation is observed between bounded neural chunks (a running kernel is not interrupted).
pub fn extract_audio_features_cancellable(
    model: &FeatherHubert,
    samples_16k: &[f32],
    cancel: &Cancellation,
) -> Result<AudioFeatures> {
    if samples_16k.len() < 400 {
        return Err(invalid("Audio is too short to produce a HuBERT token"));
    }
    let samples = normalize_waveform(samples_16k)?;
    let expected = (samples.len() - 80) / 320;
    let chunk = 320000;
    let mut values = Vec::with_capacity(expected * 1024);
    for i in 0..=samples.len() / chunk {
        crate::media::ffmpeg::check(cancel)?;
        let start = i * chunk;
        let end = (start + chunk + 80).min(samples.len());
        if end - start < 400 {
            continue;
        }
        let waveform = Tensor::<2>::from_data(
            TensorData::new(samples[start..end].to_vec(), [1, end - start]),
            model.device(),
        );
        // Synchronous host transfer releases this chunk's intermediates before the next.
        let output = model
            .forward(waveform)
            .into_data()
            .try_to_vec::<f32>()
            .map_err(|e| invalid(&format!("Cannot read audio features: {e:?}")))?;
        values.extend(output);
    }
    values.resize(expected * 1024, 0.0);
    values.truncate((expected / 2) * 2 * 1024);
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("Model produced nonfinite audio features"));
    }
    Ok(AudioFeatures {
        identity: model.identity().clone(),
        video_frames: expected / 2,
        values,
    })
}
