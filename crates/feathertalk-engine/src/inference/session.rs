//! Bounded offline boundaries. Whole-utterance HuBERT normalization is not streaming.
use crate::data::landmarks::BgrFrame;
use crate::media::ffmpeg::error;
use feathertalk_core::*;
use std::{
    io::{Read, Seek, SeekFrom},
    path::Path,
};
pub const AUDIO_BATCH_LIMIT: usize = 16000;
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimeBase {
    pub numerator: u32,
    pub denominator: u32,
}
pub const AUDIO_TIME_BASE: TimeBase = TimeBase {
    numerator: 1,
    denominator: 16000,
};
pub const VIDEO_TIME_BASE: TimeBase = TimeBase {
    numerator: 1,
    denominator: 25,
};
pub struct TimestampedAudio {
    pub pts: u64,
    pub time_base: TimeBase,
    pub samples: Vec<f32>,
}
pub struct TimestampedFrame {
    pub pts: u64,
    pub time_base: TimeBase,
    pub frame: BgrFrame,
}
pub trait AudioSource {
    fn session(&self) -> &str;
    fn reset(&mut self) -> Result<()>;
    fn read(&mut self, max_samples: usize) -> Result<Option<TimestampedAudio>>;
    fn finish(&mut self) -> Result<()>;
}
pub trait FrameSink {
    type Output;
    fn write(&mut self, frame: TimestampedFrame, cancel: &Cancellation) -> Result<()>;
    fn finish(&mut self, cancel: &Cancellation) -> Result<Self::Output>;
    fn abort(&mut self) -> Result<()>;
}
/// Input is canonical mono f32le 16k PCM, created by the file adapter.
pub struct FileAudioSource {
    file: std::fs::File,
    id: String,
    pts: u64,
    finished: bool,
}
impl FileAudioSource {
    pub fn open(path: &Path) -> Result<Self> {
        let file = std::fs::File::open(path).map_err(error)?;
        if file.metadata().map_err(error)?.len() % 4 != 0 {
            return Err(error("Invalid PCM length"));
        }
        Ok(Self {
            file,
            id: path.to_string_lossy().into(),
            pts: 0,
            finished: false,
        })
    }
}
impl AudioSource for FileAudioSource {
    fn session(&self) -> &str {
        &self.id
    }
    fn reset(&mut self) -> Result<()> {
        self.file.seek(SeekFrom::Start(0)).map_err(error)?;
        self.pts = 0;
        self.finished = false;
        Ok(())
    }
    fn read(&mut self, max_samples: usize) -> Result<Option<TimestampedAudio>> {
        if self.finished || max_samples == 0 || max_samples > AUDIO_BATCH_LIMIT {
            return Err(error("Audio session closed or batch exceeds 16000 samples"));
        }
        let mut bytes = vec![];
        (&mut self.file)
            .take((max_samples * 4) as u64)
            .read_to_end(&mut bytes)
            .map_err(error)?;
        if bytes.is_empty() {
            return Ok(None);
        }
        if bytes.len() % 4 != 0 {
            return Err(error("Truncated PCM sample"));
        }
        let samples = bytes
            .chunks_exact(4)
            .map(|v| f32::from_le_bytes(v.try_into().unwrap()))
            .collect::<Vec<_>>();
        if samples.iter().any(|v| !v.is_finite()) {
            return Err(error("Nonfinite PCM"));
        }
        let pts = self.pts;
        self.pts += samples.len() as u64;
        Ok(Some(TimestampedAudio {
            pts,
            time_base: AUDIO_TIME_BASE,
            samples,
        }))
    }
    fn finish(&mut self) -> Result<()> {
        self.finished = true;
        Ok(())
    }
}
