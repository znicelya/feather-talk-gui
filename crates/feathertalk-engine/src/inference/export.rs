use super::{
    process::{Staging, run},
    session::*,
};
use crate::media::{
    MediaTools,
    ffmpeg::{check, error},
};
use feathertalk_core::*;
use std::{
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
};
/// Bounded RAM, disk-backed frame spool. H.264 4:4:4 preserves odd canvas sizes.
pub struct FileFrameSink {
    stage: Option<Staging>,
    raw: Option<fs::File>,
    output: PathBuf,
    audio: PathBuf,
    width: usize,
    height: usize,
    expected: u64,
    written: u64,
    tools: MediaTools,
}
impl FileFrameSink {
    pub fn new(
        output: &Path,
        audio: &Path,
        width: usize,
        height: usize,
        frames: u64,
        tools: &MediaTools,
    ) -> Result<Self> {
        if frames == 0
            || frames.checked_mul(640).is_none()
            || width == 0
            || height == 0
            || width > 16384
            || height > 16384
        {
            return Err(error(
                "Export requires nonzero feature frames and a supported original canvas",
            ));
        }
        let parent = output
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(Path::new("."));
        let stage = Staging::new(parent)?;
        let output = fs::canonicalize(parent).map_err(error)?.join(
            output
                .file_name()
                .ok_or_else(|| error("Missing output filename"))?,
        );
        let raw = fs::File::create(stage.0.join("frames.bgr")).map_err(error)?;
        Ok(Self {
            stage: Some(stage),
            raw: Some(raw),
            output,
            audio: audio.to_owned(),
            width,
            height,
            expected: frames,
            written: 0,
            tools: tools.clone(),
        })
    }
    fn finalize(&mut self, cancel: &Cancellation) -> Result<PathBuf> {
        check(cancel)?;
        if self.written != self.expected {
            return Err(error("Incomplete frame timeline"));
        }
        self.raw
            .take()
            .ok_or_else(|| error("Export already closed"))?
            .sync_all()
            .map_err(error)?;
        let stage = &self
            .stage
            .as_ref()
            .ok_or_else(|| error("Export aborted"))?
            .0;
        let encoded = stage.join("final.mp4");
        let mut command = Command::new(&self.tools.ffmpeg);
        command
            .args([
                "-nostdin",
                "-y",
                "-v",
                "error",
                "-f",
                "rawvideo",
                "-pixel_format",
                "bgr24",
                "-video_size",
            ])
            .arg(format!("{}x{}", self.width, self.height))
            .args(["-framerate", "25", "-i"])
            .arg(stage.join("frames.bgr"))
            .args(["-f", "f32le", "-ar", "16000", "-ac", "1", "-i"])
            .arg(&self.audio)
            .args([
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "fast", "-pix_fmt",
            ])
            .arg(if self.width % 2 == 0 && self.height % 2 == 0 {
                "yuv420p"
            } else {
                "yuv444p"
            })
            .args(["-c:a", "aac", "-af"])
            .arg(format!(
                "apad=whole_len={},atrim=end_sample={},asetpts=N/SR/TB",
                self.expected * 640,
                self.expected * 640
            ))
            .args(["-movflags", "+faststart", "-f", "mp4"])
            .arg(&encoded);
        run(&mut command, cancel)?;
        let mut probe = Command::new(&self.tools.ffprobe);
        probe
            .args([
                "-v",
                "error",
                "-count_frames",
                "-show_streams",
                "-of",
                "json",
            ])
            .arg(&encoded);
        let value: serde_json::Value =
            serde_json::from_slice(&run(&mut probe, cancel)?).map_err(error)?;
        let streams = value["streams"]
            .as_array()
            .ok_or_else(|| error("Missing encoded streams"))?;
        let v = streams
            .iter()
            .find(|v| v["codec_type"] == "video")
            .ok_or_else(|| error("Missing video"))?;
        let a = streams
            .iter()
            .find(|v| v["codec_type"] == "audio")
            .ok_or_else(|| error("Missing audio"))?;
        let duration = a["duration"]
            .as_str()
            .and_then(|s| s.parse::<f64>().ok())
            .unwrap_or(f64::NAN);
        if v["codec_name"] != "h264"
            || v["width"] != self.width
            || v["height"] != self.height
            || v["avg_frame_rate"] != "25/1"
            || v["nb_read_frames"] != self.expected.to_string()
            || a["codec_name"] != "aac"
            || !duration.is_finite()
            || (duration - self.expected as f64 / 25.).abs() > 0.065
        {
            return Err(error(format!(
                "Finalized export failed timeline/codec/canvas verification: {value}"
            )));
        }
        fs::OpenOptions::new()
            .write(true)
            .open(&encoded)
            .map_err(error)?
            .sync_all()
            .map_err(error)?;
        check(cancel)?;
        // std::fs::rename uses replacement semantics on Windows and POSIX, same filesystem.
        fs::rename(&encoded, &self.output).map_err(error)?;
        self.stage.take();
        Ok(self.output.clone())
    }
}
impl FrameSink for FileFrameSink {
    type Output = PathBuf;
    fn write(&mut self, frame: TimestampedFrame, cancel: &Cancellation) -> Result<()> {
        check(cancel)?;
        let f = frame.frame;
        if frame.pts != self.written
            || frame.time_base != VIDEO_TIME_BASE
            || self.written >= self.expected
            || f.width != self.width
            || f.height != self.height
            || f.stride < f.width * 3
            || f.data.len() != f.stride * f.height
        {
            return Err(error("Invalid frame canvas or integer timeline"));
        }
        let raw = self.raw.as_mut().ok_or_else(|| error("Export is closed"))?;
        for row in 0..f.height {
            raw.write_all(&f.data[row * f.stride..row * f.stride + f.width * 3])
                .map_err(error)?;
        }
        self.written += 1;
        Ok(())
    }
    fn finish(&mut self, cancel: &Cancellation) -> Result<PathBuf> {
        let result = self.finalize(cancel);
        if result.is_err() {
            self.abort()?;
        }
        result
    }
    fn abort(&mut self) -> Result<()> {
        self.raw.take();
        self.stage.take();
        Ok(())
    }
}
impl Drop for FileFrameSink {
    fn drop(&mut self) {
        let _ = self.abort();
    }
}
