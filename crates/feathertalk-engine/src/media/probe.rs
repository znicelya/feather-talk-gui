use super::{
    MediaTools,
    ffmpeg::{error, run},
};
use feathertalk_core::*;
use std::{path::Path, process::Command};
pub struct MediaInfo {
    pub width: usize,
    pub height: usize,
    pub has_audio: bool,
    pub fps: String,
}
pub fn probe(
    source: &Path,
    directory: &Path,
    tools: &MediaTools,
    cancel: &Cancellation,
) -> Result<MediaInfo> {
    let mut cmd = Command::new(&tools.ffprobe);
    cmd.args(["-v", "error", "-show_streams", "-of", "json"])
        .arg(source);
    let v: serde_json::Value =
        serde_json::from_slice(&run(&mut cmd, directory, cancel)?).map_err(error)?;
    let streams = v["streams"]
        .as_array()
        .ok_or_else(|| error("No media streams"))?;
    let video = streams
        .iter()
        .find(|s| s["codec_type"] == "video")
        .ok_or_else(|| error("No video stream"))?;
    let width = video["width"].as_u64().unwrap_or(0) as usize;
    let height = video["height"].as_u64().unwrap_or(0) as usize;
    if width == 0 || height == 0 || width > 16384 || height > 16384 {
        return Err(error("Unsupported video dimensions"));
    }
    Ok(MediaInfo {
        width,
        height,
        has_audio: streams.iter().any(|s| s["codec_type"] == "audio"),
        fps: video["avg_frame_rate"].as_str().unwrap_or("").into(),
    })
}
pub fn first_video_time(
    source: &Path,
    directory: &Path,
    tools: &MediaTools,
    cancel: &Cancellation,
) -> Result<f64> {
    let mut cmd = Command::new(&tools.ffprobe);
    cmd.args([
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-read_intervals",
        "%+#1",
        "-show_frames",
        "-show_entries",
        "frame=best_effort_timestamp_time",
        "-of",
        "json",
    ])
    .arg(source);
    let v: serde_json::Value =
        serde_json::from_slice(&run(&mut cmd, directory, cancel)?).map_err(error)?;
    let value = v["frames"][0]["best_effort_timestamp_time"]
        .as_str()
        .ok_or_else(|| error("Cannot determine first presented video timestamp"))?
        .parse::<f64>()
        .map_err(error)?;
    if !value.is_finite() {
        return Err(error("Invalid video timestamp"));
    }
    Ok(value)
}
