use super::{
    MediaTools,
    probe::{first_video_time, probe},
};
use feathertalk_core::*;
use std::{
    fs,
    io::Read,
    path::Path,
    process::{Command, Stdio},
    thread,
    time::Duration,
};
pub(crate) fn error(e: impl ToString) -> AppError {
    let message = e.to_string();
    let disk = message.contains("No space left")
        || message.contains("not enough space")
        || message.contains("os error 112")
        || message.contains("os error 28");
    let mut error = AppError::new(
        if disk {
            ErrorCode::DiskFull
        } else {
            ErrorCode::EncodeFailed
        },
        "media",
        message,
    );
    if disk {
        error.actions.push("Free space on the project drive; canonical video and temporary raw BGR frames require duration-scaled disk storage".into());
    }
    error
}
pub(crate) fn check(cancel: &Cancellation) -> Result<()> {
    if cancel.is_cancelled() {
        Err(AppError::new(
            ErrorCode::Cancelled,
            "prepare",
            "Preparation cancelled",
        ))
    } else {
        Ok(())
    }
}
/// Child stdout/stderr are files: no pipe deadlock or unbounded captured output.
/// All files belong to the caller's staging directory. Child is reaped on every exit.
pub(crate) fn run(
    command: &mut Command,
    directory: &Path,
    cancel: &Cancellation,
) -> Result<Vec<u8>> {
    check(cancel)?;
    let stdout = directory.join("command.stdout");
    let stderr = directory.join("command.stderr");
    let mut child = command
        .stdin(Stdio::null())
        .stdout(fs::File::create(&stdout).map_err(error)?)
        .stderr(fs::File::create(&stderr).map_err(error)?)
        .spawn()
        .map_err(error)?;
    let status = loop {
        if let Err(e) = check(cancel) {
            let _ = child.kill();
            let _ = child.wait();
            return Err(e);
        }
        match child.try_wait() {
            Ok(Some(s)) => break s,
            Ok(None) => thread::sleep(Duration::from_millis(10)),
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error(e));
            }
        }
    };
    if !status.success() {
        let mut bytes = vec![];
        fs::File::open(stderr)
            .map_err(error)?
            .take(65536)
            .read_to_end(&mut bytes)
            .map_err(error)?;
        return Err(error(format!(
            "Media tool exited {status}: {}",
            String::from_utf8_lossy(&bytes)
        )));
    }
    let mut bytes = vec![];
    fs::File::open(stdout)
        .map_err(error)?
        .take(4 * 1024 * 1024)
        .read_to_end(&mut bytes)
        .map_err(error)?;
    Ok(bytes)
}
pub fn canonicalize(
    source: &Path,
    directory: &Path,
    tools: &MediaTools,
    cancel: &Cancellation,
) -> Result<()> {
    fs::create_dir_all(directory).map_err(error)?;
    let info = probe(source, directory, tools, cancel)?;
    if !info.has_audio {
        let mut e = AppError::new(
            ErrorCode::NoAudio,
            "media-probe",
            "Source video has no audio track",
        );
        e.actions
            .push("Choose a speaking video with an audio track".into());
        return Err(e);
    }
    let origin = first_video_time(source, directory, tools, cancel)?;
    let mut video = Command::new(&tools.ffmpeg);
    video
        .args(["-nostdin", "-y", "-v", "error", "-copyts", "-i"])
        .arg(source)
        .args([
            "-map",
            "0:v:0",
            "-an",
            "-vf",
            &format!("setpts=PTS-({origin:.12})/TB,fps=25"),
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "bgr0",
        ])
        .arg(directory.join("video.mkv"));
    run(&mut video, directory, cancel)?;
    let mut audio = Command::new(&tools.ffmpeg);
    audio
        .args(["-nostdin", "-y", "-v", "error", "-copyts", "-i"])
        .arg(source)
        .args([
            "-map",
            "0:a:0",
            "-vn",
            "-af",
            &format!("pan=mono|c0=c0,asetpts=PTS-({origin:.12})/TB,aresample=16000:first_pts=0"),
            "-c:a",
            "pcm_f32le",
        ])
        .arg(directory.join("audio.wav"));
    run(&mut audio, directory, cancel)?;
    Ok(())
}
pub(crate) fn decode_frames(
    directory: &Path,
    tools: &MediaTools,
    cancel: &Cancellation,
) -> Result<std::fs::File> {
    let path = directory.join("decoded.bgr");
    let mut command = Command::new(&tools.ffmpeg);
    command
        .args(["-nostdin", "-y", "-v", "error", "-i"])
        .arg(directory.join("video.mkv"))
        .args([
            "-map", "0:v:0", "-an", "-pix_fmt", "bgr24", "-f", "rawvideo",
        ])
        .arg(&path);
    run(&mut command, directory, cancel)?;
    fs::File::open(path).map_err(error)
}
pub fn read_waveform(path: &Path) -> Result<Vec<f32>> {
    let mut file = fs::File::open(path).map_err(error)?;
    let mut header = [0; 12];
    file.read_exact(&mut header).map_err(error)?;
    if &header[..4] != b"RIFF" || &header[8..] != b"WAVE" {
        return Err(error("Expected RIFF WAV"));
    }
    let mut valid = false;
    loop {
        let mut chunk = [0; 8];
        file.read_exact(&mut chunk).map_err(error)?;
        let n = u32::from_le_bytes(chunk[4..].try_into().unwrap()) as usize;
        if &chunk[..4] == b"fmt " {
            if n > 4096 {
                return Err(error("Invalid WAV format"));
            }
            let mut fmt = vec![0; n];
            file.read_exact(&mut fmt).map_err(error)?;
            valid = n >= 16
                && u16::from_le_bytes(fmt[2..4].try_into().unwrap()) == 1
                && u32::from_le_bytes(fmt[4..8].try_into().unwrap()) == 16000
                && u16::from_le_bytes(fmt[14..16].try_into().unwrap()) == 32
                && (fmt[..2] == [3, 0]
                    || (n >= 40 && fmt[..2] == [254, 255] && fmt[24..26] == [3, 0]));
        } else if &chunk[..4] == b"data" {
            if !valid || n % 4 != 0 {
                return Err(error("Expected mono 16 kHz float32 WAV"));
            }
            let mut values = Vec::with_capacity(n / 4);
            let mut bytes = [0; 4];
            for _ in 0..n / 4 {
                file.read_exact(&mut bytes).map_err(error)?;
                let v = f32::from_le_bytes(bytes);
                if !v.is_finite() {
                    return Err(error("Nonfinite waveform"));
                }
                values.push(v);
            }
            return Ok(values);
        } else {
            std::io::copy(&mut (&mut file).take(n as u64), &mut std::io::sink()).map_err(error)?;
        }
        if n % 2 != 0 {
            let mut pad = [0];
            file.read_exact(&mut pad).map_err(error)?;
        }
    }
}
