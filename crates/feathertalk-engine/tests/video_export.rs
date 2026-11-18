use feathertalk_core::*;
use feathertalk_engine::{
    data::landmarks::BgrFrame,
    inference::{export::FileFrameSink, session::*},
    media::MediaTools,
};
use std::{fs, path::PathBuf};
fn tools() -> MediaTools {
    let bin = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../.superpowers/toolchain/ffmpeg/ffmpeg-9.0.2-essentials_build/bin");
    MediaTools {
        ffmpeg: bin.join("ffmpeg.exe"),
        ffprobe: bin.join("ffprobe.exe"),
    }
}
fn root() -> PathBuf {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../.artifacts/task10-export-tests")
        .join(format!("中文 空格 {}", TrainingRunId::new()));
    fs::create_dir_all(&root).unwrap();
    root
}
fn frame(pts: u64, w: usize, h: usize) -> TimestampedFrame {
    TimestampedFrame {
        pts,
        time_base: VIDEO_TIME_BASE,
        frame: BgrFrame {
            width: w,
            height: h,
            stride: w * 3,
            data: vec![90; w * h * 3],
        },
    }
}
#[test]
fn fixed_timeline_canvas_and_atomic_preservation() {
    for (f, samples, w, h) in [
        (1, 1, 33, 35),
        (24, 640, 32, 34),
        (24, 15360, 32, 34),
        (24, 20000, 32, 34),
    ] {
        let root = root();
        let output = root.join("结果 video.mp4");
        let audio = root.join("audio.f32");
        fs::write(&audio, vec![0u8; samples * 4]).unwrap();
        fs::write(&output, b"prior successful bytes").unwrap();
        let mut sink = FileFrameSink::new(&output, &audio, w, h, f, &tools()).unwrap();
        for k in 0..f {
            sink.write(frame(k, w, h), &Cancellation::new()).unwrap();
        }
        sink.finish(&Cancellation::new()).unwrap();
        let probe = std::process::Command::new(tools().ffprobe)
            .args([
                "-v",
                "error",
                "-count_frames",
                "-show_streams",
                "-of",
                "json",
            ])
            .arg(&output)
            .output()
            .unwrap();
        assert!(probe.status.success());
        let v: serde_json::Value = serde_json::from_slice(&probe.stdout).unwrap();
        assert_eq!(v["streams"][0]["codec_name"], "h264");
        assert_eq!(v["streams"][0]["width"], w);
        assert_eq!(v["streams"][0]["height"], h);
        assert_eq!(v["streams"][0]["avg_frame_rate"], "25/1");
        assert_eq!(v["streams"][0]["nb_read_frames"], f.to_string());
        assert_eq!(v["streams"][1]["codec_name"], "aac");
        let duration = v["streams"][1]["duration"]
            .as_str()
            .unwrap()
            .parse::<f64>()
            .unwrap();
        assert!((duration - f as f64 / 25.).abs() < 0.065);
        let prior = fs::read(&output).unwrap();
        let cancel = Cancellation::new();
        let mut sink = FileFrameSink::new(&output, &audio, w, h, f, &tools()).unwrap();
        sink.write(frame(0, w, h), &cancel).unwrap();
        cancel.cancel();
        assert!(sink.finish(&cancel).is_err());
        drop(sink);
        assert_eq!(fs::read(&output).unwrap(), prior);
        let mut broken = tools();
        broken.ffmpeg = broken.ffprobe.clone();
        let mut sink = FileFrameSink::new(&output, &audio, w, h, 1, &broken).unwrap();
        sink.write(frame(0, w, h), &Cancellation::new()).unwrap();
        let e = sink.finish(&Cancellation::new()).unwrap_err();
        assert!(!e.message.is_empty());
        drop(sink);
        assert_eq!(fs::read(&output).unwrap(), prior);
        assert!(FileFrameSink::new(&output, &audio, w, h, 0, &tools()).is_err());
        assert!(!fs::read_dir(&root).unwrap().any(|e| {
            e.unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".export-")
        }));
    }
}
#[test]
fn file_audio_session_bounds_reset_and_integer_pts() {
    let root = root();
    let path = root.join("pcm.f32");
    fs::write(
        &path,
        (0..16003)
            .flat_map(|i| (i as f32).to_le_bytes())
            .collect::<Vec<_>>(),
    )
    .unwrap();
    let mut source = FileAudioSource::open(&path).unwrap();
    assert!(source.read(16001).is_err());
    assert!(source.read(0).is_err());
    let first = source.read(16000).unwrap().unwrap();
    assert_eq!(first.pts, 0);
    assert_eq!(first.time_base, AUDIO_TIME_BASE);
    let last = source.read(16000).unwrap().unwrap();
    assert_eq!(last.pts, 16000);
    assert_eq!(last.samples, vec![16000., 16001., 16002.]);
    assert!(source.read(1).unwrap().is_none());
    source.finish().unwrap();
    assert!(source.read(1).is_err());
    source.reset().unwrap();
    assert_eq!(source.read(1).unwrap().unwrap().samples, vec![0.]);
}
#[test]
fn probe_failure_and_publication_failure_preserve_target() {
    let root = root();
    let audio = root.join("pcm.f32");
    fs::write(&audio, vec![0u8; 640 * 4]).unwrap();
    let output = root.join("prior.mp4");
    fs::write(&output, b"old successful output").unwrap();
    let mut tools = tools();
    tools.ffprobe = tools.ffmpeg.clone();
    let mut sink = FileFrameSink::new(&output, &audio, 32, 32, 1, &tools).unwrap();
    sink.write(frame(0, 32, 32), &Cancellation::new()).unwrap();
    assert!(sink.finish(&Cancellation::new()).is_err());
    assert_eq!(fs::read(&output).unwrap(), b"old successful output");
    let destination = root.join("occupied.mp4");
    fs::create_dir(&destination).unwrap();
    fs::write(destination.join("unrelated"), b"retain").unwrap();
    let mut sink = FileFrameSink::new(&destination, &audio, 32, 32, 1, &self::tools()).unwrap();
    sink.write(frame(0, 32, 32), &Cancellation::new()).unwrap();
    assert!(sink.finish(&Cancellation::new()).is_err());
    assert_eq!(fs::read(destination.join("unrelated")).unwrap(), b"retain");
    let mut sink = FileFrameSink::new(&output, &audio, 32, 32, 2, &self::tools()).unwrap();
    assert!(sink.write(frame(1, 32, 32), &Cancellation::new()).is_err());
    sink.write(frame(0, 32, 32), &Cancellation::new()).unwrap();
    assert!(sink.finish(&Cancellation::new()).is_err());
    assert_eq!(fs::read(&output).unwrap(), b"old successful output");
}
