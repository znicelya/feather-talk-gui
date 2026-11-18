use feathertalk_core::*;
use feathertalk_engine::{
    data::prepared::file_hash,
    media::{
        MediaTools,
        ffmpeg::{canonicalize, read_waveform},
        probe::probe,
    },
};
use std::{
    path::{Path, PathBuf},
    process::Command,
};
fn root(name: &str) -> PathBuf {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!(
        "../../.artifacts/task8-media/{name}-{}",
        ProjectId::new()
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}
fn synth(tools: &MediaTools, out: &Path, audio: bool, video_delay: f64, audio_delay: f64) {
    let mut command = Command::new(&tools.ffmpeg);
    command.args([
        "-nostdin",
        "-v",
        "error",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=64x48:rate=30:duration=1.2",
    ]);
    if audio {
        command.args([
            "-f",
            "lavfi",
            "-i",
            "aevalsrc=0.25|0.75:s=48000:d=1.2",
            "-filter_complex",
            &format!("[0:v]setpts=PTS+{video_delay}/TB[v];[1:a]asetpts=PTS+{audio_delay}/TB[a]"),
            "-map",
            "[v]",
            "-map",
            "[a]",
            "-c:a",
            "pcm_f32le",
        ]);
    }
    command.args(["-c:v", "ffv1"]).arg(out);
    let result = command.output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
}
#[test]
fn canonical_media_first_channel_and_offsets() {
    let tools = MediaTools::from_env().expect("media tests require FT_FFMPEG and FT_FFPROBE");
    for (name, vd, ad) in [
        ("aligned", 0., 0.),
        ("delayed-audio", 0., 0.2),
        ("early-audio", 0.2, 0.),
    ] {
        let root = root(name);
        let source = root.join("source.mkv");
        synth(&tools, &source, true, vd, ad);
        let hash = file_hash(&source).unwrap();
        let out = root.join("canonical");
        canonicalize(&source, &out, &tools, &Cancellation::new()).unwrap();
        assert_eq!(hash, file_hash(&source).unwrap());
        let info = probe(&out.join("video.mkv"), &out, &tools, &Cancellation::new()).unwrap();
        assert_eq!(info.fps, "25/1");
        let samples = read_waveform(&out.join("audio.wav")).unwrap();
        let expected = ((1.2 + ad - vd) * 16000.) as usize;
        assert!(
            samples.len().abs_diff(expected) <= 16,
            "{name}: {} expected {expected}",
            samples.len()
        );
        let skip = ((ad - vd).max(0.) * 16000.) as usize;
        if skip > 0 {
            assert!(samples[..skip - 16].iter().all(|x| x.abs() < 1e-6));
        }
        assert!(
            samples[skip + 64..samples.len() - 64]
                .iter()
                .all(|x| (*x - 0.25).abs() < 1e-5),
            "Must select first channel, not stereo downmix"
        );
    }
}
#[test]
fn missing_audio_and_cancel_are_actionable() {
    let tools = MediaTools::from_env().unwrap();
    let root = root("noaudio");
    let source = root.join("source.mkv");
    synth(&tools, &source, false, 0., 0.);
    let out = root.join("canonical");
    let e = canonicalize(&source, &out, &tools, &Cancellation::new()).unwrap_err();
    assert_eq!(e.code, ErrorCode::NoAudio);
    assert!(!e.actions.is_empty());
    let cancel = Cancellation::new();
    cancel.cancel();
    assert_eq!(
        canonicalize(&source, &out, &tools, &cancel)
            .unwrap_err()
            .code,
        ErrorCode::Cancelled
    );
}
#[test]
fn real_models_attach_frame_time_and_preserve_old_cache_on_no_face() {
    use feathertalk_engine::{data::prepare_project, open_device};
    struct Observer(std::sync::Mutex<Vec<EventPayload>>);
    impl TaskObserver for Observer {
        fn emit(&self, event: EventPayload) -> Result<()> {
            self.0.lock().unwrap().push(event);
            Ok(())
        }
    }
    let tools = MediaTools::from_env().unwrap();
    let root = root("real-no-face");
    let source = root.join("source.mkv");
    synth(&tools, &source, true, 0., 0.);
    let project = ProjectManifest {
        schema_version: 1,
        id: ProjectId::new(),
        name: "Synthetic no face".into(),
        source_video: SourceVideo {
            path: source.clone(),
            sha256: file_hash(&source).unwrap(),
            size_bytes: std::fs::metadata(&source).unwrap().len(),
        },
        preprocessing: None,
        model_identity: None,
        checkpoints: vec![],
        metrics_path: PathBuf::from("metrics.jsonl"),
    };
    std::fs::write(
        root.join("project.json"),
        serde_json::to_vec(&project).unwrap(),
    )
    .unwrap();
    std::fs::create_dir_all(root.join("cache/prepared")).unwrap();
    std::fs::write(
        root.join("cache/prepared/index.json"),
        b"previous committed index",
    )
    .unwrap();
    let backend = std::env::var("FT_TEST_BACKEND").unwrap_or("flex".into());
    let kind = match backend.as_str() {
        "vulkan" => BackendKind::Vulkan,
        "cuda" => BackendKind::Cuda,
        _ => BackendKind::Flex,
    };
    let device = open_device(&DeviceRequest {
        backend: kind,
        device_id: format!("{backend}:0"),
    })
    .unwrap();
    eprintln!("Actual preparation device {:?}", device.descriptor);
    let observer = Observer(Default::default());
    let assets = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.artifacts/assets");
    let error = match prepare_project(
        &root,
        &project,
        &assets,
        &tools,
        &device,
        &observer,
        &Cancellation::new(),
    ) {
        Err(e) => e,
        Ok(_) => panic!("Synthetic test pattern unexpectedly detected as a face"),
    };
    assert_eq!(error.code, ErrorCode::NoFace, "{error:?}");
    assert!(error.evidence.iter().any(|s| {
        s.contains("canonical_frame=0; timestamp_seconds=0.000000; stage=frame-landmarks")
    }));
    assert_eq!(
        std::fs::read(root.join("cache/prepared/index.json")).unwrap(),
        b"previous committed index"
    );
    let lock = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(root.join("cache/prepared/prepare.lock"))
        .unwrap();
    fs2::FileExt::try_lock_exclusive(&lock).unwrap();
    assert!(
        std::fs::read_dir(root.join("cache/prepared"))
            .unwrap()
            .all(|p| !p
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with("pending-"))
    );
    assert!(
        observer
            .0
            .lock()
            .unwrap()
            .iter()
            .any(|p| matches!(p,EventPayload::Progress{stage,..} if stage=="canonical-media"))
    );
}
