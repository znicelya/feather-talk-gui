//! Versioned private process protocol; the only accepted success is a complete step.
use super::{
    batch_probe::ProbeContext,
    checkpoint::{MaterializedSnapshot, read_project},
    state::invalid,
};
use crate::{
    ExecutionDevice,
    data::prepared::{PreparedProject, reopen_generation},
};
use feathertalk_core::*;
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{Read, Write},
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::Duration,
};
const VERSION: u32 = 1;
fn rewind_disposable_sampler(sampler: &mut crate::data::sampler::SamplerState) {
    sampler.cursor = 0;
}
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Binding {
    version: u32,
    id: TrainingRunId,
    candidate: u32,
    effective: u32,
    device: DeviceRequest,
    step: u64,
    vgg_identity: ModelIdentity,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    binding: Binding,
    project: PathBuf,
    assets: PathBuf,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Response {
    binding: Binding,
    outcome: std::result::Result<CompletedStep, AppError>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CompletedStep {
    losses: LossScalars,
    global_step: u64,
    pairs: u32,
    device: DeviceRequest,
}
struct OwnedDirectory(PathBuf);
impl Drop for OwnedDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn atomic_json(path: &Path, value: &impl Serialize) -> Result<()> {
    let temporary = path.with_extension("tmp");
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)
        .map_err(invalid)?;
    file.write_all(&serde_json::to_vec(value).map_err(invalid)?)
        .map_err(invalid)?;
    file.sync_all().map_err(invalid)?;
    drop(file);
    fs::rename(temporary, path).map_err(invalid)
}
// Burn 0.22.1 server_err -> ExecutionError Display -> state::invalid.
// CubeCL 0.11.4 IoError::OutOfMemory only. Unknown panics stay errors.
fn allocation_failure(error: &AppError, backend: BackendKind) -> bool {
    if !matches!(
        backend,
        BackendKind::Cuda | BackendKind::Vulkan | BackendKind::Metal
    ) || error.stage != "training-checkpoint"
    {
        return false;
    }
    let prefix = "An error happened during execution\nCaused by:\n  An IO error happened\nCaused by:\n  out of device memory allocating ";
    error
        .message
        .strip_prefix(prefix)
        .and_then(|s| s.split_once(" bytes\n"))
        .is_some_and(|(size, _)| {
            !size.is_empty()
                && size.bytes().all(|c| c.is_ascii_digit())
                && size.parse::<u64>().is_ok()
        })
}
pub(super) fn candidate(
    snapshot: &MaterializedSnapshot,
    root: &Path,
    prepared: &PreparedProject,
    device: &ExecutionDevice,
    context: &ProbeContext<'_>,
    candidate: u32,
) -> Result<u32> {
    if !context.runtime.probe_runner.is_absolute() || !context.runtime.probe_runner.is_file() {
        return Err(invalid(
            "Inject an absolute verified probe runner executable path",
        ));
    }
    let root = fs::canonicalize(root).map_err(invalid)?;
    let dir = root.join(format!(".probe-{}", TrainingRunId::new()));
    fs::create_dir(&dir).map_err(invalid)?;
    let canonical = fs::canonicalize(&dir).map_err(invalid)?;
    if !canonical.starts_with(&root) {
        return Err(invalid("Probe directory escapes project"));
    }
    let owned = OwnedDirectory(canonical);
    snapshot.write_private(&owned.0.join("snapshot"))?;
    let selected = crate::verified_assets::VerifiedAssets::load(context.vgg_assets)?
        .manifest
        .identity;
    crate::models::vgg::validate_identity(&selected)?;
    if snapshot
        .metadata()
        .vgg_identity
        .as_ref()
        .is_some_and(|id| id != &selected)
    {
        return Err(invalid("Probe assets differ from selected VGG dependency"));
    }
    let binding = Binding {
        version: VERSION,
        id: TrainingRunId::new(),
        candidate,
        effective: candidate.min(prepared.usable_frames.saturating_sub(1) as u32),
        device: DeviceRequest {
            backend: device.descriptor.backend,
            device_id: device.descriptor.id.clone(),
        },
        step: snapshot.metadata().global_step,
        vgg_identity: selected,
    };
    let request = Request {
        binding: binding.clone(),
        project: root,
        assets: fs::canonicalize(context.vgg_assets).map_err(invalid)?,
    };
    let request_path = owned.0.join("request.json");
    atomic_json(&request_path, &request)?;
    let mut command = Command::new(&context.runtime.probe_runner);
    command
        .arg("--request")
        .arg(&request_path)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    #[cfg(test)]
    if let Some(mode) = SIMULATED.with(|m| m.borrow().clone()) {
        command = Command::new(std::env::current_exe().map_err(invalid)?);
        command
            .args([
                "--exact",
                "training::probe_transport::tests::simulated_child",
                "--ignored",
                "--nocapture",
            ])
            .env("FT_SIM_REQUEST", &request_path)
            .env("FT_SIM_MODE", mode)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::piped());
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    let mut child = command.spawn().map_err(invalid)?;
    let mut stderr = child.stderr.take().expect("piped stderr");
    let drain = thread::spawn(move || {
        let mut tail = Vec::new();
        let mut buf = [0; 4096];
        while let Ok(n) = stderr.read(&mut buf) {
            if n == 0 {
                break;
            }
            tail.extend_from_slice(&buf[..n]);
            if tail.len() > 16384 {
                tail.drain(..tail.len() - 16384);
            }
        }
        tail
    });
    let started = std::time::Instant::now();
    let status = loop {
        if started.elapsed() > Duration::from_secs(1800) {
            let _ = child.kill();
            let _ = child.wait();
            let _ = drain.join();
            return Err(invalid(
                "Probe exceeded 30 minute deadline; inspect device/runner and use a smaller Fixed batch",
            ));
        }
        if context.cancel.is_cancelled() {
            let _ = child.kill();
            let _ = child.wait();
            let _ = drain.join();
            return Err(AppError::new(
                ErrorCode::Cancelled,
                "batch-probe",
                "Cancelled isolated candidate",
            ));
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) => thread::sleep(Duration::from_millis(25)),
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                let _ = drain.join();
                return Err(invalid(e));
            }
        }
    };
    let tail = String::from_utf8_lossy(&drain.join().map_err(|_| invalid("stderr drain failed"))?)
        .into_owned();
    let mut error = AppError::new(
        ErrorCode::Internal,
        "batch-probe",
        format!(
            "Candidate {candidate} on {:?} exited {status}",
            binding.device
        ),
    );
    error.evidence.push(tail);
    error
        .actions
        .push("Check the verified runner/backend and logs; try a smaller Fixed batch.".into());
    if !status.success() {
        return Err(error);
    }
    let result_path = owned.0.join("result.json");
    if fs::metadata(&result_path)
        .map_err(|e| {
            error.evidence.push(e.to_string());
            error.clone()
        })?
        .len()
        > 65536
    {
        return Err(error);
    }
    let bytes = fs::read(result_path).map_err(|e| {
        error.evidence.push(e.to_string());
        error.clone()
    })?;
    let response: Response = serde_json::from_slice(&bytes).map_err(|e| {
        error.evidence.push(e.to_string());
        error.clone()
    })?;
    if response.binding != binding {
        error
            .evidence
            .push("Stale or mismatched result binding".into());
        return Err(error);
    }
    match response.outcome {
        Ok(step) => {
            if binding.step.checked_add(1) != Some(step.global_step)
                || step.pairs != binding.effective
                || step.device != binding.device
            {
                return Err(error);
            }
            step.losses.validate()?;
            Ok(step.pairs)
        }
        Err(mut e) => {
            if allocation_failure(&e, binding.device.backend) {
                e.code = ErrorCode::OutOfMemory;
            } else if e.code == ErrorCode::OutOfMemory {
                e.code = ErrorCode::Internal;
            }
            e.evidence.push(error.message);
            e.evidence.extend(error.evidence);
            e.actions.extend(error.actions);
            Err(e)
        }
    }
}
#[cfg(test)]
thread_local! {static SIMULATED:std::cell::RefCell<Option<String>>=const {std::cell::RefCell::new(None)};}
#[cfg(test)]
mod tests {
    use super::*;
    #[allow(dead_code)]
    mod shapes {
        use crate as feathertalk_engine;
        include!(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/tests/training_fixture.rs"
        ));
    }
    #[test]
    fn disposable_rewind_executes_capacity_instead_of_short_tail() {
        use crate::data::sampler::PairSampler;
        let (_, prepared) = shapes::fixture_frames(4);
        let mut sampler = PairSampler::new(&prepared, 42).unwrap();
        assert_eq!(sampler.next_batch(2).unwrap().unwrap().pairs.len(), 2);
        let parent = sampler.snapshot();
        assert_eq!(parent.permutation.len() - parent.cursor, 1);
        let mut disposable = parent.clone();
        rewind_disposable_sampler(&mut disposable);
        assert_eq!(disposable.reference_state, parent.reference_state);
        assert_eq!(disposable.shuffle_state, parent.shuffle_state);
        assert_eq!(disposable.reference_draws, parent.reference_draws);
        let mut probe = PairSampler::restore(&prepared, disposable).unwrap();
        assert_eq!(probe.next_batch(2).unwrap().unwrap().pairs.len(), 2);
        assert_eq!(sampler.snapshot(), parent);
        assert_eq!(sampler.next_batch(2).unwrap().unwrap().pairs.len(), 1);
        sampler.advance_epoch().unwrap();
        assert_eq!(sampler.snapshot().epoch, 1);
        assert_eq!(sampler.snapshot().cursor, 0);
    }
    use crate::training::{
        batch_probe::TrainingRuntime,
        checkpoint_tests::{device, fixture},
        state::TrainingState,
    };
    #[test]
    #[ignore = "Controlled subprocess fixture only; never neural execution evidence"]
    fn simulated_child() {
        let path = PathBuf::from(std::env::var_os("FT_SIM_REQUEST").unwrap());
        let dir = path.parent().unwrap();
        let request: Request = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        let mode = std::env::var("FT_SIM_MODE").unwrap();
        if mode == "missing" {
            return;
        }
        if mode == "malformed" {
            fs::write(dir.join("result.json"), b"not json").unwrap();
            return;
        }
        if mode == "crash" {
            eprint!("{}", "simulated stderr flood\n".repeat(10000));
            eprintln!("unknown crash, not an allocation failure");
            std::process::exit(17);
        }
        if mode == "wait" {
            fs::write(request.project.join("child-started"), b"simulated").unwrap();
            loop {
                eprint!("{}", "x".repeat(4096));
                thread::sleep(Duration::from_millis(5));
            }
        }
        let mut binding = request.binding;
        if mode == "stale" {
            binding.id = TrainingRunId::new();
        }
        if mode == "wrongcandidate" {
            binding.candidate += 1;
        }
        if mode == "wrongversion" {
            binding.version += 1;
        }
        let outcome = if mode == "oom"
            || [
                "wrongstep",
                "wrongcount",
                "wrongdevice",
                "wrongcandidate",
                "wrongversion",
                "nonzero-valid",
                "nonfinite",
            ]
            .contains(&mode.as_str())
        {
            if mode == "oom" && binding.candidate > 1 {
                Err(AppError::new(
                    ErrorCode::ResourceCorrupt,
                    "training-checkpoint",
                    "An error happened during execution\nCaused by:\n  An IO error happened\nCaused by:\n  out of device memory allocating 123 bytes\nsimulated diagnostic",
                ))
            } else {
                Ok(CompletedStep {
                    losses: LossScalars {
                        full: 1.,
                        mouth: 1.,
                        temporal: 1.,
                        temporal_mouth: 1.,
                        perceptual: 1.,
                        mouth_gradient: 1.,
                        total: 6.,
                    },
                    global_step: binding.step + if mode == "wrongstep" { 2 } else { 1 },
                    pairs: binding.effective + u32::from(mode == "wrongcount"),
                    device: if mode == "wrongdevice" {
                        DeviceRequest {
                            backend: BackendKind::Cpu,
                            device_id: "wrong".into(),
                        }
                    } else {
                        binding.device.clone()
                    },
                })
            }
        } else {
            Err(AppError::new(
                ErrorCode::Internal,
                "simulated-fixture",
                "simulated ordinary error",
            ))
        };
        let response = Response { binding, outcome };
        if mode == "nonfinite" {
            let mut value = serde_json::to_value(&response).unwrap();
            value["outcome"]["Ok"]["losses"]["total"] = serde_json::Value::Null;
            atomic_json(&dir.join("result.json"), &value).unwrap();
        } else {
            atomic_json(&dir.join("result.json"), &response).unwrap();
        }
        if mode == "nonzero-valid" {
            std::process::exit(17);
        }
    }
    fn assets() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../.artifacts/assets/vgg19")
    }
    fn request(prepared: &PreparedProject) -> TrainRequest {
        TrainRequest {
            project_id: prepared.project_id,
            batch_mode: BatchMode::Auto,
            config: TrainConfig {
                epochs: 1,
                batch_size: 1,
                ..Default::default()
            },
            device: DevicePolicy::Auto,
            resume: None,
        }
    }
    #[test]
    fn simulated_faults_reject_and_reap_owned_directory() {
        let (root, prepared) = fixture();
        let device = device();
        let state = TrainingState::fresh(&request(&prepared), &prepared, &device).unwrap();
        let snapshot = MaterializedSnapshot::capture(&state).unwrap();
        let runtime = TrainingRuntime {
            probe_runner: std::env::current_exe().unwrap(),
        };
        let assets = assets();
        let cancel = Cancellation::new();
        for mode in [
            "missing",
            "stale",
            "crash",
            "ordinary",
            "malformed",
            "wrongstep",
            "wrongcount",
            "wrongdevice",
            "wrongcandidate",
            "wrongversion",
            "nonzero-valid",
            "nonfinite",
        ] {
            SIMULATED.with(|m| *m.borrow_mut() = Some(mode.into()));
            let error = candidate(
                &snapshot,
                &root,
                &prepared,
                &device,
                &ProbeContext {
                    runtime: &runtime,
                    vgg_assets: &assets,
                    cancel: &cancel,
                },
                1,
            )
            .unwrap_err();
            assert_ne!(error.code, ErrorCode::OutOfMemory);
            if mode == "crash" {
                assert!(error.evidence[0].len() <= 16384);
                assert!(error.evidence[0].contains("unknown crash"));
            }
            assert!(!fs::read_dir(&root).unwrap().any(|e| {
                e.unwrap()
                    .file_name()
                    .to_string_lossy()
                    .starts_with(".probe-")
            }));
        }
        SIMULATED.with(|m| *m.borrow_mut() = None);
    }
    #[test]
    fn narrow_versioned_allocation_envelope() {
        let prefix = "An error happened during execution\nCaused by:\n  An IO error happened\nCaused by:\n  out of device memory allocating ";
        let e = AppError::new(
            ErrorCode::ResourceCorrupt,
            "training-checkpoint",
            format!("{prefix}123 bytes\ntrace"),
        );
        assert!(allocation_failure(&e, BackendKind::Cuda));
        assert!(!allocation_failure(&e, BackendKind::Flex));
        for message in [
            "out of memory",
            "out of device memory allocating 123 bytes\ntrace",
            &format!("{prefix}oops bytes\ntrace"),
            "can't allocate buffer of size: 123",
        ] {
            let mut e = e.clone();
            e.message = message.into();
            assert!(!allocation_failure(&e, BackendKind::Cuda));
        }
    }
    #[test]
    fn simulated_allocation_retry_never_exceeds_target() {
        let (root, prepared) = fixture();
        let mut device = device();
        let mut request = request(&prepared);
        request.config.batch_size = 4;
        let state = TrainingState::fresh(&request, &prepared, &device).unwrap();
        // Transport simulation only: no CUDA allocation or neural step occurs.
        device.descriptor.backend = BackendKind::Cuda;
        device.descriptor.id = "cuda:0".into();
        SIMULATED.with(|m| *m.borrow_mut() = Some("oom".into()));
        let selection = crate::training::batch_probe::probe_batch_size(
            &state,
            &prepared,
            &device,
            &ProbeContext {
                runtime: &TrainingRuntime {
                    probe_runner: std::env::current_exe().unwrap(),
                },
                vgg_assets: &assets(),
                cancel: &Cancellation::new(),
            },
        )
        .unwrap();
        SIMULATED.with(|m| *m.borrow_mut() = None);
        assert_eq!(selection.attempts, vec![4, 2, 1]);
        assert_eq!(selection.candidate, 1);
        assert!(!fs::read_dir(&root).unwrap().any(|e| {
            e.unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".probe-")
        }));
    }
    #[test]
    fn cancellation_during_simulated_probe_commits_current_state() {
        struct Observe;
        impl TaskObserver for Observe {
            fn emit(&self, _: EventPayload) -> Result<()> {
                Ok(())
            }
        }
        let (root, prepared) = fixture();
        let device = device();
        let cancel = Cancellation::new();
        let token = cancel.clone();
        let marker = root.join("child-started");
        let signal = thread::spawn(move || {
            for _ in 0..2400 {
                if marker.exists() {
                    token.cancel();
                    return;
                }
                thread::sleep(Duration::from_millis(25));
            }
            panic!("child never started");
        });
        SIMULATED.with(|m| *m.borrow_mut() = Some("wait".into()));
        let result = crate::training::train(
            &request(&prepared),
            &prepared,
            &assets(),
            &device,
            &TrainingRuntime {
                probe_runner: std::env::current_exe().unwrap(),
            },
            &Observe,
            &cancel,
        );
        signal.join().unwrap();
        SIMULATED.with(|m| *m.borrow_mut() = None);
        let result = result.expect("probe cancellation must save current coherent checkpoint");
        assert_eq!(result.global_step, 0);
        assert_eq!(
            result.stopping_reason,
            crate::training::StoppingReason::Cancelled
        );
        assert!(!fs::read_dir(&root).unwrap().any(|e| {
            e.unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".probe-")
        }));
    }
}
/// Entrypoint for the explicitly supplied backend-compatible helper executable.
pub fn run_request(path: &Path) -> Result<()> {
    if !path.is_absolute() {
        return Err(invalid("Absolute request path required"));
    }
    let request: Request =
        serde_json::from_slice(&fs::read(path).map_err(invalid)?).map_err(invalid)?;
    if request.binding.version != VERSION || request.binding.candidate == 0 {
        return Err(invalid("Invalid probe protocol"));
    }
    let dir = path
        .parent()
        .ok_or_else(|| invalid("Missing request parent"))?;
    let outcome = (|| {
        let device = crate::open_device(&request.binding.device)?;
        let snapshot = MaterializedSnapshot::read_private(&dir.join("snapshot"))?;
        let mut state = snapshot.restore(&request.project, &device)?;
        if state.metadata.global_step != request.binding.step
            || state.metadata.batch_mode != BatchMode::Auto
            || request.binding.candidate > state.metadata.requested_batch
        {
            return Err(invalid("Candidate differs from snapshot"));
        }
        let prepared = reopen_generation(
            &request.project,
            &read_project(&request.project)?,
            &state.metadata.preprocessing.cache_directory,
            &state.metadata.prepared_manifest_sha256,
        )?;
        let expected = request
            .binding
            .candidate
            .min(prepared.usable_frames.saturating_sub(1) as u32);
        if request.binding.effective != expected {
            return Err(invalid("Candidate cardinality mismatch"));
        }
        state.metadata.actual_batch = request.binding.candidate;
        rewind_disposable_sampler(&mut state.metadata.sampler); // Disposable only; RNG words and formal counters retained.
        let vgg = crate::models::vgg::VggFeatures::load(
            &request.assets,
            &device.device.clone().autodiff(),
        )?;
        if vgg.identity() != &request.binding.vgg_identity {
            return Err(invalid("Probe VGG package changed after selection"));
        }
        let step = super::trainer::complete_step(
            &mut state,
            &prepared,
            &vgg,
            &device,
            &Cancellation::new(),
        )?;
        if step.pairs.len() != expected as usize
            || state.metadata.global_step != request.binding.step + 1
        {
            return Err(invalid("Incomplete candidate update"));
        }
        let scalars = step.losses.to_scalars();
        scalars.validate()?;
        Ok(CompletedStep {
            losses: scalars,
            global_step: state.metadata.global_step,
            pairs: step.pairs.len() as u32,
            device: DeviceRequest {
                backend: device.descriptor.backend,
                device_id: device.descriptor.id,
            },
        })
    })();
    atomic_json(
        &dir.join("result.json"),
        &Response {
            binding: request.binding,
            outcome,
        },
    )
}
