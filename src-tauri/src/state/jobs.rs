//! Drain owned workers off the desktop event loop and keep Ready off the task stream.
use super::{AppState, Conclusion, invalid, state_error};
use crate::{
    projects::regular_file,
    worker_host::{HostPoll, WorkerHost},
};
use feathertalk_core::*;
use std::time::{Duration, Instant};

const POLL: Duration = Duration::from_millis(100);
const STARTUP_LIMIT: Duration = Duration::from_secs(180);

pub(super) fn run(
    app: &AppState,
    task: TaskId,
    mut job: JobSpec,
    cancel: Cancellation,
) -> Result<Conclusion> {
    check_cancel(&cancel)?;
    let (project, device) = match &job {
        JobSpec::Train(request) => (request.project_id, request.device.clone()),
        JobSpec::Generate(request) => (request.project_id, request.device.clone()),
    };
    let binding = app.0.projects.bind(project)?;
    if let JobSpec::Generate(request) = &mut job {
        let manifest = app.get_project(project)?;
        request.audio_path = regular_file(&request.audio_path)?;
        let parent = request
            .output_path
            .parent()
            .ok_or_else(|| invalid("Missing output directory"))?
            .canonicalize()
            .map_err(|e| invalid(&format!("Cannot open the selected output directory: {e}")))?;
        let filename = request
            .output_path
            .file_name()
            .ok_or_else(|| invalid("Missing MP4 filename"))?;
        request.output_path = parent.join(filename);
        let source = regular_file(&binding.directory.join(manifest.source_video.path))?;
        let existing = request
            .output_path
            .canonicalize()
            .unwrap_or_else(|_| request.output_path.clone());
        if existing == source || existing == request.audio_path {
            return Err(invalid("The output path would overwrite the input media"));
        }
    }
    let mut failures = Vec::new();
    let mut installed = None;
    for selection in app
        .resource_selections()
        .into_iter()
        .filter(|sel| match &device {
            DevicePolicy::Auto => true,
            DevicePolicy::Explicit(request) => sel.target.backend == backend_name(request.backend),
        })
    {
        check_cancel(&cancel)?;
        let attempt = || {
            let mut host = app
                .0
                .resources
                .bind_worker(&selection, binding.clone(), &job)?
                .spawn()?;
            let (capabilities, devices) = wait_ready(&mut host, &cancel)?;
            if !(match &job {
                JobSpec::Train(_) => capabilities.training,
                JobSpec::Generate(_) => capabilities.generation,
            }) {
                return Err(protocol_error(
                    "Installed worker did not announce the requested job capability",
                ));
            }
            if let DevicePolicy::Explicit(request) = &device
                && !devices.iter().any(|device| {
                    device.id == request.device_id && device.backend == request.backend
                })
            {
                return Err(AppError::new(
                    ErrorCode::UnsupportedBackend,
                    "device",
                    "The verified worker did not report the requested device",
                ));
            }
            Ok((host, devices))
        };
        match attempt() {
            Ok(pair) => {
                installed = Some(pair);
                break;
            }
            Err(error) if error.code == ErrorCode::Cancelled => return Err(error),
            Err(error) => failures.push((selection.target.backend, error)),
        }
    }
    let Some((mut host, devices)) = installed else {
        return Err(unavailable(failures));
    };
    remember(app, devices)?;
    check_cancel(&cancel)?;
    host.start(task, job)?;

    let mut sent_cancel = false;
    let mut started = false;
    let start_deadline = Instant::now() + STARTUP_LIMIT;
    loop {
        if cancel.is_cancelled() && !sent_cancel {
            // Even if stdin is already closed, drain the owned terminal/EOF evidence.
            let _ = host.cancel(task);
            sent_cancel = true;
        }
        match host.next_event(POLL) {
            Ok(HostPoll::Timeout) if !started && Instant::now() >= start_deadline => {
                return Err(AppError::new(
                    ErrorCode::DeviceInit,
                    "worker-start",
                    "Worker did not acknowledge the task",
                ));
            }
            Ok(HostPoll::Timeout) => {}
            Ok(HostPoll::Ended) => {
                return Err(protocol_error(
                    "Worker ended without an owned terminal event",
                ));
            }
            Ok(HostPoll::Event(event)) => {
                if event.task_id != Some(task) {
                    return Err(protocol_error("Unexpected event during an owned task"));
                }
                if is_terminal(&event.payload) {
                    let _ = host.shutdown();
                    // The host and contained children are reaped before the desktop lease releases.
                    drop(host);
                    return Ok(Conclusion::Worker(event));
                }
                started |= matches!(event.payload, EventPayload::Started { .. });
                app.record(event)?;
            }
            Err(error) => match host.rejected_request() {
                Some((RequestKind::Start, Some(id))) if id == task && !started => {
                    return Err(error);
                }
                Some(_) => {} // A rejected control request is not the owner's terminal.
                None => return Err(error),
            },
        }
    }
}

pub(super) fn list_devices(app: &AppState) -> Result<Vec<DeviceDescriptor>> {
    let cancel = Cancellation::new();
    let _discovery = {
        let mut tasks = app.tasks()?;
        if tasks.closing || tasks.is_busy() {
            return Ok(app.0.devices.lock().map_err(state_error)?.clone());
        }
        tasks.discovery = Some(cancel.clone());
        DiscoveryLease(app)
    };
    let mut failures = Vec::new();
    let mut reports = Vec::new();
    for selection in app.resource_selections() {
        let attempt = || {
            check_cancel(&cancel)?;
            let mut host = app.0.resources.bind_discovery(&selection)?.spawn()?;
            let (capabilities, devices) = wait_ready(&mut host, &cancel)?;
            if capabilities.training || capabilities.generation {
                return Err(protocol_error(
                    "Worker did not acknowledge discovery-only mode",
                ));
            }
            let _ = host.shutdown();
            Ok::<_, AppError>(devices)
        };
        match attempt() {
            Ok(devices) => reports.extend(devices),
            Err(error) if error.code == ErrorCode::Cancelled => {
                return Ok(app.0.devices.lock().map_err(state_error)?.clone());
            }
            Err(error) if error.code == ErrorCode::ResourceMissing => {}
            Err(error) => failures.push((selection.target.backend, error)),
        }
    }
    *app.0.devices.lock().map_err(state_error)? = reports.clone();
    if reports.is_empty() && !failures.is_empty() {
        return Err(unavailable(failures));
    }
    Ok(reports)
}

struct DiscoveryLease<'a>(&'a AppState);

impl Drop for DiscoveryLease<'_> {
    fn drop(&mut self) {
        if let Ok(mut tasks) = self.0.tasks() {
            tasks.discovery = None;
            self.0.0.idle.notify_all();
        }
    }
}

fn remember(app: &AppState, devices: Vec<DeviceDescriptor>) -> Result<()> {
    let mut known = app.0.devices.lock().map_err(state_error)?;
    known.retain(|d| !devices.iter().any(|new| new.backend == d.backend));
    known.extend(devices);
    Ok(())
}

fn wait_ready(
    host: &mut WorkerHost,
    cancel: &Cancellation,
) -> Result<(WorkerCapabilities, Vec<DeviceDescriptor>)> {
    let until = Instant::now() + STARTUP_LIMIT;
    loop {
        check_cancel(cancel)?;
        match host.next_event(POLL)? {
            HostPoll::Event(WorkerEvent {
                payload:
                    EventPayload::Ready {
                        capabilities,
                        devices,
                    },
                ..
            }) => return Ok((capabilities, devices)),
            HostPoll::Event(_) | HostPoll::Ended => {
                return Err(protocol_error("Worker ended or sent an event before Ready"));
            }
            HostPoll::Timeout if Instant::now() >= until => {
                return Err(AppError::new(
                    ErrorCode::DeviceInit,
                    "worker-start",
                    "Worker startup timed out",
                ));
            }
            HostPoll::Timeout => {}
        }
    }
}

fn unavailable(failures: Vec<(String, AppError)>) -> AppError {
    let mut error = failures
        .iter()
        .find(|(_, e)| e.code != ErrorCode::ResourceMissing)
        .or_else(|| failures.first())
        .map(|(_, error)| error.clone())
        .unwrap_or_else(|| {
            AppError::new(
                ErrorCode::ResourceMissing,
                "worker-start",
                "No configured worker for the requested backend",
            )
        });
    if failures.len() > 1 {
        for (backend, cause) in failures {
            error.evidence.push(format!("{backend}: {cause}"));
        }
    }
    error.actions.push("Check the installed resources and driver for the chosen backend. Select an available CPU worker if GPU startup fails.".into());
    error
}

fn check_cancel(cancel: &Cancellation) -> Result<()> {
    if cancel.is_cancelled() {
        Err(AppError::new(
            ErrorCode::Cancelled,
            "worker-start",
            "Cancelled before the worker started",
        ))
    } else {
        Ok(())
    }
}

fn backend_name(kind: BackendKind) -> &'static str {
    match kind {
        BackendKind::Cpu | BackendKind::Flex => "cpu",
        BackendKind::Vulkan => "vulkan",
        BackendKind::Cuda => "cuda",
        BackendKind::Metal => "metal",
    }
}

fn is_terminal(payload: &EventPayload) -> bool {
    matches!(
        payload,
        EventPayload::Completed { .. }
            | EventPayload::Cancelled { .. }
            | EventPayload::Failed { .. }
    )
}
