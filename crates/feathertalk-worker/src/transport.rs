use feathertalk_core::*;
use std::io::Write;
pub fn write_event(writer: &mut impl Write, event: &WorkerEvent) -> Result<()> {
    let bytes = serde_json::to_vec(event).map_err(protocol_error)?;
    if bytes.len() > MAX_MESSAGE_BYTES {
        return Err(protocol_error("Outbound event exceeds limit"));
    }
    writer
        .write_all(&bytes)
        .and_then(|_| writer.write_all(b"\n"))
        .and_then(|_| writer.flush())
        .map_err(protocol_error)
}

/// Own process stdout. A fatal output error cannot wait for another stdin record.
pub fn run_output(receiver: crate::outbox::EventReceiver) {
    let mut output = std::io::stdout().lock();
    loop {
        match receiver.recv_timeout(std::time::Duration::from_secs(1)) {
            Ok(event) => {
                if let Err(error) = write_event(&mut output, &event) {
                    eprintln!("worker output failed: {error}");
                    std::process::exit(1);
                }
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {}
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => return,
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fatal_writer_exits_without_more_stdin() {
        use std::{
            io::BufReader,
            process::{Command, Stdio},
            time::{Duration, Instant},
        };
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "transport::tests::fatal_writer_helper",
                "--nocapture",
            ])
            .env("FT_FATAL_WRITER", "1")
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        let mut output = BufReader::new(child.stdout.take().unwrap());
        // Test harness also prints its own preamble; find the real Ready record.
        loop {
            let frame = read_frame(&mut output).unwrap().unwrap();
            if serde_json::from_slice::<WorkerEvent>(&frame).is_ok() {
                break;
            }
        }
        let start = Instant::now();
        loop {
            if let Some(status) = child.try_wait().unwrap() {
                assert!(!status.success());
                break;
            }
            if start.elapsed() > Duration::from_secs(5) {
                let _ = child.kill();
                panic!("fatal writer left process alive with idle stdin");
            }
            std::thread::sleep(Duration::from_millis(10));
        }
        while let Some(frame) = read_frame(&mut output).unwrap() {
            let e: WorkerEvent = serde_json::from_slice(&frame).unwrap();
            assert!(matches!(e.payload, EventPayload::Started { .. }));
        }
    }
    #[test]
    fn fatal_writer_helper() {
        if std::env::var_os("FT_FATAL_WRITER").is_none() {
            return;
        }
        let (owner, rx) = crate::outbox::event_channel();
        owner
            .control(EventPayload::Ready {
                capabilities: WorkerCapabilities {
                    engine_version: "fixture".into(),
                    components: vec![],
                    training: false,
                    generation: false,
                },
                devices: vec![],
            })
            .unwrap();
        let task = TaskId::new();
        owner.start(task, JobKind::Train).unwrap();
        owner
            .emit(
                task,
                EventPayload::Failed {
                    error: AppError::new(
                        ErrorCode::Internal,
                        "fixture",
                        "x".repeat(MAX_MESSAGE_BYTES),
                    ),
                },
            )
            .unwrap();
        std::thread::spawn(move || run_output(rx));
        // Simulates the transport's stdin reader remaining idle forever.
        std::thread::sleep(std::time::Duration::from_secs(60));
    }
}
