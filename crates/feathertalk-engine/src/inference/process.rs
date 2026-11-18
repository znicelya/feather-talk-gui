//! Owned staging and reaped media children, shared by offline file adapters.
use crate::media::ffmpeg::{check, error};
use feathertalk_core::*;
use std::{
    fs,
    io::Read,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    thread,
    time::Duration,
};
pub(crate) struct Staging(pub PathBuf);
impl Staging {
    pub fn new(parent: &Path) -> Result<Self> {
        let parent = fs::canonicalize(parent).map_err(error)?;
        let path = parent.join(format!(".export-{}", TrainingRunId::new()));
        fs::create_dir(&path).map_err(error)?;
        Ok(Self(path))
    }
}
impl Drop for Staging {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn drain(mut input: impl Read, limit: usize) -> std::io::Result<Vec<u8>> {
    let mut tail = Vec::new();
    let mut buf = [0u8; 8192];
    loop {
        let n = input.read(&mut buf)?;
        if n == 0 {
            break;
        }
        tail.extend_from_slice(&buf[..n]);
        if tail.len() > limit {
            tail.drain(..tail.len() - limit);
        }
    }
    Ok(tail)
}
/// Pipe drain threads retain bounded real diagnostics; cancellation kills and waits before return.
pub(crate) fn run(command: &mut Command, cancel: &Cancellation) -> Result<Vec<u8>> {
    check(cancel)?;
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    let mut child = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(error)?;
    let stdout = child.stdout.take().unwrap();
    let stderr = child.stderr.take().unwrap();
    let out = thread::spawn(move || drain(stdout, 4 * 1024 * 1024));
    let err = thread::spawn(move || drain(stderr, 65536));
    let status = loop {
        if let Err(e) = check(cancel) {
            let _ = child.kill();
            let _ = child.wait();
            break Err(e);
        }
        match child.try_wait() {
            Ok(Some(status)) => break Ok(status),
            Ok(None) => thread::sleep(Duration::from_millis(10)),
            Err(e) => {
                let _ = child.kill();
                let _ = child.wait();
                break Err(error(e));
            }
        }
    };
    let stdout = out
        .join()
        .map_err(|_| error("stdout drain panicked"))?
        .map_err(error)?;
    let stderr = err
        .join()
        .map_err(|_| error("stderr drain panicked"))?
        .map_err(error)?;
    let status = status.map_err(|mut e| {
        if !stderr.is_empty() {
            e.evidence
                .push(String::from_utf8_lossy(&stderr).into_owned());
        }
        e
    })?;
    if !status.success() {
        return Err(error(format!(
            "Media tool exited {status}: {}",
            String::from_utf8_lossy(&stderr)
        )));
    }
    check(cancel)?;
    Ok(stdout)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn actual_media_child_cancel_reaps_and_retains_bounded_stderr() {
        let bin = Path::new(env!("CARGO_MANIFEST_DIR")).join(
            "../../.superpowers/toolchain/ffmpeg/ffmpeg-9.0.2-essentials_build/bin/ffmpeg.exe",
        );
        let cancel = Cancellation::new();
        let signal = cancel.clone();
        let t = thread::spawn(move || {
            thread::sleep(Duration::from_millis(500));
            signal.cancel();
        });
        let mut command = Command::new(&bin);
        command.args([
            "-nostdin",
            "-v",
            "error",
            "-re",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=32x32:rate=25",
            "-f",
            "null",
            "-",
        ]);
        let e = run(&mut command, &cancel).unwrap_err();
        t.join().unwrap();
        assert_eq!(e.code, ErrorCode::Cancelled);
        let input = vec![b'x'; 200000];
        let tail = drain(&input[..], 65536).unwrap();
        assert_eq!(tail.len(), 65536);
        let mut broken = Command::new(bin);
        broken.arg("-not-a-real-option-task10");
        let e = run(&mut broken, &Cancellation::new()).unwrap_err();
        assert!(e.message.contains("not-a-real-option-task10"));
    }
}
