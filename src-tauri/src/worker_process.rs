//! Own exactly one worker and its descendants. Never identify processes by name.
use feathertalk_core::*;
use std::process::{Child, Command};
#[cfg(windows)]
pub struct Containment(windows_sys::Win32::Foundation::HANDLE);
// A uniquely owned kernel Job handle has no thread affinity. Moving it does not
// duplicate ownership; only its owner closes it, including on another thread.
#[cfg(windows)]
unsafe impl Send for Containment {}
#[cfg(windows)]
impl Containment {
    pub fn configure(command: &mut Command) {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000);
    }
    pub fn assign(child: &mut Child) -> Result<Self> {
        use std::{
            mem::{size_of, zeroed},
            os::windows::io::AsRawHandle,
        };
        use windows_sys::Win32::{Foundation::CloseHandle, System::JobObjects::*};
        unsafe {
            let handle = CreateJobObjectW(std::ptr::null(), std::ptr::null());
            if handle.is_null() {
                let error = std::io::Error::last_os_error();
                let _ = child.kill();
                let _ = child.wait();
                return Err(protocol_error(error));
            }
            let mut info: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = zeroed();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            if SetInformationJobObject(
                handle,
                JobObjectExtendedLimitInformation,
                &info as *const _ as *const _,
                size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
            ) == 0
                || AssignProcessToJobObject(handle, child.as_raw_handle()) == 0
            {
                let e = std::io::Error::last_os_error();
                CloseHandle(handle);
                let _ = child.kill();
                let _ = child.wait();
                return Err(protocol_error(e));
            }
            Ok(Self(handle))
        }
    }
}
#[cfg(windows)]
impl Drop for Containment {
    fn drop(&mut self) {
        unsafe {
            windows_sys::Win32::Foundation::CloseHandle(self.0);
        }
    }
}
#[cfg(unix)]
pub struct Containment(i32);
#[cfg(unix)]
impl Containment {
    pub fn configure(command: &mut Command) {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    pub fn assign(child: &mut Child) -> Result<Self> {
        Ok(Self(child.id() as i32))
    }
}
#[cfg(unix)]
impl Drop for Containment {
    fn drop(&mut self) {
        unsafe {
            libc::kill(-self.0, libc::SIGKILL);
        }
    }
}

#[cfg(all(test, windows))]
mod tests {
    use super::*;
    use std::{
        fs,
        time::{Duration, Instant},
    };
    #[test]
    fn owned_grandchild_dies_on_job_close() {
        let dir = tempfile::tempdir().unwrap();
        let pid_file = dir.path().join("pid");
        let mut command = Command::new(std::env::current_exe().unwrap());
        command
            .args([
                "--exact",
                "worker_process::tests::child_helper",
                "--nocapture",
            ])
            .env("FT_OWNED_CHILD", &pid_file);
        Containment::configure(&mut command);
        let mut child = command.spawn().unwrap();
        let job = Containment::assign(&mut child).unwrap();
        // Start gate is released only after assignment, like worker bootstrap/Hello before Start.
        fs::write(pid_file.with_extension("start"), b"start").unwrap();
        let started = Instant::now();
        while !pid_file.exists() {
            assert!(started.elapsed() < Duration::from_secs(15));
            std::thread::sleep(Duration::from_millis(20));
        }
        let pid: u32 = fs::read_to_string(&pid_file).unwrap().parse().unwrap();
        use windows_sys::Win32::{Foundation::CloseHandle, System::Threading::*};
        unsafe {
            let handle = OpenProcess(
                PROCESS_SYNCHRONIZE | PROCESS_QUERY_LIMITED_INFORMATION,
                0,
                pid,
            );
            assert!(!handle.is_null());
            assert_eq!(WaitForSingleObject(handle, 0), 258);
            drop(job);
            child.wait().unwrap();
            assert_eq!(WaitForSingleObject(handle, 5000), 0);
            CloseHandle(handle);
        }
    }
    #[test]
    fn child_helper() {
        if let Some(path) = std::env::var_os("FT_OWNED_CHILD") {
            let path = std::path::PathBuf::from(path);
            while !path.with_extension("start").exists() {
                std::thread::sleep(Duration::from_millis(10));
            }
            let mut command = Command::new(std::env::current_exe().unwrap());
            command
                .args(["--exact", "worker_process::tests::grandchild_helper"])
                .env_remove("FT_OWNED_CHILD")
                .env("FT_OWNED_GRANDCHILD", "1");
            Containment::configure(&mut command);
            let mut grandchild = command.spawn().unwrap();
            fs::write(path, grandchild.id().to_string()).unwrap();
            let _ = grandchild.wait();
        }
    }
    #[test]
    fn grandchild_helper() {
        if std::env::var_os("FT_OWNED_GRANDCHILD").is_some() {
            std::thread::sleep(Duration::from_secs(60));
        }
    }
}
