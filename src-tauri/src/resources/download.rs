use super::{catalog::FileSpec, fail};
use feathertalk_core::*;
use std::{
    fs,
    io::{Read, Write},
    path::Path,
    time::{Duration, Instant},
};
pub(crate) fn cancelled(cancel: &Cancellation) -> Result<()> {
    if cancel.is_cancelled() {
        return Err(AppError::new(
            ErrorCode::Cancelled,
            "resources",
            "Resource preparation cancelled",
        ));
    }
    Ok(())
}
pub(crate) fn fetch(
    path: &Path,
    spec: &FileSpec,
    observer: &dyn TaskObserver,
    cancel: &Cancellation,
) -> Result<()> {
    let client = reqwest::blocking::Client::builder()
        .connect_timeout(Duration::from_secs(5))
        .timeout(Duration::from_secs(30))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(fail)?;
    let mut last = fail("download source unconfigured; import the verified resource locally");
    for url in spec.urls.iter().take(3) {
        for _attempt in 0..2 {
            cancelled(cancel)?;
            let attempt = (|| {
                let mut response = client
                    .get(url)
                    .send()
                    .map_err(fail)?
                    .error_for_status()
                    .map_err(fail)?;
                if response.content_length().is_some_and(|n| n != spec.size) {
                    return Err(fail("HTTP size mismatch"));
                }
                let mut f = fs::File::create(path).map_err(fail)?;
                let mut done = 0u64;
                let mut buffer = [0; 65536];
                let mut last_emit = Instant::now();
                loop {
                    cancelled(cancel)?;
                    let n = response.read(&mut buffer).map_err(fail)?;
                    if n == 0 {
                        break;
                    }
                    done += n as u64;
                    if done > spec.size {
                        return Err(fail("HTTP body exceeds expected size"));
                    }
                    f.write_all(&buffer[..n]).map_err(fail)?;
                    if last_emit.elapsed() >= Duration::from_millis(100) {
                        observer.emit(EventPayload::Progress {
                            stage: "resources/download".into(),
                            completed: done,
                            total: Some(spec.size),
                        })?;
                        last_emit = Instant::now();
                    }
                }
                if done != spec.size {
                    return Err(fail("truncated HTTP body"));
                }
                f.sync_all().map_err(fail)?;
                super::inventory::verify_file(
                    path.parent().unwrap(),
                    path.file_name().unwrap().as_ref(),
                    spec,
                )?;
                Ok(())
            })();
            match attempt {
                Ok(()) => return Ok(()),
                Err(e) => {
                    if cancel.is_cancelled() {
                        return cancelled(cancel);
                    }
                    last = e;
                }
            }
        }
    }
    Err(last)
}
