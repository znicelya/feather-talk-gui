use feathertalk_core::*;
use feathertalk_worker::runtime::WorkerRuntime;
use std::io::{self, BufReader};
fn run() -> Result<()> {
    let mut input = BufReader::new(io::stdin().lock());
    let bytes = read_frame(&mut input)?.ok_or_else(|| protocol_error("Missing bootstrap"))?;
    let bootstrap: WorkerBootstrap = serde_json::from_slice(&bytes).map_err(protocol_error)?;
    if bootstrap.protocol_version != PROTOCOL_VERSION {
        return Err(protocol_error("Bootstrap version mismatch"));
    }
    let (tx, rx) = feathertalk_worker::runtime::event_channel();
    let mut runtime = WorkerRuntime::new(bootstrap.context, tx.clone())?;
    let writer = std::thread::spawn(move || feathertalk_worker::transport::run_output(rx));
    let result = (|| {
        while let Some(bytes) = read_frame(&mut input)? {
            let request = serde_json::from_slice(&bytes).map_err(protocol_error)?;
            let _ = runtime.handle(request);
            if tx.is_closed() {
                eprintln!("Fatal transport overload");
                std::process::exit(1);
            }
            if runtime.stopping() {
                break;
            }
        }
        Ok(())
    })();
    if result.is_err() {
        tx.close();
    }
    runtime.finish();
    tx.close();
    drop(runtime);
    writer
        .join()
        .map_err(|_| protocol_error("Writer panicked"))?;
    result
}
fn main() {
    if let Err(e) = run() {
        eprintln!("{e:?}");
        std::process::exit(1);
    }
}
