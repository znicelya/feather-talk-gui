use feathertalk_core::{EventPayload, Result as EngineResult, TaskObserver};
use feathertalk_engine::benchmark::{
    BenchmarkReport, BenchmarkSpec, run_benchmark, select_backend,
};
use std::{fs, io::Write, path::PathBuf};

struct Observer;
impl TaskObserver for Observer {
    fn emit(&self, event: EventPayload) -> EngineResult<()> {
        if let EventPayload::Progress {
            stage,
            completed,
            total,
        } = event
            && stage.starts_with("benchmark-")
        {
            eprintln!("{stage}: {}/{}", completed + 1, total.unwrap_or(1));
        }
        Ok(())
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let arguments = std::env::args_os().skip(1).collect::<Vec<_>>();
    if arguments.first().is_some_and(|arg| arg == "--compare") {
        let reports = arguments[1..]
            .iter()
            .map(|path| {
                let bytes = fs::read(path)?;
                Ok(serde_json::from_slice::<BenchmarkReport>(&bytes)?)
            })
            .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
        println!(
            "{}",
            serde_json::to_string_pretty(&select_backend(&Default::default(), &reports)?)?
        );
        return Ok(());
    }
    let [spec_flag, spec_file, output_flag, output_file] = arguments.as_slice() else {
        return Err(
            "Usage: benchmark --spec FILE --output NEW_REPORT.json | --compare REPORT.json ..."
                .into(),
        );
    };
    if spec_flag != "--spec" || output_flag != "--output" {
        return Err("Expected --spec FILE --output NEW_REPORT.json".into());
    }
    let output = PathBuf::from(output_file);
    if output.exists() {
        return Err("Benchmark report already exists; use a new filename".into());
    }
    let spec: BenchmarkSpec = serde_json::from_slice(&fs::read(spec_file)?)?;
    let report = run_benchmark(&spec, &Observer)?;
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(output)?;
    file.write_all(&serde_json::to_vec_pretty(&report)?)?;
    file.sync_all()?;
    if !report.validation_passed {
        return Err("Benchmark failed the numerical validation; see the saved report".into());
    }
    Ok(())
}
