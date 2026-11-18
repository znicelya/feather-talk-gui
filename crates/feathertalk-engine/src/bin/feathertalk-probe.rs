fn main() {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 || args[0] != "--request" {
        eprintln!("Usage: feathertalk-probe --request <absolute request.json>");
        std::process::exit(2);
    }
    if let Err(e) =
        feathertalk_engine::training::probe_transport::run_request(std::path::Path::new(&args[1]))
    {
        eprintln!("{e}");
        std::process::exit(1);
    }
}
