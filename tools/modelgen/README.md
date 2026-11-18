# SCRFD generation

Run from the workspace root with the pinned developer reference Python environment:

```powershell
$env:FT_MODELGEN_PYTHON = (Resolve-Path .superpowers/toolchain/reference-venv/Scripts/python.exe).Path
cargo run -p feathertalk-modelgen -- --input .artifacts/assets/scrfd_2_5g/architecture.onnx --source-output crates/feathertalk-engine/src/models/generated/scrfd.rs --assets-output .artifacts/assets/scrfd_2_5g
```

The exact original graph digest is mandatory. The developer helper proves the three fixed-640 AveragePool windows have identical floor/ceil geometry, changes only those redundant attributes, and requires all nine original/derived ORT outputs to be bit-identical for the original Torch linspace fixture and a second deterministic input. This works around onnx-ir 0.22.0's incorrect opset-19 ceil-mode check. Original architecture.onnx stays untouched; generation.json records both graph hashes and the proof.

The Rust CLI runs burn-onnx 0.22.0, strips generated path-dependent Default/panicking convenience loaders, formats the source, and re-encodes Burnpack using its public Reader/Writer API with stable full-tensor-path SHA256-derived ParamIds (collision checked). Tensor name/dtype/shape/value digest is logged. The manifest records the actual final binary hash. Component and bundle inventories and reference asset digest are refreshed together. Copy the refreshed reference manifest to tests/reference/manifest.json when intentionally publishing a new asset catalog. Binary assets and staging stay under ignored .artifacts.

The app uses VerifiedAssets and fallible ModuleSnapshot loading, requiring complete state coverage. It needs no Python/ORT. PFLD loads the original six-branch eval graph: 351 counters are accounted, and only the four explicitly named and shape-checked localization parameters are inactive. No branch fusion or CPU neural fallback is performed.
