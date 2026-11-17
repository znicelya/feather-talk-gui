# Reference oracle and current asset conversion

These scripts read the original Python project and supplied models without modifying them. They produce CPU FP32 SafeTensors and JSON under `.artifacts`. `manifest.json` uses the Rust core `AssetManifest` fields; each component has its own identity and architecture/configuration sidecar. SCRFD is rebuilt from original ONNX initializers with original names, with its executable graph retained as architecture evidence. Historical Burn 0.21 names are deliberately not reused. FeatherHuBERT is rebuilt using the actual current checkpoint configuration, with strict state loading. The historical 188 checkpoint is rejected by digest.

Use Python 3.12 and an isolated environment. The versions in `requirements.lock` are the tested CPU environment. PyTorch and torchvision CPU wheels come from the official CPU index; other packages use PyPI. No global PATH or registry changes are needed.

```powershell
$py = '.superpowers/toolchain/reference-venv/Scripts/python.exe'
$uv = '.superpowers/toolchain/uv/uv.exe'
$env:UV_HTTP_TIMEOUT = '1800'
& $uv pip install --python $py -r tools/reference/requirements.lock --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
$env:FT_SOURCE_DIR = 'D:/workspace/github/feather-talk-sync/FeatherTalk'
$env:FT_SUPPLIED_MODELS_DIR = 'D:/workspace/github/feather-talk-sync/models'
$env:FT_ASSETS_DIR = (Join-Path $PWD '.artifacts/assets')
$env:FT_REFERENCE_DIR = (Join-Path $PWD '.artifacts/reference')
```

The official full VGG19 checkpoint must be downloaded once to `.artifacts/source-downloads/vgg19-dcbb9e9d.pth`, or supplied using `FT_VGG_SOURCE`. Obtain it from https://download.pytorch.org/models/vgg19-dcbb9e9d.pth. The audit checks its full SHA-256, compares every supplied subset tensor against the original, and exports only `features[0..14]`. It does not download implicitly.

```powershell
& $py tools/reference/audit.py --source $env:FT_SOURCE_DIR --models $env:FT_SUPPLIED_MODELS_DIR --output .artifacts/audit.json
& $py tools/reference/convert.py --source $env:FT_SOURCE_DIR --models $env:FT_SUPPLIED_MODELS_DIR --output $env:FT_ASSETS_DIR
& $py tools/reference/generate.py --source $env:FT_SOURCE_DIR --assets $env:FT_ASSETS_DIR --output $env:FT_REFERENCE_DIR
& $py tools/reference/verify.py --reference $env:FT_REFERENCE_DIR
& $py -m pytest tools/reference/tests/test_assets.py -q
```

The tests regenerate references twice, require identical manifests and tensor digests, and reject corrupted files. `tests/reference/manifest.json` is the small checked-in index; binaries remain ignored. Run the verifier before using an oracle in Rust and check the index against the checked-in baseline. `FT_REFERENCE_DIR` and `FT_ASSETS_DIR` are the downstream Rust entry points.

Every case records inputs, outputs, shapes, configuration and tolerances. Operator backward objectives use nonuniform explicit upstream gradients. BatchNorm has two train batches, train input/affine gradients, unbiased running variance updates and eval output. GroupNorm uses biased variance. Average pooling includes zero padding in the denominator. Attention includes trainable positions, relative biases and gate gradients. The fixed character state contains parameters and buffers so Rust need not reproduce PyTorch initialization. The character training fixture uses two adjacent frames, the actual original loss functions, fixed VGG features, all six loss terms and representative gradients across visual/audio/attention/output modules. An eval fixture captures the main visual layers.

Geometry fixtures retain BGR, area resize to 152, central 144, mouth fill and ROI masks. Audio fixtures include clip padding, first/last frame windows, short waveforms, exact convolution thresholds and a chunk boundary. Synthetic fixtures establish numerical parity only. `fixtures.json` explicitly tracks the outstanding real footage acceptance requirement.

FP32 tolerances are prospective gates frozen before a Burn comparison: operators 2e-6 absolute/2e-5 relative, networks 2e-5/2e-4, gradients 2e-6/5e-4, geometry and serialization exact. These reflect reduction accumulation and near-zero gradient sensitivity, not measured Burn results. Same-environment regeneration must be byte-identical. Investigate any port failure; do not automatically widen gates.
