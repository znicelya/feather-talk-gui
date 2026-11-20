Resource catalog schema version 1 wraps strict core AssetManifest values. It does not modify source manifests. Each package additionally pins the exact manifest.json bytes and any separately inventoried notice attachments. No production URLs or distribution approval are provided here.

ResourceManager::new(root, catalog) validates metadata. inventory(target) checks that target's explicit default application selection. inventory_selection(selection) checks exact pinned subsets. prepare(selection, observer, cancellation) is the only HTTP entry point. import_local(path/to/manifest.json) accepts only a package matching configured metadata and copies verified bytes. All returned paths are installed cache roots, never developer toolchain paths.

public-assets installs the original bundle (20 listed files + top-level manifest) with five unchanged component directories. Use its root for preprocessing/generation; root/vgg19 for training. resolve_identity(identity, target) also resolves nested model identities from retained, verified package metadata when a later catalog no longer contains their pin. The old version is never redirected to a newer version.

A component pin (id, version) is immutable. Installation stages and validates an entire component before directory rename. A selection may contain multiple components; successfully installed earlier components remain if a later component fails. The selection is ready only when every specified pin is verified. Corrupt copies can be replaced with the same pinned bytes after the replacement verifies; older valid versions remain. Concurrent mutators fail Busy through a cache file lock. Inventory does not acquire that lock and can momentarily report missing during repair.

Target names are windows/linux/macos and x86_64/aarch64. A component target backend of any permits backend-independent media on the same OS/architecture. A null target denotes platform-independent models. Selection backends must be cpu/cuda/vulkan/metal with supported platform policy; Intel Mac supports cpu only. Binary checks support 64-bit little-endian PE, ELF and thin Mach-O headers; unsupported formats (including fat Mach-O) fail explicitly. This is header compatibility evidence, not runtime platform certification. Unix execute bits are applied during installation and checked by inventory.

HTTP uses HTTPS or loopback-only HTTP, no redirects, 5s connection and 30s request deadlines, at most three sources and two attempts each. Reads are bounded by expected byte count; cancellation is checked between reads (a blocked request can take up to its 30s deadline). Download progress is emitted at most 10 times per second. An already verified selection performs no HTTP. Catalogs are trusted configuration; there is no catalog-signature trust service in this local development phase.

Development package generation is explicit and offline:
  python -B tools/resources/package_local.py --assets <converted-assets> --media <ffmpeg-build-root> --probe <built-feathertalk-probe.exe> --probe-build <actual-build-record.json> --cuda <cuda-archive-root> --source <read-only-FeatherTalk> --supplied <read-only-models> --output <local-package-root> --catalog resources/development-catalog.json
The script only reads input bytes and writes its output packages. For the development integration test use .artifacts/task11-input as output. It preserves Gyan FFmpeg GPLv3 LICENSE, README/build configuration, docs and presets; selected tools are ffmpeg and ffprobe. Model notice attachments explicitly preserve historical records without applying old HuBERT licensing claims to current HuBERT. CUDA math support includes the actual libdevice bitcode and NVIDIA LICENSE. The base catalog includes the local CPU/Flex probe and deliberately leaves workers unconfigured.

To add built worker/probe pairs to a separate local catalog:

```text
python -B tools/resources/package_runtime.py --base-catalog resources/development-catalog.json --base-imports .artifacts/task11-input/imports.json --lockfile Cargo.lock --output .artifacts/runtime-packages --build <cpu-build-directory> --build <cuda-build-directory>
```

Each build directory must contain the two `feathertalk-worker` and `feathertalk-probe` executables (with `.exe` on Windows) and its actual `BUILD.json` record, including the Rust target, backend, build command, source revision and `exit: 0`. The output directory must be new. The script retains existing component versions and updates only the matching target's worker/probe pins. Versions derive from exact manifest bytes; runtime identity includes the target and every payload's path, size and SHA-256. Record files are build provenance, not redistribution authorization. Import the resulting catalog and `imports.json` into a local cache with `ResourceManager`.

`ResourceManager::bind_worker` validates all selected runtime components and creates an `InstalledWorker` from the installed paths. For resume and generation, it reads the verified committed checkpoint manifest and selects a single retained model bundle with matching preprocessing and VGG identities. A missing or newer default bundle does not override the checkpoint. Generic model identity lookup excludes media and executable packages; runtime components resolve by their exact pins.

The ignored `installed_worker` integration tests require `FT_WORKER_ACCEPTANCE_ROOT` (a directory containing `packages`, or set `FT_WORKER_ACCEPTANCE_PACKAGES`), `FT_WORKER_ACCEPTANCE_VIDEO` (the local derived four-frame fixture) and `FT_WORKER_ACCEPTANCE_BACKEND` (`cpu` or `cuda`). Run them separately per backend with `cargo test --offline -p feather-talk-gui --test installed_worker installed_train_cancel_crash_and_restart -- --ignored --nocapture`. The tests create isolated projects under the acceptance root; they use the local cache and neither download nor run Python during training or generation.

CUDA_PATH may be set only on an owned worker/probe child to the verified cuda-math root. This module neither changes global environment nor probes inference devices. CUDA driver availability remains a separate runtime requirement.

## Build an installer or run the desktop in development

Prepare `.artifacts/bundle-inputs` (or set `FT_BUNDLE_INPUTS` to its location). The same public model package is used on every target. Copy the **packaged** `public-assets` directory, with its manifest and inventoried notices, from the output of `package_local.py`. Use the current FeatherTalk converted models; a directory containing only weights is not a package. `imports.json` can be as short as `["public-assets/manifest.json"]`. Relative paths are based on the directory containing `imports.json`; absolute paths also work. If the imported manifest comes from a different catalog generated for this model package, include that file as `catalog.json` alongside `imports.json`. Otherwise the repository's development catalog pins the public assets.

```text
bundle-inputs/
  imports.json
  catalog.json                        # optional
  public-assets/                      # manifest, five models and notices
  <triple>/
    media/
      bin/ffmpeg[.exe]
      bin/ffprobe[.exe]
      LICENSE
      BUILD.json
    cuda-math/                        # Windows and Linux only
      nvvm/libdevice/libdevice.10.bc
      LICENSE
```

Use a native FFmpeg build with `libx264` and AAC encoding, preferably self-contained apart from system libraries. Use `.exe` names on Windows and thin Mach-O executables of the appropriate architecture on macOS. Include its license and actual version, configuration and source in `BUILD.json`. Use `libdevice` from the matching CUDA SDK with its license (CUDA 12.6 is the default worker build). The drivers themselves are installed on the destination system.

Run `pnpm tauri build --ci --no-sign` or `pnpm tauri dev` on the build host. Both default to the native Rust target; `--target <triple>` selects an entry from [targets.json](targets.json). The script builds locked release worker/probe pairs following the matrix: CPU, Vulkan and CUDA on Windows/Linux x64; CPU and Metal on Apple Silicon; CPU on Intel Mac. Set `FT_PACKAGING_PYTHON` to the Python executable if `python` is unavailable. `FT_BUNDLE_STAGE_DIR` can name a new staging directory; otherwise a new directory is made under `.artifacts/bundles`. Set `FT_BUNDLE_RUNTIMES` to reuse validated staging directories containing `<triple>/<backend>/BUILD.json` and both executables.

The generated `tauri-overlay.json` maps the target catalog and every pinned file (including notices) into `bootstrap` under the standard Tauri resource directory. Missing inputs, wrong executable architecture and mismatched file hashes fail the build. `build` creates installers in Cargo's target directory under `<triple>/release/bundle`. `dev` places the same resources beside the development binary. Both import the native selections on startup through the verified cache. Shared models and media are bundled and imported once per pin. An existing `resource-catalog.json` in the app's local data directory (`com.ft` by default) still takes priority; this only imports bundled components whose pins match that catalog. Keep it for a custom installation, or move it aside to use the installer defaults. Previously installed versions are retained for existing checkpoints.

On macOS, set `FT_COMPONENT_SIGNING_IDENTITY` for a signed build so worker/probe bytes are signed before hashing. Prepare already signed FFmpeg/FFprobe for signing and notarizing the final app. Review the model and component redistribution terms before uploading bundles or inputs. The CI workflow `Desktop component builds` can compile component pairs alone; for installers supply `bundle-inputs-run`, the run ID of an authorized run in the same repository with `bundle-inputs-<triple>` artifacts containing the layout above. The installer job uses the same command as a local build.

## Prepare a local hardware kit

Use `node tools/package/build-runtime.mjs --target <triple> --backend <cpu|cuda|vulkan|metal> --output .artifacts/staged` to compile a single pair. To stage binaries already built together, use `stage-components.mjs --target <triple> --backend <backend> --input <directory> --output .artifacts/staged` with the actual `BUILD.json`. Then run `node tools/package/build-catalog.mjs --artifacts .artifacts/staged --output .artifacts/desktop/catalog.json --base-imports <imports.json>`. The optional `--base-catalog <catalog.json>` accepts a configured catalog. Use `--resource-target <triple> --media <directory>` and `--cuda-math <directory>` for explicit platform inputs. `--bundle-target <triple>` additionally checks the complete installer dependency set and generates the overlay used by the normal build command. The output `packages/imports.json` records local paths for the hardware smoke kit. See [release checklist](../docs/validation/release-checklist.md) for how to move and run the kit.
