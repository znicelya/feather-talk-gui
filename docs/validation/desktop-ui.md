# Windows desktop UI validation

The project, resource and device views and shared task diagnostics use shadcn/ui (`radix-nova`), Tailwind 4 and the bundled Geist font. `components.json` points the official CLI to `src/components/ui` and the shared theme in `src/App.css`. Task 15 and Task 16 use those same primitives.

`pnpm test --run` covers single-video selection and replacement, resource preparation and the busy guard, local readiness, and structured diagnostics. `pnpm test:contracts`, `pnpm build`, and `cargo build --offline --locked -p feather-talk-gui --bin feather-talk-gui --features tauri/custom-protocol` check the bridge and embedded desktop build.

On Windows, the WebView2 test used an isolated debug data directory and the [local CUDA test packages](desktop-bridge.md). It downloaded one missing probe package from a fixture HTTP server, verified its manifest and payload, and discovered the RTX 2060. After the server stopped, a new resource check still showed the local CUDA set ready. Selecting the CPU set with a deliberately unconfigured source produced the actual resource error and displayed its stage, code and suggested action. Debug builds can take tens of seconds to rehash the large packages.

The empty project view supported Tab and Enter on the skip link. Native Windows dialogs selected a 145-character project directory and 154-character MKV path, both with Chinese characters and spaces, then created the project and reopened a relocated fixture with three checkpoints. At 1120, 820, 640, 390 and 320 CSS px, the create action could be scrolled into view without horizontal overflow. Closing the tested window also exited its process. Screenshots and fixture helpers are under `.artifacts/task14-logs/` locally.

The layout follows the design skill's applicable desktop checks for alignment, semantic colors, visible control labels, keyboard focus, narrow layouts and reduced motion. The marketing-site hero, photographs, effects and site-specific gate are out of scope for this local workspace; the verification above used the actual desktop WebView. The test establishes readiness without the fixture server; whole-machine offline, production downloads, and Linux/macOS rendering still need their own acceptance runs.

## Training and loss on Windows

The native WebView2 exercised both training modes with a short clip derived from the demo and the installed CUDA worker on an RTX 2060 (driver 560.94). **One-click training** started from a project without a checkpoint, preprocessed its single video, and used the Rust defaults (60 epochs, requested batch 8, learning rate 0.001). The auto probe selected an effective batch of 1 because this fixture has one adjacent frame pair. The first real optimizer step produced a finite total loss of 1.36531 and cancellation published a trained checkpoint.

**Advanced resume** used a checkpoint at step 1 from a run whose history already reached step 3. It produced a live, finite loss at step 2 (total 1.12325), and the shadcn Chart showed both the total and individual losses. Switching to the project list and back while the worker was active took 35 ms in this smoke run. The subsequent cancellation committed a checkpoint for that run. After closing and restarting the app, the history loaded as `1, 2, 3, 2` with cumulative timestamps `36925, 44531, 86713, 127285` ms; the source records were preserved. The chart stayed within the viewport at 1120, 390 and 320 CSS px.

The Rust regression covers real training from an older checkpoint and the 500 ms sample interval. The frontend tests cover default versus advanced submission, safe numeric input, cancellation, history pagination, and filtering by project, run and task. Chart drawing is limited to 2000 points; the disk retains the sampled records. This run demonstrates that the UI remains usable during training. A matched throughput comparison belongs to [Task 17](../superpowers/plans/2026-11-17-feathertalk-rust-migration.md#task-17-全链路基准与按设备启用优化). Visual quality still needs representative footage and the matching Python reference.

## Audio to MP4 on Windows

WebView2 opened the checkpoint saved by the fresh training above. The native open and save dialogs accepted Chinese filenames and directories with spaces. With the installed RTX 2060 CUDA worker, the page received audio decode, feature extraction, frame generation and encoding stages. The generation action was disabled while the worker owned the task.

Both the first export and the retry played in the app. FFprobe reported 29 H.264 frames at 25 FPS, 640×600, with 1.16 seconds of AAC audio. An invalid WAV produced the actual FFmpeg diagnostic and an audio-specific recovery suggestion; no MP4 appeared at the failed destination and the first export was unchanged. The page kept the selected audio in the form, required a new save selection on retry, and previewed only the new successful task. The UI was checked at 1120, 390 and 320 CSS px. The Rust integration test also verifies stage order, rejects an untrained checkpoint, and cancels a generation without replacing a successful output.

The local fixture scripts, logs and output are under `.artifacts/task16/`. Inline playback was verified for the even-size H.264 output; other platform WebViews and odd-size H.264 profiles still need testing. The UI offers to open the output directory if the WebView cannot play a file.
