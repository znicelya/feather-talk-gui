# Installed worker validation

The integration test launches a worker from a verified resource cache through `WorkerHost` and `ResourceManager::bind_worker`. It starts with a fresh project and a single derived, four-frame talking-head clip. Training runs the full FeatherTalk objective, and generation uses a separate WAV decoded from that clip. This is a functional fixture; it does not establish lip-sync or visual quality on real training footage.

Run `installed_train_cancel_crash_and_restart` as described in [the resource guide](../../resources/README.md). The test checks the actual reported backend, rejects a concurrent training request, receives a finite loss, cancels training, and checks that the emitted checkpoint is in the project index. It then force-stops a generation worker, checks the single failure event and preservation of the prior output and project index, and starts a new worker with the same checkpoint. The replacement worker uses the retained model bundle even when the default bundle is unavailable. FFprobe checks the new H.264/AAC MP4 at 25 fps, including export to a path with Chinese text and a space.

On Windows, the separate `owned_grandchild_dies_on_job_close` and `crash_releases_process_lock` tests exercise a live helper grandchild and recovery of the same preparation lock after process death. The generation crash occurs during audio feature extraction; it does not attempt to infer whether that particular worker had a live FFmpeg child at that moment.

The initial RTX 2060 run completed its first training step, then failed to save because a new project did not yet have a `checkpoints` directory. `training::lifecycle` now creates the directory on the first save and checks that its canonical path stays inside the project. The test does not pre-create it. The rebuilt packages passed on both Windows backends:

| Worker backend | Actual device | Batch policy exercised | Cancelled checkpoint | Restarted export |
| --- | --- | --- | --- | --- |
| CUDA | NVIDIA GeForce RTX 2060, driver 560.94 | Auto, target 1, probe selected 1 | Step 1 | 3 frames, 640x600, H.264/AAC, 25 fps |
| CPU | Burn Flex, host CPU | Fixed, 1 pair | Step 1 | 3 frames, 640x600, H.264/AAC, 25 fps |

Raw worker events are in `.artifacts/task12b/cuda-checked.log` and `.artifacts/task12b/cpu-checked.log`. `installed_binding_rejects_wrong_backend_and_changed_executable` also passed against the rebuilt local catalog. These are debug-build runs and are not throughput benchmarks.

The packages and logs are local development artifacts. Packaging has no production endpoint or redistribution approval. These tests use already available files and no network calls in the worker; they do not simulate an OS-level network disconnect. Linux, macOS, other GPU vendors, installation, cold-start downloads, and WebView playback require separate acceptance.
