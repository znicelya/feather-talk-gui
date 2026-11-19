# FeatherTalk Task 12 staged worker implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. This supplements Task12 of the approved migration plan; use the SAME worktree/branch/parent SDD workspace/ledger. Serial fresh implementer and independent review per stage. No scope or acceptance reduction.

**Goal:** Isolate real Burn training/generation in a single-task worker with bounded protocol, owned process-tree cleanup, verified components and truthful restart acceptance.

**Architecture:** Shared pure core framing/context and nonterminal request rejection feed a worker event owner and explicit host bootstrap. After that producer is independently reviewed, real resource-bound worker/probe artifacts and actual neural process jobs close the integration boundary. ParentTask12 remains incomplete until both stages/reviews and available-host acceptance finish.

**Tech Stack:** Current Rust2024 workspace, Burn0.22.1 worker-only, Tauri host without Burn, strict serde NDJSON, fs2 OS lock and Windows Job/Unix process group APIs.

**Spec:** docs/superpowers/specs/2026-11-17-feathertalk-rust-migration-design.md sections4/7/8/12; parent docs/superpowers/plans/2026-11-17-feathertalk-rust-migration.md Task12. Parentworkspace task-12-rulings.md/project-root-ruling.md/task-9-probe-ruling.md apply.

## Global Constraints

- Shared protocol version1, NDJSON maximum1MiB enforced before unbounded accumulation; stdout protocol only, stderr bounded evidence.
- One heavy job, one serialized event owner/sequence, exactly one terminal; RequestRejected is nonterminal control without task ID. Sequence is strictly increasing per worker session.
- Control remains responsive during actual compute; normal cancellation waits coherent completed update/checkpoint, crash only advertises committed state.
- Explicit trusted project/resource/bootstrap paths; no cwd/latest-assets/build paths or product Python. GUI must not depend on Burn.
- Windows hidden worker/media processes, owned Job kill-on-close; Unix owned process group. Never kill unrelated executable names.
- Actual CPU/RTX2060 evidence distinct from unavailable external hardware/OS, production catalog authority and real-footage effect acceptance.

## Review Focus

- Wrong-task/late/duplicate events cannot contaminate or terminate active work; raw Busy rejection does not release owner.
- Oversized newline-free input stays bounded; malformed/EOF produces truthful transport failure with last stage and stderr.
- Worker abnormal exit kills owned grandchildren and releases preparation lock without deleting unrelated state.
- Old checkpoints bind original prepared/model/VGG resources, not current UI selection/latest bundle.
- Installed runtime artifact identity comes from actual payload hashes/target/backend, not a shared LICENSE hash or a declared catalog label alone.

## Task 12a: Protocol, runtime and host ownership producers

**Files:** core src/worker.rs/protocol.rs and tests; new worker Cargo.toml/src/{lib,main,runtime,transport}.rs/tests/worker_protocol.rs; GUI worker_host.rs/worker_process.rs and tests; minimal engine prepared.rs OS-lock amendment. Small context validation helper allowed to keep responsibilities focused.
**Consumes:** reviewed Task9 training, Task10 generation, Task11 resource contracts and pure shared core.
**Produces:** WorkerRuntime::handle(WorkerRequest)->Result<()>, WorkerHost::{spawn(component,&explicitcontext),start,cancel,next_event} (record exact owned signatures), typed bounded bootstrap/context, actual worker runtime dispatch to prepare/train/generate, resource/path/identity validation, hidden owned process lifetime. CPU build usable by raw process tests; package artifacts may remain developer-only and not full-app ready.

- [x] Step1: Capture meaningful framing/event-owner RED, implement strict RequestRejected/version/context and bounded IO with actual behavioral assertions.
- [x] Step2: Implement single-task control/compute separation, session sequence/terminal owner, actual engine dispatch and callback rejection; host accepts only correct owner/sequence and retains actual failure evidence.
- [x] Step3: Verify raw process Hello/repeated Hello/Busy/wrong Cancel/protocol version/fragmented and oversized input, stdout/stderr split and host crash context; no simulated process counted as neural acceptance.
- [x] Step4: Verify real Windows helper child/grandchild termination, cancellation/control boundaries, persistent OS prepare lock concurrent Busy and actual process-kill/reacquire; preserve staged cache/checkpoint semantics.
- [x] Step5: CPU worker/probe build and focused core/worker/host checks, self-review/API report/commit then fresh stage review. Actual full-resource neural jobs and backend packaging remain12b, not placeholder PASS.

Stage 12a passed scoped independent re-review at `b1ab4df`: all seven original findings were addressed. The saved tests include 12 worker, one host fixture matrix, four preparation-lock checks, and the live Windows Job grandchild check. The unperformed installed-context test is not included in these counts. Historical regression reversals are identified in the implementation report.

## Task 12b: Verified runtime artifacts and real worker job acceptance

**Files:** worker/context/runtime/host focused integration refinements; tools/resources/package_local.py and actual development catalog metadata; tests worker_protocol/worker_host plus focused runtime job fixture; parent Task12 report. Reuse12a modules; no new protocol/engine implementation duplication.
**Consumes:** independently reviewed12a producers, Task11 ResourceManager exact component pins/installed identities, Task10 actual derived singleface functional project and committed step1 checkpoint.
**Produces:** actual target/backend worker+matching probe components, verified WorkerContext construction from installed resources, real Train/Generate process roundtrip and abnormal-restart evidence; parentTask12 completion only after review.

- [x] Step1: Write failing real boundary tests for missing/wrong target/backend/payload binding and retained project resource resolution; correct nonmodel component identity to complete stable payload inventory or explicitly restrict model-only lookup before consuming it.
- [x] Step2: Build actual CPU and available RTX2060 backend worker/probe artifacts, generate actual hash/size/target/backend notices/provenance manifests and import via ResourceManager. CUDA receives only explicit verified installed libdevice root; full readiness remains false for missing mandatory resources.
- [x] Step3: Send actual Train and Generate JobSpec using accepted installed context, validate progress/metrics/checkpoint/Completed/Cancelled once, cancellation during compute saves coherent state and remains responsive. Runtime uses no product Python/download, with host-disconnect test limits explicit.
- [x] Step4: Force exit after actual committed checkpoint; verify old checkpoint still valid, all owned children gone and new worker/new job succeeds with same project/resource identities and OS prepare lock. Cross-project/wrong-context rejects before work.
- [ ] Step5: Cover amended files, consolidate task-12-report.md APIs/rulings/raw commands/exits/actual devices/external limits, scoped commit/fresh review. Only then mark parentTask12 complete and start13. External Linux/Mac/other-vendor effects remain unrun, never inferred from compilation.

Partition changes review granularity only. No user reapproval is required for this internal sequencing of already authorized work. Historical tests/evidence remain in parent SDD ledger; do not create a second forgotten execution workspace.
