# FeatherTalk Task 9 staged implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. This supplements Task 9 of the approved migration plan; continue in its existing worktree, branch, SDD ledger and review workflow. Steps use checkbox syntax. No scope or acceptance reduction.

**Goal:** Deliver actual paired Burn training, exact resumable state, atomic checkpoints and survivable Auto batch probing in independently reviewable stages.

**Architecture:** Shared request contracts precede full model/Adam persistence. A committed training-step API consumes that state and deterministic sampler, then the isolated probe process and public train loop compose it. Each stage has one implementer and fresh review; parent Task 9 completes only after all stages and hardware integration evidence.

**Tech Stack:** Existing pinned Burn0.22.1, Rust2024, Burnpack, current core/PreparedProject/PairSampler APIs.

**Spec:** docs/superpowers/specs/2026-11-17-feathertalk-rust-migration-design.md, sections7/12. Exact parent requirements: docs/superpowers/plans/2026-11-17-feathertalk-rust-migration.md Task9. Controller rulings project-root-ruling.md and task-9-probe-ruling.md in parent SDD workspace apply.

## Global Constraints

- All original migration constraints and exact Task9 values remain binding; training uses default FeatherTalk+all six losses+frozen VGG on selected actual device.
- Preserve latest complete model/BN/Adam/ParamIds/sampler/step seed/config/project/data identities. No fictitious portable raw backend RNG state.
- Atomic immutable checkpoint publication updates committed project index last; failure preserves previous checkpoint.
- Fixed batch is never probed/changed. Auto full-step probes <=target are disposable independent processes, no unknown crash classified as OOM.
- Original inputs/models read-only; artifacts/private media ignored; no product Python.
- Separate actual local CPU/RTX2060 results from pending external hardware/effects.

## Review Focus

- Removed/foreign optimizer moment or time must reject rather than silently restart Adam (9b).
- Interrupted staging/reference write must preserve previous committed state and ModelRef (9b).
- Prefetch and cancellation must not commit sampler/BN/Adam progress at different boundaries (9c).
- Cross-device valid/train/fork must keep buffers nontrainable and latest counters/ParamIds (9b/9c).
- A candidate abort/stale result/unknown exit must preserve parent and fail truthfully, successful probes must run complete objective (9d).

## Stage 9a: Explicit shared training settings

**Files:** core config.rs/protocol.rs/tests and focused new configuration test, optional tests-only dependency cleanup.
**Interfaces:** TrainRequest.batch_mode Auto/Fixed (serde defaultAuto); TrainConfig.seed default42, prefetch_capacity default2 with0disabled..32, existingbatch<=1024 to match sampler. Other defaults unchanged.

- [x] Step1: Record genuine focused RED for new setting/default contracts.
- [x] Step2: Implement exact defaults/serde/validation/roundtrip; distinguish config-only Fixed from trainer enforcement.
- [x] Step3: Run core tests/fmt/diff, self-review and scoped commit; fresh stage review.

## Stage 9b: Complete state and atomic checkpoints

**Files:** training/state.rs, checkpoint.rs and focused checkpoint validation/integrity helpers as needed, tests/training_checkpoint.rs. Extend training/mod.rs. Developer-only prepared/test data helper allowed if needed for meaningful fixed-data tests, never product historical import.
**Consumes:** reviewed 9a, FeatherTalk Module and all447 records, ModuleOptimizer, PreparedProject/PairSampler/SamplerState.
**Produces:** TrainingState (model/Adam/config/batch mode/target+actual/run/epoch/step/sampler/seed-policy/identity), constructor from scratch; save_checkpoint(&TrainingState,destination)->Result<ModelRef>; load_checkpoint(project_directory,model_ref,device)->Result<TrainingState>. Public helpers may expose complete validated snapshot bytes for isolated reconstruction; exact final API recorded for9c/9d. No trainer loop/probe executable.

- [x] Step1: Failing tests for complete roundtrip including newest BN/counter/ParamIds; missing moment/time/foreign ID and wrong data/project identity reject; interrupted staging leaves old checkpoint/reference usable.
- [x] Step2: Implement source-compatible fresh state + versioned deterministic application seed policy metadata, full Burnpack inventory prevalidation and atomic publication. ModelRef resolves project.json committedindex under explicitroot. Store no arbitrary checkpoint path in ModelRef.
- [x] Step3: Test real model state/moments roundtrip and focused small real optimizer state mutation, CPU+availableGPU state load/finite continuation when feasible; direct BN lifecycle transition Minor. State-only protocols may use small real modules but must not claim full objective trained.
- [x] Step4: Record exact covering evidence, self-review, scoped commit and fresh review before9c.

Execution note: initial RED covered the missing state/seed module; most full protocol tests were introduced alongside implementation. Passing behavior coverage and the learning-rate fix RED/GREEN are recorded in task-9b-report.md; this does not claim a complete preimplementation RED sequence.

## Stage 9c: Committed training step, sampled metrics and exact continuation

**Files:** training/trainer.rs, metrics.rs, tests/training_resume.rs and training_convergence.rs.
**Consumes:** reviewed9b state/checkpoints, current PreparedProject+PairSampler, VggFeatures/compute_loss.
**Produces:** complete_step(state,prepared,vgg,device,cancel)->Result<StepOutcome> (concrete name/signature implementer records); sampled history writer, real same-device/cross-device continuation test helpers. Public train loop may wait for9d; do not publish a fake Auto implementation.

- [x] Step1: Failing tests for state commit after complete optimizer step, failed/canceled-before-step sampler rollback, nextstep continuousvsresume parameters/moments/time/BN/indices/references/seeds.
- [x] Step2: Implement actual paired full-loss backward/Adam, explicit learning settings, serialized per-step RNG seed and staged sampler progress, next epoch transition without droppedtail, sampled metrics <=2Hz with durable bounded page support.
- [x] Step3: Run minimum meaningful full-model objective finite/decreasing short overfit on CPU and actualGPU, same-device exact continuation and CPU<->GPU finite continuation with tolerances. Keep VGG frozen. Real footage/effect pending not simulated.
- [x] Step4: Cover amended code, report APIs/evidence, scoped commit and fresh review.

Execution note: initial RED established the missing API; broader behavioral cases were added alongside implementation. The VGG dependency fix has explicit behavioral RED/GREEN. Actual CPU/CUDA/Vulkan full-objective results and evidence limits are recorded in task-9c-report.md.

## Stage 9d: Isolated probing and public train lifecycle

**Files:** training/batch_probe.rs, minimal src/bin/feathertalk-probe.rs + focused transport/snapshot helper, integrate trainer.rs periodic/cancel/finalsave; tests/probe_isolation.rs and remaining Task9 tests.
**Consumes:** reviewed9b snapshot +9c complete step, explicit TrainingRuntime{probe_runner:PathBuf} and ProbeContext per task-9-probe-ruling.md.
**Produces:** probe_batch_size and train with explicit runtime injection; TrainingResult with committedModelRef/stoppingreason.

- [ ] Step1: Failing tests for Fixedneverprobe, Auto <=target, stateunchanged+nextrealstepmatch, processfaultclassification/cleanup/staleresultrejection, cancellationcommitaftercompletedstep, interruptedrecovery.
- [ ] Step2: Reconstruct full state in disposable candidate process, actual fullVGGbackwardAdam, request-bound atomic typedresult, narrow evidencedOOMretry and unknowncrasherror, hiddenWindowschild/killwait/ownedcleanup. No deliberate machineOOMtest.
- [ ] Step3: Public loop createsnewrun/preservesresumedrun, targettotalepochs, periodic+cancel+finalimmutablecheckpoints, history sampling, manualOOMsuggestion. Complete actualCPU/GPU trainandresume/probesuccess integration withtruthful logs.
- [ ] Step4: Cover final APIs, full parentTask9 requiredchecks, self-review and scopedcommit/freshreview. Only then mark parentTask9 complete and startTask10.

This partition changes implementation granularity only. Missing external footage/devices remain explicit pending acceptance; no stage substitutes compilation/protocolfixtures for real full-objective execution.

