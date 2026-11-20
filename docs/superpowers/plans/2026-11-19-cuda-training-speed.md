# CUDA Training Speed Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task by task. Steps use checkboxes for tracking.

**Goal:** 修复长素材训练时每步耗时约 22 秒的问题，使用同一份素材和检查点在 RTX 2060 上比较修复前后。

**Architecture:** 训练开始时校验检查点绑定的预处理数据，任务内复用同一份数据。Burn 模型、Adam 和采样器只在完整更新成功后提交。Task 1 移除训练步中的缓存全量扫描；Task 2 进一步消除逐帧哈希、训练步的 Burn 记录往返和保存时的输入重扫。

**Tech Stack:** Rust、Burn 0.22.1、生产 worker 的 JSONL 协议、CUDA FP32。

**Spec:** [FeatherTalk 设计](../specs/2026-11-17-feathertalk-rust-migration-design.md)；[迁移计划 Task 17](2026-11-17-feathertalk-rust-migration.md#task-17-全链路基准与按设备启用优化)。

## Global Constraints

- 从 `main` 新建 `features/cuda-training-speed`；实验使用独立项目副本。
- 比较同一检查点、固定 batch size 1、相同随机种子、学习率、损失配置与设备。
- 保留训练数值、取消、错误回滚和检查点格式。素材与记录不提交到 Git。

## Review Focus

- 缓存启动前损坏：完整校验仍应拒绝。
- 训练期间某个抽样帧的路径或长度无效：该步应失败且不推进模型或采样器；等长变动留到新任务启动时检测。
- 不同检查点或内存特征：验证结果不可用于另一份数据。
- 取消时恰好有一步在执行：发布最后完整更新，续训应保持 Adam 和采样顺序。
- 计时区分首次 CUDA 编译、连续热步和检查点发布。

### Task 1: 定位并修复每个训练步重复读取整个缓存

**Files:** `crates/feathertalk-engine/src/training/{state,checkpoint,trainer,lifecycle,batch_probe}.rs`；`crates/feathertalk-engine/tests/training_lifecycle.rs`；`docs/validation/performance.md`。

**Interfaces:** 保留 `complete_step` 与 `TrainingState::validate_prepared` 的校验语义。新增引擎内部 `TrainingData::verify(state: &TrainingState, prepared: &PreparedProject)` 和 `complete_step_validated`，在一次训练内共享已验证的数据绑定。

- [x] 使用原版 release worker 在同一项目副本续训 6 步；保留热步耗时和 7 项 loss。临时分段计时确认每步四次扫描 1.08 GiB 缓存，CUDA 实际热计算约 0.95 秒。
- [x] `corrupted_sampled_frame_rejects_the_update` 验证坏帧拒绝该步；`unused_cache_artifact_is_rechecked_at_checkpoint_publication` 验证非训练用缓存损坏在发布时拒绝（原版在第 2 步前提前失败，已复现）。保留现有回滚、检查点损坏及续训测试。
- [x] 在任务边界验证整套数据，把训练步内部的 Burn 记录校验与外部文件校验拆开。回滚沿用独立候选，不修改原始状态。
- [x] `cargo fmt --all --check` 通过。CUDA 运行 `training_lifecycle`、`training_checkpoint`、`training_resume`、`probe_isolation` 共 10 项集成测试和 `--lib training::` 的 14 项单元测试；Flex CPU 重跑 `training_lifecycle` 的 3 项测试，均通过。CUDA release worker 热步中位数从 22.450 秒降到 1.513 秒；第 22 步检查点从新进程续训到第 25 步与连续训练结果一致。
- [x] 将结果记录在 [性能记录](../../validation/performance.md#60-秒素材上的-cuda-训练) 和 Task 17；提交 `perf: reuse verified training data across steps`。

### Task 2: 每次训练启动时校验一次

2026-11-20：任务 1 仍在每步校验帧哈希、往返序列化模型与 Adam，保存也会重读整套缓存。本次采用任务内信任：初始化或从磁盘续训时验证源文件、预处理缓存及传入的检查点；之后使用同一份已验证的数据和设备状态。跨任务持久缓存“已校验”会漏掉离线期间的改动；把所有帧复制到内存则会让内存随视频长度增长。

**Files:** `data/{prepared,sampler}.rs`、`training/{state,checkpoint,trainer,lifecycle,probe_transport}.rs`、worker `runtime.rs`、`tests/training_lifecycle.rs`、性能记录和迁移 Task 17。

**Interfaces:** worker 调用 `training::train_project`，从准备数据或加载检查点直接进入同一个训练循环。循环复用 `TrainingData` 和已验证的采样器输入；训练用 `save_training_checkpoint` 不重扫输入。独立的 `train`、`complete_step`、`read_frame` 及检查点加载仍验证从外部收到的输入。

- [x] 测试在首步后更改等长缓存还能完成并保存；重启加载必须拒绝损坏缓存。截断抽样帧仍应在读取时失败。用任务 1 的实现复现首个测试失败（`first-only-red.log`）。
- [x] `train_project` 每次任务只加载并验证一次完整缓存；采样使用已验证的帧索引，保持路径和长度检查。训练更新用独立的 Burn 模型和 Adam 候选，省去步间 Burn 文件往返；保持 loss 和梯度的数值保护。
- [x] 任务内定期及取消保存验证新产生的 Burn 记录并写入摘要，保留项目归属、锁和原子发布；外部导入继续验证所有记录。Auto 探测子进程作为独立接收者，加载时验证一次。
- [x] `cargo fmt --all --check`、CUDA 10 项集成回归和 14 项训练单元测试、Flex 5 项训练回归、14 项数据测试与 8 项 worker 协议测试均通过。用同一素材、检查点和 release 设置复测，热步中位数 0.727–0.836 秒，取消保存 0.931–1.097 秒。两轮对照的模型和 Adam 记录相同，step 22 → 25 的重启续训对照完全一致；首轮独立 CUDA 进程出现的浮点差异也保留在 [性能记录](../../validation/performance.md#每次训练启动时校验一次)。迁移计划 Task 17 已更新，提交 `perf: verify training inputs once per task`。
