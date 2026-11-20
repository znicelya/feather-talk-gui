# FeatherTalk Rust Migration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 GUI 项目中交付无需 Python 运行环境、支持三系统和指定硬件的 FeatherTalk 本地训练与离线视频生成应用。

**Architecture:** React 通过 Tauri 调用一个串行任务服务，任务由独立 Rust worker 执行。共享 core crate 定义项目、配置、资产与事件协议，engine crate 实现 Burn 模型、训练、数据与媒体处理；按硬件选择后端，并保留未来音频来源/视频接收端的扩展边界。

**Tech Stack:** Rust 2024、Burn（候选 0.22.1，关键算子验证后锁定）、Tauri 2、现有 React 19/TypeScript/Vite/pnpm、FFmpeg/FFprobe。Python/PyTorch/ONNX 仅用于开发阶段转换和参考验证；Rust 集成测试、pytest、Vitest/Testing Library 与浏览器/桌面实测负责不同边界。

**Spec:** [已确认设计](../specs/2026-11-17-feathertalk-rust-migration-design.md)。执行者先读设计再读本计划。全部相对路径以 `feather-talk-gui/` 为根目录。

**当前进度（2026-11-19）：** Task 1–16 已完成，含 [Windows CPU/CUDA worker](../../validation/installed-worker.md)、[Tauri 桥接](../../validation/desktop-bridge.md) 与 [shadcn/ui 桌面流程](../../validation/desktop-ui.md)。Task 17 已完成本机 CUDA/Vulkan/Flex 同输入测量。Task 18 已复验 Windows NSIS 的空缓存首启，CPU 和 RTX 2060 Vulkan 的 release worker 均跑通从头训练、取消保存、重开续训及导出。B580 便携包已从新目录在 RTX 2060 复跑，待 Win11/B580 实测。正式首启下载、整机断网、真人效果和其余平台仍开放。见 [性能记录](../../validation/performance.md)、[平台矩阵](../../validation/platform-matrix.md) 与 [B580复跑步骤](../../validation/release-checklist.md)。

## Global Constraints

- 训练与推理均使用 Burn；GPU 必须在对应显卡实际执行，Intel Mac 使用 CPU。
- 首版仅默认 FeatherTalk 人物网络，每个人物一段训练视频，从头训练，使用新的 Burn 检查点；PFLD 内部 GhostOne/MobileOne 模块仍在范围内。
- NVIDIA 比较 CUDA/Vulkan；AMD、Arc 使用 Vulkan；Apple Silicon 使用 Metal；另有可训练、可推理的 CPU 路径。
- 首轮打包验证：Win11 x86_64、Ubuntu 24.04 x86_64、macOS 13+ arm64/x86_64；最终最低系统版本以依赖与实测结果固定。
- 保留 152×152 裁切、中心 144×144、BGR、40×1024 音频窗口、相邻帧共享参考图、原版全部损失及图像/音频对齐语义。
- 一键默认 60 epochs、目标 batch size 8、Adam lr=0.001、beta1=0.9、beta2=0.999、epsilon=1e-8、weight decay=0、每 5 epochs 保存；epochs 表示总轮数。
- 损失系数：全图 1、嘴部 4.0、时序 0.5、嘴部时序 4.0、VGG 0.01、嘴部梯度 0.5；时序低通核 5、步长 1。
- 一次只运行一个训练或生成任务；实时 loss 向 UI 推送上限为每秒 2 次；支持检查点续训、取消保存和可操作错误提示。
- 首次可下载资源，准备完成后断网训练/续训/导出；公共权重、组件与项目缓存按版本及哈希管理。
- 当前 FeatherHuBERT 原权重哈希为 `3b147e66502132bac3ff76b4730c2a83c341c75251d22a28182a5a1b689cfc46`，重新转换，不复用来源不同的 188 版音频资产。
- 不覆盖同级 Python 原文件或原始 `models` 输入。转换输出、参考张量、性能结果进入本项目 `.artifacts/`；小型配置、生成工具和生成源码入 Git，大权重和私人素材不入 Git。
- 实时采集/播放、旧人物模型导入、MobileOne 人物网络、多视频、多任务并行和分布式训练不进入首版。

## Review Focus

1. 新音频权重与旧特征缓存混用：资源身份变化必须使特征缓存失效，见 Task 2、8。
2. 人脸靠边、检测不到或坐标取整不同：保留补边、均值脸与原坐标规则，失败定位帧号，见 Task 5、8。
3. 保存中断、worker 崩溃或取消：已提交检查点/已有成品仍可用，未完成状态不可伪装成功，见 Task 9、10、12。
4. 双击开始、迟到事件、切换项目和重连：保持单任务且曲线/进度归属正确，见 Task 12、13、15。
5. 下载截断/损坏后断网，以及中文/空格路径：完整资源可离线使用，缺损资源明确提示且旧版本仍可用，见 Task 10、11、18。

## 执行顺序、文件与共用约定

任务依赖：`1 → 2 → 3 → (4,5,6,7) → 8 → 9 → 10 → 11 → 12 → 13 → 14 → 15 → 16 → 17 → 18`。任务 7 依赖 Task 6 的人物前向和 Task 2 的 VGG 参考；Task 8 集成 4/5；最终验收必须包含全部任务。括号表示依赖关系，不指定采用多代理或并行执行。

文件职责：

| 路径 | 职责 |
| --- | --- |
| `Cargo.toml`、`Cargo.lock` | 根 workspace、锁定依赖与公共构建设置，纳入 `src-tauri` 和三个 crate |
| `crates/feathertalk-core/src/{config,project,assets,protocol,error}.rs` | 可序列化契约与业务约束，不依赖 Burn/Tauri |
| `crates/feathertalk-engine/src/{device,models,ops,data,media,training,inference,benchmark}/` | 实际数学计算、媒体管线和性能测量 |
| `crates/feathertalk-worker/src/{main,runtime,transport}.rs` | 进程协议与任务执行，stdout 仅协议、stderr 日志 |
| `src-tauri/src/{commands,state,worker_host,resources,projects}.rs` | 桌面应用服务、资源和 worker 生命周期 |
| `src/{lib,features,components}/` | 类型化桥接、人物/训练/生成/资源功能与共用显示 |
| `tools/reference/`、`tools/modelgen/`、`tools/acceptance/` | 可重现转换、对齐与实机报告工具 |
| `.artifacts/` | 本地权重、张量、测试视频、基准与打包暂存，忽略 Git |

Rust 中标注为 `Result<T>` 的自有接口均使用 core 的 `type Result<T> = std::result::Result<T, AppError>`；张量前向等明确返回普通值的接口维持其签名。接口中的 `Device`/`Tensor<N>` 是锁定版本 Burn 的类型；不照搬旧版本带 backend 泛型的示例。所有任务的验证命令均从项目根目录运行。以下 `python` 表示 Task 2 准备的隔离解释器，不要求修改全局 PATH。`FT_REFERENCE_DIR`/`FT_ASSETS_DIR` 未设置时，测试默认使用根目录 `.artifacts/reference`/`.artifacts/assets`。

每个任务先建立有意义的失败验证，再实现并运行同一验证，最后仅提交该任务相关文件。普通界面排版和文案调整通过浏览器核对，不为静态样式写实现镜像测试。GPU 专项显式执行并写报告；未运行不计通过。下文预计 FAIL 是首次引入对应契约/能力时的结果，不允许为了红测破坏已有功能。

## Task 1: 可验证的配置、项目和协议契约

**Files:** Create `Cargo.toml`、`crates/feathertalk-core/Cargo.toml`、`crates/feathertalk-core/src/{lib,config,project,assets,protocol,error}.rs`、`crates/feathertalk-core/tests/contracts.rs`、`contracts/protocol-v1.json`；Modify `.gitignore`、`src-tauri/Cargo.toml`；生成根 `Cargo.lock`。

**Interfaces:** 产出 `TrainConfig::default()`、`TrainConfig::validate(&self) -> Result<()>`、`LossConfig`（全局约束中的命名权重）、`DeviceRequest { backend: BackendKind, device_id: String }`、`DevicePolicy::{Auto,Explicit(DeviceRequest)}`、`BackendKind::{Flex,Cpu,Vulkan,Cuda,Metal}`、`DeviceDescriptor { id, name, vendor, driver, backend, memory_bytes }`、`ProjectManifest`、`NewProject { name, directory, source_video }`、`ProjectSummary`、`AssetManifest`、`AppError { code: ErrorCode, stage, message, evidence, actions, log_path }`。ErrorCode 包括 InvalidConfig/ModelMismatch/ResourceMissing/ResourceCorrupt/UnsupportedBackend/DeviceInit/DeviceLost/OutOfMemory/NoFace/NoAudio/EncodeFailed/DiskFull/PermissionDenied/Busy/Protocol/Cancelled/Internal，JSON 使用 snake_case。`TaskId`/`ProjectId`/`TrainingRunId` 为 UUID 包装；`JobSpec::{Train(TrainRequest),Generate(GenerateRequest)}`。`TrainRequest { project_id, config, device: DevicePolicy, resume }`；`GenerateRequest { project_id, audio_path, output_path, model_ref, device: DevicePolicy }`；`ModelRef` 指向已提交检查点。协议 `WorkerRequest::{Hello,Start,Cancel,Shutdown}` 与 `WorkerEvent { protocol_version: 1, task_id: Option<TaskId>, seq, payload: EventPayload }`；payload 包括 Ready/Started/Progress/Metrics/CheckpointCommitted/Completed/Cancelled/Failed。Ready 携带组件能力和 `Vec<DeviceDescriptor>`，无任务 ID；所有任务事件必须有 ID。Progress 含 stage/completed/total；Metrics 含 project_id/run_id/step/epoch/elapsed/lr/各 loss/设备/精度/实际 batch，数值 loss 为 `LossScalars { full, mouth, temporal, temporal_mouth, perceptual, mouth_gradient, total }`。core 同时定义 `TaskObserver::emit(&self, payload: EventPayload) -> Result<()>` 与可克隆 `Cancellation::new() -> Self`、`cancel(&self)`、`is_cancelled(&self) -> bool`。

- [x] **Step 1:** 在 `contracts.rs` 写 `defaults_match_python`、`rejects_invalid_hyperparameters`、`protocol_roundtrips`、`project_accepts_one_video`；断言 `epochs=60, batch_size=8, lr=0.001, save_every=5`，损失系数逐项等于全局约束；batch=0、偶数低通核、多视频数组输入拒绝；事件序列化保留任务 ID 和全部分项。关键断言：`assert_eq!(TrainConfig::default().batch_size, 8); assert_eq!(TrainConfig::default().epochs, 60);`。
- [x] **Step 2:** 建立该测试所需最小 workspace（先仅 core 与现有 src-tauri，engine/worker 在各自任务加入），运行 `cargo test -p feathertalk-core --test contracts`，预期因接口尚未实现 FAIL。
- [x] **Step 3:** 实现上述契约、校验及 snake_case JSON 标签；`ProjectManifest` 保存素材/预处理/模型身份、检查点索引和曲线位置。定义 `ModelIdentity { kind, architecture_version, source_sha256, weights_sha256 }` 与 `AssetManifest { identity, relative_files, sizes, hashes, urls, min_engine_version, provenance }`。现有 release profile 转入 workspace 根配置；锁定 workspace 解析，不无故更新已有前端依赖。

  `Completed` 使用 core 的 `ArtifactResult::{Model(ModelRef),Video(ExportResult)}`，其中 `ExportResult { output_path, video_frames: u64, elapsed_ms, device: DeviceDescriptor }`；任务快照与指标分页也放 core protocol，不让 Tauri 因结果类型依赖 engine。
- [x] **Step 4:** 重跑同一测试，预期全部 PASS；`cargo check -p feather-talk-gui` 保持 Tauri 可编译。
- [x] **Step 5:** 提交 `feat: define versioned project and worker contracts`，包含实际需要的 workspace/忽略项与锁文件调整。

## Task 2: 原权重审计、重新转换和参考张量

**Files:** Create `tools/reference/{README.md,requirements.lock,audit.py,convert.py,generate.py,fixtures.json}`、`tools/reference/tests/test_assets.py`、`tests/reference/manifest.json`。

**Interfaces:** `audit_sources(source_dir: Path, supplied_models: Path) -> dict`；`convert_current_assets(source_dir: Path, supplied_models: Path, output_dir: Path) -> Path` 返回资源清单路径；`generate_reference(source_dir: Path, assets: Path, output_dir: Path) -> Path` 返回包含输入/输出/梯度/配置/版本/阈值的参考索引。CLI 使用 `--source ../FeatherTalk --models ../models --output .artifacts/...`。Rust 后续通过 `FT_REFERENCE_DIR` 读取参考目录，通过 `FT_ASSETS_DIR` 读取转换后的资源。

- [x] **Step 1:** 准备隔离 Python 环境并锁定兼容的 CPU PyTorch/torchvision、safetensors、onnxruntime、numpy、OpenCV 和 pytest；写 `test_rejects_188_source`、`test_exports_checkpoint_configuration`、`test_mean_face_provenance`：旧来源不得标为 current；config 来自实际 checkpoint；均值脸为 220 值且哈希为 `40bed2653ae861e9a24b4766c3d085986b7c1245a27373913a9a5b1a6ce069ef`。关键断言：`assert audit["feather_hubert"]["source_sha256"] == "3b147e66502132bac3ff76b4730c2a83c341c75251d22a28182a5a1b689cfc46"; assert audit["mean_face"]["count"] == 220`。
- [x] **Step 2:** `python -m pytest tools/reference/tests/test_assets.py -q`，预期转换/审计能力缺失 FAIL。
- [x] **Step 3:** 实现转换与参考生成，原文件只读。重新导出当前 FeatherHuBERT；核验其他三组张量名称/形状/来源，不兼容则从原来源重建。将均值脸作为辅助资源；VGG 只导出原训练使用的 features[0..14]。生成可复现输入、固定人物测试参数、公共网络关键层、几何/音频边界、人物前向、分项损失与梯度参考；所有二进制 SafeTensors/媒体进 `.artifacts`，索引与工具入 Git。`fixtures.json` 记录真实效果验收素材路径/哈希，素材尚未提供时明确记录未完成的效果验收，不以随机张量替代画质验收。
- [x] **Step 4:** 重跑 pytest，预期 PASS；运行 `python tools/reference/audit.py --source ../FeatherTalk --models ../models`、`python tools/reference/convert.py --source ../FeatherTalk --models ../models --output .artifacts/assets`、`python tools/reference/generate.py --source ../FeatherTalk --assets .artifacts/assets --output .artifacts/reference`，预期索引源哈希正确、每项张量可重读且重复生成一致。基于 FP32 误差冻结逐项容差到参考索引。
- [x] **Step 5:** 提交 `feat: add reproducible model conversion and reference fixtures`，不提交大权重或人物素材。

## Task 3: Burn 版本、设备探测与关键算子门槛

**Files:** Create `crates/feathertalk-engine/{Cargo.toml,src/lib.rs,src/device/mod.rs,src/ops/mod.rs,src/ops/resize.rs,src/bin/backend-check.rs,tests/operator_parity.rs}`、`docs/validation/backend-capabilities.md`。

**Interfaces:** `open_device(request: &DeviceRequest) -> Result<ExecutionDevice>`；`ExecutionDevice { device: Device, descriptor: DeviceDescriptor }`，使用 core 的 descriptor。`probe_devices() -> Result<Vec<DeviceDescriptor>>`；`resolve_device(policy: &DevicePolicy, descriptors: &[DeviceDescriptor]) -> Result<DeviceRequest>` 提供确定性的初始设备解析，后续由 Task 17 的已验证基准增强。`run_operator_checks(device: &ExecutionDevice, reference: &Path) -> Result<OperatorReport>`；`OperatorReport { device: DeviceDescriptor, checks: Vec<ParityCheck> }`，`ParityCheck { name, passed, max_abs_error, max_rel_error }`。自有 `resize_bilinear_align_corners(input: Tensor<4>, output_hw: [usize;2]) -> Tensor<4>` 必须可反向。

- [x] **Step 1:** 写 `operator_parity.rs` 的 `resize_forward_and_backward_match`、`normalization_train_eval_match`、`grouped_dilated_conv_gradients_match`、`avg_pool_padding_and_attention_match`。分别加载 Task 2 的期望值并断言逐项容差、有限/非零梯度、BN 运行统计与 train/eval 行为；设备报告必须匹配请求的实际执行设备。关键断言：`assert!(report.checks.iter().all(|check| check.passed)); assert_eq!(report.device.id, request.device_id);`。
- [x] **Step 2:** 用最小 engine 壳运行 `cargo test -p feathertalk-engine --features flex --test operator_parity`，预期缺失算子或未对齐时 FAIL。
- [x] **Step 3:** 使用候选 Burn 0.22.1 验证 API；按平台配置 flex/cpu/vulkan/cuda/metal feature。缺失能力实现可验证的 Burn/CubeCL 扩展；禁止删损失绕过。若候选版本不可用/关键功能不成立，记录证据、选可行版本、重生成匹配源码和参考存储，再锁定 Burn 相关精确版本。CPU 要明确执行在 host CPU，不以软件 GPU 适配器代替；分发基线能够在本机 i5-10400 执行，SIMD 按实际可用能力选择。
- [x] **Step 4:** 重跑 CPU 测试并运行 `cargo run --release -p feathertalk-engine --features vulkan --bin backend-check -- --backend vulkan --reference .artifacts/reference --output .artifacts/reports/operators-vulkan.json`；CUDA 使用 `--features cuda`/`--backend cuda`，Mac 使用 metal，CPU 使用 flex。当前可用设备先实测，其余能力表记录未运行；通过结果含设备、数值误差与梯度。
- [x] **Step 5:** 提交 `feat: validate Burn operators and execution backends`。本任务未通过的数学能力不能带入模型集成。

## Task 4: 当前 FeatherHuBERT 前向与音频协议

**Files:** Create `crates/feathertalk-engine/src/models/{mod.rs,feather_hubert.rs}`、`src/data/audio.rs`（均在 engine crate 内）、`crates/feathertalk-engine/tests/hubert_parity.rs`。

**Interfaces:** `FeatherHubert::load(assets: &Path, device: &Device) -> Result<Self>`；`forward(&self, waveform: Tensor<2>) -> Tensor<3>`；`normalize_waveform(samples: &[f32]) -> Result<Vec<f32>>`；`extract_audio_features(model: &FeatherHubert, samples_16k: &[f32]) -> Result<AudioFeatures>`；`AudioFeatures { identity: ModelIdentity, video_frames: usize, values: Vec<f32> }`，布局 `[N,2,1024]`。

- [x] **Step 1:** 写 `hubert_parity`：当前 checkpoint 配置被正确采用；静音、短于 400 samples、奇数特征长度、长音频分块/上下文边界分别与参考相符；输出原始 token 数满足 `(samples - 80) / 320`，视频特征按原逻辑截为偶数 token 后分组；静音不产生 NaN，过短输入返回有动作建议的错误。关键断言：`assert_eq!(features.values.len(), features.video_frames * 2 * 1024); assert!(features.values.iter().all(|v| v.is_finite()));`。
- [x] **Step 2:** `cargo test -p feathertalk-engine --features flex --test hubert_parity`，预期当前模型缺失 FAIL。
- [x] **Step 3:** 实现配置驱动的前端卷积、GroupNorm/GELU/TCN/投影和映射；逐层载入校验权重，禁止静默随机初始化缺失参数。归一化、长音频处理与截断复刻当前参考，全部输出携带实际模型身份。
- [x] **Step 4:** 重跑 CPU 测试，并在已有 GPU 路径上执行相同测试，预期误差在固定容差内；大音频测试验证内存上界而非把所有中间激活永久缓存。
- [x] **Step 5:** 提交 `feat: port current FeatherHuBERT to Burn`。

## Task 5: SCRFD、GhostOne 关键点与均值脸后处理

**Files:** Create `tools/modelgen/{Cargo.toml,src/main.rs}`、`crates/feathertalk-engine/src/models/{scrfd.rs,pfld.rs,ghost_one.rs,generated/scrfd.rs}`、`src/data/landmarks.rs`（engine 内）、`crates/feathertalk-engine/tests/face_models_parity.rs`；Modify `Cargo.toml` 将 `tools/modelgen` 加为 workspace 成员。

**Interfaces:** `Scrfd::load(assets: &Path, device: &Device) -> Result<Self>`、`detect(&self, frame: &BgrFrame) -> Result<Vec<Detection>>`；`PfldGhostOne::load(assets: &Path, device: &Device) -> Result<Self>`、`forward(&self, input: Tensor<4>) -> Tensor<2>`；`locate_face(detector: &Scrfd, landmarks: &PfldGhostOne, mean_face: &[f32;220], frame: &BgrFrame) -> Result<Landmarks>`。`BgrFrame { width, height, stride, data }`、`Rect { x,y,width,height }`、`Detection { bbox,score,keypoints }`、`Landmarks { points: [[i32;2];110] }` 定义于 engine 的 data 模块。

- [x] **Step 1:** 写 `scrfd_levels_and_nms_match`、`pfld_raw_outputs_match`、`mean_face_and_border_coordinates_match`、`no_face_is_actionable`；验证 SCRFD 三个 stride 输出与 score/bbox/kps 后处理、0.1 检测阈值/0.5 NMS、原版第一个有效主体规则、1.05 扩框/黑色补边、PFLD 192 输入、均值脸相加、坐标缩放及向零截断。关键断言：`assert_eq!(actual.points, reference.points); assert_eq!(no_face_error.code, ErrorCode::NoFace);`，整数坐标 fixture 覆盖负值截断与靠边补黑。
- [x] **Step 2:** `cargo test -p feathertalk-engine --features flex --test face_models_parity`，预期模型缺失或后处理误差 FAIL。
- [x] **Step 3:** 用匹配版本 burn-onnx 从原 ONNX 生成 SCRFD 代码，显式从已验证资源路径加载权重，避免 `Model::default()` 隐式依赖构建机绝对路径，依据 [burn-onnx 官方加载说明](https://docs.rs/burn-onnx/latest/burn_onnx/#loading-weights)。生成工具包名 `feathertalk-modelgen`，接受 `--input <onnx> --source-output <rs> --assets-output <dir>`，同步更新资产生成器版本和新权重哈希，保留原 ONNX 来源哈希。PFLD 先复刻 `inference_mode=False`、6 分支的 eval 结构及 checkpoint 映射；如重参数化，另生成并验证等价产物。移除未被 forward 使用的结构前先检查权重覆盖报告，不静默忽略执行必需参数。
- [x] **Step 4:** 重跑 CPU/GPU 对齐测试，预期 raw tensor 和最终 110 点均通过；修改均值脸任意值或哈希的测试副本必须拒绝加载。不存在人脸时保留帧号/阶段，不能下标越界崩溃。
- [x] **Step 5:** 提交 `feat: port face detection and landmark preprocessing`。

## Task 6: 默认 FeatherTalk 人物网络

**Files:** Create `crates/feathertalk-engine/src/models/{feathertalk.rs,audio_adapter.rs,attention.rs,visual_blocks.rs}`、`crates/feathertalk-engine/tests/visual_parity.rs`。

**Interfaces:** `FeatherTalk::new(device: &Device) -> Self`；`forward(&self, image: Tensor<4>, audio: Tensor<3>) -> Tensor<4>`；`encode_image(&self, image: Tensor<4>) -> ImagePyramid`；`decode_features(&self, features: ImagePyramid, audio: Tensor<3>) -> Tensor<4>`；`ImagePyramid` 含 x1..x5 五个 rank-4 特征。测试用固定参数加载只用于开发验证，不向产品暴露旧人物导入。

- [x] **Step 1:** 写 `visual_parity` 的前向/关键层/代表性参数梯度比较；输入 `[B,6,144,144]` 与 `[B,40,1024]`，输出 `[B,3,144,144]`；覆盖奇数中间尺寸、相对位置偏置、residual gate、softmax 轴和上采样 padding；`forward` 与 encode/decode 分开执行应等价。关键断言：`assert_eq!(output.dims(), [1, 3, 144, 144]); assert!(parity.passed);`。
- [x] **Step 2:** `cargo test -p feathertalk-engine --features flex --test visual_parity`，预期网络缺失 FAIL。
- [x] **Step 3:** 逐模块迁移 inverted residual、AudioAdapter、局部上下文、三层空间音频注意力与 UNet；默认视觉通道 `(24,32,64,128,256)`、audio_dim=512、center index=10，初始化/归一化/训练与 eval 语义与参考一致。注意可训练位置参数和 gate 不得在转换时遗漏。
- [x] **Step 4:** 重跑 CPU 与已具备 GPU 的同一测试，预期固定参数前向和梯度通过；验证随机新模型参数可参与优化且保存/载入后前向不变。
- [x] **Step 5:** 提交 `feat: port FeatherTalk visual network to Burn`。

## Task 7: VGG 感知分支与全部训练损失

**Files:** Create `crates/feathertalk-engine/src/models/vgg.rs`、`src/training/{mod.rs,losses.rs}`（engine 内）、`crates/feathertalk-engine/tests/loss_parity.rs`。

**Interfaces:** `VggFeatures::load(assets: &Path, device: &Device) -> Result<Self>`、`forward(&self, images: Tensor<4>) -> Tensor<4>`；`compute_loss(prediction: Tensor<5>, target: Tensor<5>, mouth_mask: Tensor<5>, vgg: &VggFeatures, config: &LossConfig) -> LossOutput`；pair 轴长度 2，`LossOutput { total: Tensor<1>, parts: LossParts }`，parts 六个命名分项为 `Tensor<1>`，名称与 core 的 LossScalars 相同。仅指标采样时调用 `LossParts::to_scalars() -> LossScalars`，避免逐步同步所有标量。

- [x] **Step 1:** 写 `all_loss_terms_match_reference`、`vgg_frozen_but_input_gradient_flows`、`zero_mask_is_finite`、`temporal_weights_zero_disable_temporal_path`；固定系数逐项比较值与梯度，断言 VGG 参数不更新而生成输入梯度非零；梯度差分/掩码交集、时序掩码并集和 avg_pool 边界与原版相同。关键断言：`assert!(prediction_gradient.iter().any(|g| g.abs() > 0.0)); assert_eq!(vgg_weights_before, vgg_weights_after);`。
- [x] **Step 2:** `cargo test -p feathertalk-engine --features flex --test loss_parity`，预期未实现损失或梯度丢失 FAIL。
- [x] **Step 3:** 实现 VGG features[0..14]、全图/嘴部 L1、低通时序/嘴部时序、VGG MSE、嘴部横纵梯度项；保留 BGR/0..1/无额外 ImageNet 归一化约定。仅 target 分支停止梯度；感知分支冻结权重不能阻断 prediction 梯度。
- [x] **Step 4:** 重跑 CPU/GPU 验证，预期所有分项与代表性梯度通过固定容差；一次完整人物更新后仅人物模型变化。
- [x] **Step 5:** 提交 `feat: reproduce FeatherTalk training losses and gradients`。

## Task 8: 单视频规范化、几何和可恢复数据集

**Files:** Create `crates/feathertalk-engine/src/media/{mod.rs,ffmpeg.rs,probe.rs}`、`src/data/{mod.rs,geometry.rs,prepared.rs,sampler.rs}`（engine 内）、`crates/feathertalk-engine/tests/{data_parity.rs,media_prepare.rs}`。

**Interfaces:** `MediaTools { ffmpeg: PathBuf, ffprobe: PathBuf }`；`prepare_project(project: &ProjectManifest, assets: &Path, tools: &MediaTools, device: &ExecutionDevice, observer: &dyn TaskObserver, cancel: &Cancellation) -> Result<PreparedProject>`，使用 Task 1 的观察器与取消令牌。`PreparedProject` 含 canonical video/audio、帧/关键点/特征索引及预处理身份。`PairSampler::new(prepared: &PreparedProject, seed: u64) -> Result<Self>`、`PairSampler::restore(prepared: &PreparedProject, state: SamplerState) -> Result<Self>`、`snapshot(&self) -> SamplerState`、`next_batch(&mut self, batch_size: usize) -> Result<Option<TrainingBatch>>`；SamplerState 保存 epoch、下一个 pair、置乱身份、参考帧随机计数/状态。`TrainingBatch` 为 image `[B,2,6,144,144]`、audio `[B,2,40,1024]`、target `[B,2,3,144,144]`、mask `[B,2,1,144,144]` 的可上传 CPU 数据，使用 `HostTensor<const N: usize> { dims: [usize;N], values: Vec<f32> }` 表达。

- [x] **Step 1:** 准备可执行的隔离 FFmpeg/FFprobe 组件供测试使用，并记录版本与来源。写 `geometry_matches_opencv`、`audio_edges_and_shared_reference_match`、`cache_invalidates_when_model_changes`、`sampler_replays_after_restore`；验证 152→144/边框4、MASK_RECT 缩放/取整、面积缩放、参考图顺序、嘴部 ROI、首尾补零、相邻帧共享参考及不配对最后一帧。`media_prepare` 使用 FFmpeg 合成 30 FPS/立体声与无音轨视频，验证副本 25 FPS、选择原版第一声道并准备 16 kHz、原文件哈希不变、无音轨有明确提示。关键断言：`assert_eq!(batch.image.dims, [1, 2, 6, 144, 144]); assert_eq!(batch.audio.dims, [1, 2, 40, 1024]);`。
- [x] **Step 2:** `cargo test -p feathertalk-engine --features flex --test data_parity --test media_prepare`，预期数据管线缺失 FAIL；FFmpeg 路径由 `FT_FFMPEG`/`FT_FFPROBE` 提供，缺失需明确报错。
- [x] **Step 3:** 实现规范化与有界抽帧/预取、关键点/特征缓存和身份校验；几何纯函数与参考对齐。Sampler 将 shuffle、参考帧选择与采样游标序列化为 `SamplerState`，随机性不依赖预取线程调度；检查取消并输出阶段/帧号进度。素材检测不到人脸时精确失败，不静默删除视频帧。
- [x] **Step 4:** 重跑上述测试，预期 PASS；变更音频模型哈希后必须重新提取特征；预取启用/禁用时，相同 sampler state 的下一批完全相同。采用真实验收素材后再跑全预处理并记录源/规范化文件与输出身份。
- [x] **Step 5:** 提交 `feat: prepare single-video datasets with reproducible sampling`。

## Task 9: 训练循环、批量探测和可恢复检查点

**Files:** Create `crates/feathertalk-engine/src/training/{trainer.rs,checkpoint.rs,batch_probe.rs,metrics.rs}`、`crates/feathertalk-engine/tests/{training_resume.rs,training_convergence.rs}`。

**Interfaces:** `train(request: &TrainRequest, prepared: &PreparedProject, assets: &Path, device: &ExecutionDevice, observer: &dyn TaskObserver, cancel: &Cancellation) -> Result<TrainingResult>`；`save_checkpoint(state: &TrainingState, destination: &Path) -> Result<ModelRef>`；`load_checkpoint(model_ref: &ModelRef, device: &ExecutionDevice) -> Result<TrainingState>`；`probe_batch_size(state: &mut TrainingState, prepared: &PreparedProject, vgg: &VggFeatures, device: &ExecutionDevice, candidates: &[usize]) -> Result<usize>`。`TrainingState` 包含人物参数/BN、Adam、配置、run_id、epoch/global_step、sampler_state、随机状态及所有数据/模型指纹；`TrainingResult` 含最新 ModelRef 与停止原因。从头训练分配新的 TrainingRunId，续训保留原 run_id，用于曲线与历史归属。

- [x] **Step 1:** 写 `continuous_and_resumed_steps_match`、`cancel_commits_after_complete_step`、`probe_restores_all_mutable_state`、`interrupted_checkpoint_keeps_previous`、`manual_batch_is_never_overridden`；使用真实小模型/固定批验证参数、优化器、BN、采样/随机状态恢复；在写入阶段注入失败验证旧 checkpoint 仍可加载。`training_convergence` 固定几对样本反复训练，断言 loss 有限、参数确实更新且最终 loss 低于起点。关键断言：`assert_eq!(restored.global_step, before.global_step); assert_eq!(restored.sampler_state, before.sampler_state); assert!(final_loss < initial_loss);`。
- [x] **Step 2:** `cargo test -p feathertalk-engine --features flex --test training_resume --test training_convergence`，预期训练/恢复能力缺失 FAIL。
- [x] **Step 3:** 实现 Adam 和原始成对训练路径、完整 dry-run 批量探测、状态回滚、定期/取消保存、总 epoch 续训和跨设备记录。检查点写入独立 staging 目录后原子提交，已提交引用最后更新；曲线历史追加采样记录，UI 指标最多 2Hz，不在每一步同步读回所有分项。恢复前校验素材、预处理及模型身份，明确拒绝不匹配的续训。
- [x] **Step 4:** 重跑测试，预期 CPU 全部 PASS；已有 GPU 各跑完整训练步与短程过拟合；同一检查点在 CPU↔GPU 加载后前向通过容差、可继续更新。OOM 在一键探测中可尝试更小批量，在手动模式提供建议。实际训练步中取消保存与意外终止后恢复分别验证。
- [x] **Step 5:** 提交 `feat: train and resume FeatherTalk with atomic checkpoints`。

执行记录：Task9 按训练阶段补充计划完成，并通过逐阶段独立审查。初始行为测试的完整 RED 顺序未全部在实现前执行；真实回归具有单独 RED/GREEN 记录。勾选项表示行为实现与可用验证完成，不宣称补齐历史测试顺序。CPU/RTX2060 的实际训练与恢复已验证，其他硬件、平台和真人素材效果仍为后续验收事项。

## Task 10: 离线生成、回贴和原子导出

**Files:** Create `crates/feathertalk-engine/src/inference/{mod.rs,renderer.rs,export.rs,session.rs}`、`crates/feathertalk-engine/tests/{render_parity.rs,video_export.rs}`。

**Interfaces:** `generate(request: &GenerateRequest, prepared: &PreparedProject, assets: &Path, tools: &MediaTools, device: &ExecutionDevice, observer: &dyn TaskObserver, cancel: &Cancellation) -> Result<ExportResult>`；`FramePicker::next() -> usize`；`FrameRenderer::render(frame_index: u64, features: &AudioFeatures) -> Result<BgrFrame>`。`ExportResult { output_path, video_frames, elapsed_ms, device }`。`AudioSource` 提供 session/reset/read/finish，`FrameSink` 提供 write/finish/abort，使用带整数时间基的 `TimestampedAudio`/`TimestampedFrame`；首版实现文件来源与文件输出，不伪装为已有流式模型。

- [x] **Step 1:** `render_parity` 验证原版往返资源帧序列、模型输入/窗口、回贴中心和边框不变；`video_export` 验证 25 FPS、H.264/AAC、输入画幅、按特征得到的帧数、中文/空格路径、已有目标文件和编码失败。取消/失败不得覆盖已有成功成品，子进程须退出且新临时成品不被报告成功。关键断言：`assert_eq!(probe["streams"][0]["codec_name"], "h264"); assert_eq!(export.video_frames, features.video_frames as u64); assert_eq!(old_output_bytes, output_bytes_after_cancel);`。
- [x] **Step 2:** `cargo test -p feathertalk-engine --features flex --test render_parity --test video_export`，预期尚无导出实现 FAIL。
- [x] **Step 3:** 实现音频输入规范化、一次提取特征、按序视觉推理与回贴、FFmpeg stdin/文件接入及最终封装；进程参数以 OsString/args 传递，不拼 shell 字符串。输出先写同目标目录临时文件，成功关闭并核验后替换目标；日志保留 FFmpeg 的真实失败信息。会话接口使用有界批量，模型层与前端无耦合。
- [x] **Step 4:** 重跑验证，预期 PASS；CPU 与已具备 GPU 使用相同检查点生成，逐帧回贴结果在固定容差内且可播放；断网运行完整生成；记录实际设备与阶段耗时。
- [x] **Step 5:** 提交 `feat: generate talking-head videos with recoverable export`。

执行记录：Task10 在 CPU/RTX2060 CUDA 完成真实预处理、训练检查点和生成验证，并通过独立审查。原 Python 目录可能发生首次导入字节码缓存写入，已记录偏差并禁止后续写入；未声称恢复未知历史缓存。当前记录总生成耗时，分阶段耗时留待性能任务；整机强制断网、WebView 奇数画幅播放和真人素材效果仍留作对应后续验收。

## Task 11: 可校验的资源缓存与离线就绪检查

**Files:** Create `src-tauri/src/resources/{mod.rs,catalog.rs,download.rs,inventory.rs}`、`src-tauri/tests/resource_cache.rs`、`resources/catalog.schema.json`、`resources/development-catalog.json`。

**Interfaces:** `ResourceManager::new(root: PathBuf, catalog: Catalog) -> Result<Self>`；`inventory(&self, target: &TargetPlatform) -> Result<ResourceInventory>`；`prepare(&self, selection: &ResourceSelection, observer: &dyn TaskObserver, cancel: &Cancellation) -> Result<ResourceInventory>`；`import_local(&self, manifest: &Path) -> Result<()>`。Catalog 使用 Task 1 AssetManifest，并增加平台/架构/组件 kind；`ResourceSelection` 指定目标与所需版本；`ResourceInventory { ready: bool, items: Vec<ResourceStatus> }` 列出每项身份/路径/就绪或失败原因，类型放入 core assets 模块供 UI 共用。

- [x] **Step 1:** 用 loopback HTTP fixture 写 `partial_and_bad_hash_never_become_ready`、`offline_uses_complete_versions`、`upgrade_preserves_project_dependency`、`executable_matches_target`；截断/错误哈希/不支持的架构不能安装，失败后旧版仍可用；准备好的资源断网验证不发 HTTP 请求，缺资源给出所缺项。关键断言：`assert!(!corrupt_inventory.ready); assert!(complete_inventory.ready); assert_eq!(offline_request_count, 0);`。
- [x] **Step 2:** `cargo test -p feather-talk-gui --test resource_cache`，预期资源服务缺失 FAIL。
- [x] **Step 3:** 实现版本目录、文件大小/SHA-256 校验、暂存提交、进度与重试；从 Task 2 输出导入开发资源；下载只发生在显式准备/更新阶段。使用真实可配置 catalog，不编造生产 URL，不把现有标记未批准分发的模型自动公开上传。组件安装维护平台可执行权限，并保留 notices/来源信息。
- [x] **Step 4:** 重跑资源测试，预期 PASS；从开发 catalog 导入公共模型/均值脸/worker/媒体组件后拔网，inventory 全就绪；替换一个文件字节必须显示损坏并可安全重新获取。
- [x] **Step 5:** 提交 `feat: manage verified model and component resources offline`。

执行记录：Task11 资源管理及当前模型/媒体/CPU 探测/CUDA 数学库导入已验证，并通过修复后独立审查。worker 与其他后端/平台组件留待后续实际构建，因此完整应用仍明确为未就绪；当前验证本地资源复用与零 HTTP，整机断网和生产资源地址/分发依据仍待对应验收。

## Task 12: 独立 worker、单任务运行与故障归属

**Files:** Create `crates/feathertalk-worker/{Cargo.toml,src/main.rs,src/runtime.rs,src/transport.rs,tests/worker_protocol.rs}`、`src-tauri/src/worker_host.rs`、`src-tauri/tests/worker_host.rs`。

**Interfaces:** `WorkerRuntime::handle(request: WorkerRequest) -> Result<()>`；`WorkerHost::spawn(component: &Path, context: WorkerContext) -> Result<Self>`；`start(&mut self, task_id: TaskId, job: JobSpec) -> Result<()>`、`cancel(&mut self, task_id: TaskId) -> Result<()>`、`next_event(&mut self, timeout: Duration) -> Result<HostPoll>`，区分事件、超时和会话结束。`ResourceManager::bind_worker(&ResourceSelection, ProjectBinding, &JobSpec) -> Result<InstalledWorker>` 绑定已校验的组件与项目，再通过 `InstalledWorker::spawn` 启动。`WorkerContext` 提供项目/资产/媒体工具路径和协议版本；实际计算调用 Task 8/9/10 接口。

- [x] **Step 1:** 编写进程协议和宿主测试，覆盖握手、重复开始、错发/迟到事件、偏长或分片消息、控制流拥塞时的取消、stderr 分离与异常退出。关键断言是重复开始返回 `Busy`，一个任务最多只有一个终态。真实训练边界由安装包测试验证。
- [x] **Step 2:** 用有界消息、任务事件归属和取消等回归测试复现缺失或错误行为，针对修复保存前后的结果；见 Task 12 阶段记录。
- [x] **Step 3:** 实现一行一个 JSON、协议版本 1、单条消息上限 1 MiB、独立控制读取/计算与事件发射。事件先按已接收任务 ID 归属，seq 在每个 worker 会话严格递增，终态只提交一次。Windows 隐藏控制台；FFmpeg/worker 子进程生命周期受控，正常取消优先安全保存，异常退出恢复到最后已提交 checkpoint。默认 CPU worker 可独立运行，GPU worker 仅启用对应编译 feature。
- [x] **Step 4:** 进程和宿主测试通过。用 Windows CPU 与 RTX 2060 CUDA 的已校验安装包完成真实训练、取消保存和重启导出；在 RTX 2060 上续训并于发布 step 2 后强制结束仍在运行的训练，宿主报告该检查点，新进程读取并导出通过 FFprobe 验证的 MP4。
- [x] **Step 5:** 归档分阶段提交与独立审查结论。进程和宿主修复提交 `b1ab4df`，安装包绑定提交 `4b374b2`，资源就绪与训练边界验证提交 `0a467ff`。两个阶段均通过独立复核。

## Task 13: Tauri 项目服务与类型化命令桥接

**Files:** Create `src-tauri/src/{state.rs,projects.rs,commands/mod.rs,commands/projects.rs,commands/jobs.rs,commands/resources.rs,commands/devices.rs}`、`src-tauri/tests/app_service.rs`、`src/lib/{contracts.ts,api.ts,events.ts}`；Modify `src-tauri/src/lib.rs`、`src-tauri/Cargo.toml`、`src-tauri/capabilities/default.json`、`src-tauri/tauri.conf.json`、`crates/feathertalk-core/src/protocol.rs`（任务快照与指标页）。

**Interfaces:** 下列 Commands 的结果均包裹 core Result：`list_projects() -> Vec<ProjectSummary>`、`create_project(NewProject) -> ProjectSummary`、`get_project(ProjectId) -> ProjectManifest`、`list_devices() -> Vec<DeviceDescriptor>`、`get_resources(TargetPlatform) -> ResourceInventory`、`prepare_resources(ResourceSelection) -> TaskId`、`start_training(TrainRequest) -> TaskId`、`start_generation(GenerateRequest) -> TaskId`、`cancel_task(TaskId) -> ()`、`get_task() -> Option<TaskSnapshot>`、`read_metrics(MetricsPageRequest) -> MetricsPage`。命令委托 AppState，`AppState::snapshot(&self) -> Result<Option<TaskSnapshot>>` 对应 get_task。`list_devices` 使用 worker Ready 中的 core DeviceDescriptor，不在 Tauri 主进程引入 Burn 计算。前端 `api` 导出 camelCase 包装；事件名固定 `feathertalk://task`，仅转发有任务 ID 的任务事件。快照保留任务、项目、训练 run、状态、最近 seq、进度、指标、已提交检查点与错误；资源任务没有项目，生成没有训练 run。指标页沿用 core 的 `offset`、`limit`、`next_offset`，按该 run 的持久化记录分页。

- [x] **Step 1:** `app_service` 与任务状态测试覆盖单任务占用、重连、切换项目与迟到事件、关闭等待、Unicode 路径迁移和损坏提示；Rust 与 TypeScript 校验同一 [JSON 样例](../../../contracts/desktop-fixture.json)。
- [x] **Step 2:** `cargo test -p feather-talk-gui --test app_service` 已复现服务接口缺失（退出码 101，`.artifacts/task13-logs/service-red.log`）。
- [x] **Step 3:** 注册命令、项目索引、资源管理与后台 worker；接入原生文件选择、成品目录打开与已完成 MP4 的精确访问范围。关闭活动窗口会请求取消并等待终态和进程清理。设备发现可在没有项目时运行，发现会话拒绝训练和生成。
- [x] **Step 4:** Rust 服务/契约/资源/worker/历史测试、`pnpm test:contracts`、`pnpm build` 和 `cargo check -p feather-talk-gui` 通过。Windows WebView 从已校验的 RTX 2060 包完成项目打开、生成、受控播放及续训；一次续训从 step 2 到 step 3 并收到 loss 与提交事件。
- [x] **Step 5:** 提交 `feat: expose project and task services to the desktop UI`。

执行记录：[桌面桥接实测](../../validation/desktop-bridge.md)使用本地开发资源和短素材。人物、资源、训练和生成页面分别由 Task 14–16 完成。

## Task 14: React 应用、人物项目与资源/设备页面

**Files:** Create `src/features/projects/{ProjectPage.tsx,project-state.ts}`、`src/features/settings/{ResourcesPage.tsx,DevicePicker.tsx}`、`src/components/{AppShell.tsx,TaskStatus.tsx,ErrorDetails.tsx}`、`src/components/ErrorDetails.test.tsx`、`src/features/projects/project-state.test.ts`、`src/test/setup.ts`、`vitest.config.ts`；Modify `src/App.tsx`、`src/App.css`、`package.json`、`pnpm-lock.yaml`。

**Interfaces:** `useTaskSnapshot(): { snapshot: TaskSnapshot | null, error: AppError | null }` 定义在 `src/lib/events.ts`，订阅、初始 snapshot 和 cleanup 归该 hook；先订阅再读取快照，按 ID/seq 合并以覆盖初始化竞态。`DevicePicker` 消费 `DeviceDescriptor[]` 与 `DevicePolicy`；`ErrorDetails({ error: AppError })` 展示实际阶段/原因/建议/日志入口；`AppShell` 导航人物/训练/生成/资源设备四个功能。

**界面约定：** 用户要求统一使用 shadcn/ui。项目采用 `radix-nova`、Tailwind CSS 4 和本地字体，组件由官方 registry 添加；训练、loss 图表、生成及后续设置均沿用同一组件库与语义主题。

- [x] **Step 1:** 行为测试覆盖单视频替换、忙碌时限制重复启动、已验证资源不会触发下载、项目切换与移动，以及 OOM、无脸、编码失败和未知异常的原始证据。
- [x] **Step 2:** Vitest + Testing Library 新增测试先验证缺失行为（`.artifacts/task14-logs/red.log`），实现后 `pnpm test --run` 通过。
- [x] **Step 3:** 实现人物项目、资源清单、实际设备与共用任务状态、错误诊断。shadcn/ui 组件统一按钮、表单、状态、折叠诊断和空状态；没有实时和旧模型导入入口。
- [x] **Step 4:** 行为测试、桥接契约、前端及原生构建通过。Windows WebView2 验证初次下载缺失组件、服务器停止后校验已就绪、真实 RTX 2060、空项目和键盘跳转、资源错误详情、原生中文长路径选取与项目重开，以及 320–1120 CSS px 下的操作可达性。详见 [桌面界面验收](../../validation/desktop-ui.md)。
- [x] **Step 5:** 提交 `feat: add project and resource management screens`。

## Task 15: 一键/高级训练与实时 loss 曲线

**Files:** Create `src/features/training/{TrainingPage.tsx,TrainingForm.tsx,LossChart.tsx,metrics.ts,training.test.tsx,metrics.test.ts}`。

**Interfaces:** `validateTrainingForm(value: TrainingFormValue): TrainConfig | FormErrors`；`appendMetric(history: MetricPoint[], event: WorkerEvent, owner: { taskId: TaskId; projectId: ProjectId; runId: TrainingRunId }): MetricPoint[]`；`selectChartPoints(history: MetricPoint[], maxPoints: number): MetricPoint[]`。`MetricPoint` 直接对应 Task 1 Metrics；页面实时点按 task/project/run 归属过滤，历史来自对应 run_id 的 `read_metrics`，续训接原曲线，新训练开启新的历史。

- [x] **Step 1:** 新增一键默认、高级参数与安全整数校验、续训失败、取消保存、历史分页和串线防护测试；Rust 实测回归还覆盖旧检查点重复步、500 ms 限频和重开历史。
- [x] **Step 2:** 训练流程与指标的缺失行为先复现失败；修复后 17 项前端测试通过，RTX 2060 的训练恢复与历史测试通过。
- [x] **Step 3:** 接入从头训练、预处理、检查点恢复，使用 shadcn Chart 展示总 loss、分项、实际批量、设备、精度、学习率和耗时。只读历史页与实时事件按任务及 run 归属合并，保留全部磁盘记录，显示最多 2000 点。
- [x] **Step 4:** 前端测试与构建通过。Windows WebView2 + RTX 2060 完成一键从头训练、旧检查点续训、实时分项曲线、取消保存及重启读回。运行中切换页面约 35 ms；指标最多每 500 ms 采样一次。同输入、多次运行的吞吐对比纳入 Task 17，详见 [训练验收](../../validation/desktop-ui.md#training-and-loss-on-windows)。
- [x] **Step 5:** 提交 `feat: add one-click training and live loss monitoring`。

## Task 16: 音频导入、视频生成与失败恢复页面

**Files:** Create `src/features/generation/{GenerationPage.tsx,generation.test.tsx}`；Modify 共用 `TaskStatus.tsx`/`ErrorDetails.tsx`（仅有实际需求时）。

**Interfaces:** 页面调用 `api.startGeneration(GenerateRequest): Promise<TaskId>`；成功仅采纳当前任务 Completed 中的 `ExportResult`；通过已有本地文件访问能力预览成品并打开目录。

- [x] **Step 1:** `generation.test.tsx` 覆盖未训练及忙碌禁用、Unicode 路径、取消后拒绝旧成品、编码和磁盘错误建议以及重试时重选输出。Rust 实际导出测试记录阶段顺序和取消保留原成品。
- [x] **Step 2:** 首次前端测试确认页面缺失；实际 CPU 导出观察到原引擎缺少音频读取及封装进度（`.artifacts/task15/generation-progress-red.log`）。
- [x] **Step 3:** 用 shadcn/ui 接入检查点选择、原生音频和保存对话框、分阶段进度、取消、精确授权的成品预览和打开目录。失败保持所选音频；再次生成要求重新确认保存位置。损坏音频显示解码阶段和原始错误。
- [x] **Step 4:** 前端测试、构建与 RTX 2060 Rust 生成回归通过。Windows WebView2 使用 Task 15 刚训练的检查点导出并播放 MP4；损坏 WAV 失败后未产生输出、更换音频和保存位置后再次成功。详见 [导出验收](../../validation/desktop-ui.md#audio-to-mp4-on-windows)。
- [x] **Step 5:** 提交 `feat: add offline audio-to-video generation workflow`。

## Task 17: 全链路基准与按设备启用优化

**Files:** Create `crates/feathertalk-engine/src/benchmark/{mod.rs,report.rs,policy.rs}`、`src/bin/benchmark.rs`（engine 内）、`crates/feathertalk-engine/tests/performance_policy.rs`、`docs/validation/performance.md`；Modify 需要优化的 engine 模块和设备设置页面。

**Interfaces:** `run_benchmark(spec: &BenchmarkSpec, observer: &dyn TaskObserver) -> Result<BenchmarkReport>`；`select_backend(policy: &DevicePolicy, reports: &[BenchmarkReport]) -> Result<DeviceRequest>`。BenchmarkSpec 固定项目/素材/资源/工具/检查点路径、候选设备、精度、batch、warmup、repetitions 和优化开关，示例随工具说明保存。使用 Task 1 的 DevicePolicy；worker 有有效报告时调用 select_backend，无报告时仍调用 Task 3 的 resolve_device 并做初始能力验证，不改该函数签名。报告绑定模型/引擎/驱动/设备/精度/batch/输入身份，记录 validation_passed、cold_runs/warm_runs 原始样本、预处理、训练吞吐、特征提取、视觉与端到端导出、内存/显存可用测量。不能取得某指标时写 null 与原因。

- [x] **Step 1:** `performance_policy` 验证最快但数值验证失败的后端不被选择、驱动或模型变化时报告不可复用、手动选择保留，以及同 batch 热测和冷启动分别处理。未知驱动的报告同样不会被当作有效缓存。
- [x] **Step 2:** 首次 `cargo test -p feathertalk-engine --features flex --test performance_policy` 因模块缺失而失败。实现后 3 项策略测试通过，日志在 `.artifacts/task17/`。
- [ ] **Step 3:** 实现基准 CLI、候选筛选与缓存；基准使用隔离的状态/输出，不改变用户模型与训练进度。在 RTX 2060 比较 CUDA/Vulkan、B580 测 Vulkan、CPU 比较合适后端。分别评估批量推理、有限静态视觉缓存、预取、融合/内核缓存、适用混合精度；每个优化先跑对应数值/梯度/恢复验证，再比较相同输入下多次端到端结果。通过热身后至少保留三次原始测量；只有效果门槛通过且可重复获益的策略默认启用。
- [ ] **Step 4:** 重跑策略测试；运行 `cargo run --release -p feathertalk-engine --features vulkan --bin benchmark -- --spec .artifacts/benchmark-spec.json --output .artifacts/reports/benchmark-vulkan.json`，CUDA/CPU 对应替换 feature/spec。预期报告写实测设备与数据，不把缺失硬件标为支持已验证；优化开启/关闭均能正常训练、续训和生成。
- [ ] **Step 5:** 提交 `perf: select validated backends and optimize measured workloads`，附可重放基准说明与小型结果摘要，不提交私人人物素材。

执行记录：同一 release 程序在 Windows 10 的 RTX 2060（560.94）与 i5-10400 上各跑了 1 次冷测、1 次热身、3 次热测。CUDA 热训练步更快，Vulkan 在 29 帧短导出中更快。去掉了 worker 续训和生成的重复 Burn 恢复，已在 CPU/Vulkan release 验证。Vulkan 预取对照做了四轮独立进程测量，采样 loss 相同，记录进程工作集并保留原默认值。完整样本和测量边界见 [性能记录](../../validation/performance.md)。生产自动选择暂未接入报告缓存，其他设备和其余优化仍待验证。

长素材跟进：[CUDA 训练提速](2026-11-19-cuda-training-speed.md) 在 60 秒、1,500 帧素材上定位到每步重复扫描整套预处理缓存。RTX 2060、固定 batch size 1 的热步中位数由 22.450 秒降到 1.513 秒。随后将源视频和预处理缓存的完整校验集中到每次任务启动，从头训练和续训均复用已校验的数据；训练步直接在设备上更新。三次生产 worker 测得热步中位数 0.727–0.836 秒，取消保存 0.931–1.097 秒；保留路径、长度和数值检查，以及原子保存。模型、Adam 和续训对照详见 [性能记录](../../validation/performance.md#每次训练启动时校验一次)。

## Task 18: 多平台组件打包、干净环境与完整验收

**Files:** Create `.github/workflows/{desktop-checks.yml,component-builds.yml}`、`tools/acceptance/{run.ps1,run.sh,check_reports.py,test_reports.py}`、`tools/package/{stage-components.mjs,build-catalog.mjs}`、`resources/targets.json`、`docs/validation/{platform-matrix.md,release-checklist.md}`；Modify `src-tauri/tauri.conf.json`、`src-tauri/capabilities/default.json`、`README.md`。

**Interfaces:** `stage-components.mjs --target <triple> --backend <backend> --input <dir> --output <dir>`；`build-catalog.mjs --artifacts <dir> --output <json>`；`check_reports.py --matrix resources/targets.json --reports <dir> --require-complete`；内部 `validate_reports(matrix: dict, reports: list[dict], require_complete: bool) -> ValidationResult`，结果含 passed/missing/failures。每份报告含 OS/架构/显卡/驱动/实际后端、组件和模型身份、训练/恢复/生成/离线结果与日志路径。非完整模式允许报告环境尚未取得，完整发布检查必须拒绝缺失的必需项；platform-matrix.md 展示 JSON 目标与实际报告结果。

- [x] **Step 1:** 报告检查器针对仅编译、Intel Mac 错标 GPU、CPU 冒充 AMD、缺离线验收、错误 worker target、缺均值脸及无效资源地址均有测试。已记录缺失模块时的失败与实现后的通过结果；`--require-complete` 会拒绝缺失的发布验收。
- [x] **Step 2:** 已实现 target 清单、组件暂存与 catalog 构建，配置 Windows NSIS、Linux deb/AppImage、macOS arm64/x86_64 app/dmg。CPU worker/probe 由 Tauri 资源目录随包导入，FFmpeg、GPU worker 和模型使用同一校验缓存；运行路径来自安装目录和缓存。CI 配置仅构建、测试、暂存，实际平台构建进度见 Step 3。
- [ ] **Step 3:** 运行 `cargo fmt --all --check`、各已启用 feature 的针对性 Rust 集成测试、`pnpm test -- --run`、`pnpm build`、`pnpm tauri build`，预期目标平台构建和所需验证通过；CPU 检查在 CI 可执行，GPU 任务使用能实测对应硬件的环境。macOS 分发同时验证组件签名/公证、可执行权限及下载后实际启动；记录最低系统版本、编解码构建信息/notices。
- [ ] **Step 4:** 在 Win11/RTX 2060、Win11/B580 及 CPU 先跑验收脚本，再补齐 Linux/NVIDIA/AMD/Arc、M 系列 Mac、Intel Mac CPU 的必需路径。用固定人物/音频比较当前 Python 效果；执行干净资源目录首启→下载→断网→一键训练→曲线→取消保存→重开续训→生成播放，另测损坏资源、OOM/驱动失败、中文路径和编码中断。生产 catalog 需要真实可访问地址、有效哈希和可分发来源；没有素材/设备/地址的项保持未通过，不把它们计作任务全部完成。
- [ ] **Step 5:** `python tools/acceptance/check_reports.py --matrix resources/targets.json --reports .artifacts/reports --require-complete` 预期全部必需行 PASS；README 写清首启/离线/硬件与资源要求、基准和已验证范围。提交 `build: package and validate supported desktop hardware paths`；待验证的环境或发布依赖必须在交付记录保留为未完成。

执行记录：Windows 10 x64 上用最终 Tauri overlay 构建新版 NSIS，并验证独立目录安装、空缓存首启、随包 CPU worker/probe 的复制及退出。CPU 和 RTX 2060 Vulkan 在 release worker 上从 step 1 续训到 step 2，导出 640×600、29 帧、25 FPS 的 H.264/AAC。完整发布门禁退出 1，确认的发布项仅为 Windows x64 构建；其余平台构建与完整发布验收见 [平台矩阵](../../validation/platform-matrix.md)。

#### Task 18a: 可搬到 Win11 / Arc B580 的本地验收包

目标是让另一台机器直接执行现有训练、续训和导出验收，无需安装 Rust、Node 或 Python。继续使用目标机显卡驱动和本地已校验资源；测试素材由运行人选择。安装验收和发布验收仍按 Task 18 记录。

- [x] **打包契约：** `tools/acceptance/stage_smoke_kit.py --packages <dir> --runner <exe> --output <new-dir>` 复制并校验 Windows x64 CPU/Vulkan 默认资源、原 catalog、验收程序和 `run.ps1`。相对地址搬移、缺失组件和同大小内容损坏三项测试通过。
- [x] **运行契约：** `run.ps1 -Runner <exe>` 直接运行 `hardware_acceptance`；省略时沿用 Cargo，测试兼容新相对地址和已有绝对地址。搬移后用无开发工具的 PATH 在 RTX 2060 复跑成功，记录位于 `.artifacts/task18/relocated kit/result/`。
- [x] **交付：** 已保留 `.artifacts/task18/arc-b580-local-kit.zip`、SHA-256 和 [复跑命令](../../validation/release-checklist.md)。B580 验收仍待 Win11 实机执行。

## 外部依赖与执行边界

- 生产模型/组件地址和资产分发依据由发布配置提供；实现阶段用本地导入和 loopback 下载测试覆盖同一资源代码，生产离线交付验收仍需要真实 catalog。
- 真实人物训练视频、驱动音频及相同输入的 Python 参考产物在实施早期固定。当前没有已指定素材，数值/协议测试可先推进，实际画质验收随后完成并留存身份。
- 当前主机是 Windows/RTX 2060，用户还可提供 Win11/B580；其他硬件与系统需要补充实机或合适测试环境。不能用 CI 编译结果替代实际训练/推理支持验证。
- 若在算子可行性中发现必须更换 Burn 版本、实现自定义反向或调整数学实现，先用最小对齐用例证实原因。保持已批准的模型与损失语义；若必须改变产品范围或算法，再带证据提出设计变更。

## 自审记录与计划完成定义

已按设计逐项映射：资产/网络/梯度 Task 2–7，数据/训练/检查点 Task 8–9，离线与实时接口边界 Task 10，资源/worker/Tauri Task 11–13，一键/高级/loss/排错 Task 14–16，性能 Task 17，全部平台与离线交付 Task 18。Review Focus 五项均有对应测试。签名、类型与事件名称按各 Task 的 Interfaces block 贯通；具体依赖的小版本、图表实现与 Burn 原生记录的物理布局留给实现者，并在实际锁定后提交。

本计划编写完成不代表模型或硬件已验证。执行时只有对应任务的验证与提交完成才勾选；整个项目完成必须满足已确认设计的全部平台、工作流、效果和离线交付标准。
