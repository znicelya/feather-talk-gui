# FeatherTalk 性能基准

2026-11-19，Windows 10 22H2 x64；Intel Core i5-10400、RTX 2060 6 GiB，NVIDIA 驱动 560.94（`nvidia-smi`，PCI `0000:01:00.0`）。CUDA 与 Vulkan 的 Burn 设备身份都指向这块 RTX 2060。CPU 路径使用 Burn Flex。

以下结果使用同一个 `flex,cuda,vulkan` release 程序，SHA-256 `d10f0b78e4bf7e8dc7f95af1a84c72837167e4b28d2b538dcd9ef037807c2426`。输入是开发用短人物素材和一个已经提交的 Burn 检查点；FP32，每步 1 对帧，导出 29 帧（640×600，25 FPS）。同一设备完成首次执行、一次热身和三次热测。三个后端均通过固定算子输出和梯度校验，采样的总 loss 分别为 CUDA 1.123248、Vulkan 1.123247 和 Flex 1.123178。

| 后端 | 首次预处理 / 完整训练步 / 导出 | 热训练步，三次（ms） | 热导出，三次（ms） | 热视觉阶段中位数（ms） |
| --- | ---: | ---: | ---: | ---: |
| CUDA | 6279 / 24387 / 4627 ms | 1852, 1883, 1862 | 3731, 3747, 3687 | 2494 |
| Vulkan | 4295 / 13086 / 4040 ms | 2099, 2052, 2073 | 3467, 3415, 3475 | 2148 |
| Flex | 3598 / 4605 / 5474 ms | 4630, 4686, 4619 | 5335, 5339, 5474 | 4066 |

在这组短素材上，CUDA 的热训练步耗时比 Vulkan 少约 10%，Vulkan 的热导出耗时比 CUDA 少约 7%。只对这组输入有效。`--compare` 按同一轮的训练步加导出耗时取中位数，这三个报告返回 Vulkan；训练和导出的使用频率不同时应分别看对应列。

首次执行指新基准进程的首轮；未清空操作系统或 GPU 驱动持久缓存。首次预处理会在隔离目录建立新索引，热测包含验证与读取已准备缓存。检查点载入单独计时；热训练步包含输入组装、恢复候选状态、前向、完整损失、反向、Adam 更新和采样 loss，不包含发布磁盘检查点。导出总耗时包含载入、音频与视频解码、HuBERT、渲染、回贴和封装；视觉阶段还包括抽取资源帧与回贴准备。完整分阶段原始样本在 `.artifacts/reports/benchmark-{cuda,vulkan,flex}.json`。本轮报告的进程峰值内存和显存为 `null`；显卡总容量不等于本任务峰值。

worker 现在只读取已提交的检查点 manifest 来定位绑定的预处理与模型依赖；实际续训或生成仍读取并校验完整 Burn 记录。此前定位依赖时会先在设备上完整恢复一次，随后任务再恢复。在本基准里，一次热载入（含准备缓存读取）约 0.74–0.85 秒，表中的生成耗时没有把 worker 这次重复恢复计进去，也不把该耗时直接当作应用提速。检查点测试验证 manifest 损坏仍被拒绝、权重损坏仍无法恢复；CPU 和 Vulkan release worker 均实际通过续训和导出。

## 预取对照与进程内存

在同一个 Vulkan release 程序上只改 `prefetch_capacity`，依次使用 `2 → 0 → 0 → 2` 四个新进程。每个进程保留一次冷测、一次热身和三次热测。结果在 `.artifacts/task17/prefetch-abba/`：

| 预取容量 | 热训练步，六次（ms） | 热训练步中位数 | 热导出中位数 | Benchmark 进程峰值工作集 |
| --- | --- | ---: | ---: | ---: |
| 2 | 2047, 2122, 1960, 2068, 1972, 2113 | 2058 ms | 4092 ms | 643.0–648.2 MiB |
| 0 | 2151, 2135, 2130, 2156, 2095, 1785 | 2133 ms | 4063 ms | 647.3–662.4 MiB |

四次的比较均通过固定算子检查，所有热测总 loss 都为 `1.12324655056`。这组数据的热训练步中位数相差约 3.5%，样本范围重叠；继续保留预取默认值。工作集取 Windows `Process.PeakWorkingSet64`，每 200 ms 读取基准进程，包含其启动、训练、导出及校验阶段，FFmpeg 子进程与显存不计。该测量作为旁路结果放在 `windows-sampling.json`，没有回填之前的基准报告。

## 60 秒素材上的 CUDA 训练

2026-11-19，同上方的 RTX 2060 和驱动。用户提供的 1280×720、25 FPS、1,500 帧素材有约 1.08 GiB 的预处理缓存。原训练日志中，固定 batch size 1、CUDA FP32 的第 6–16 步耗时为 21.840–22.159 秒。用独立副本从 step 16 续训，原版 release worker 的 5 个连续热步分别为 22.450、22.416、22.503、22.443、22.485 秒。

用同一份输入临时给原版加计时，step 18/19 的分段时间如下。CUDA 首步包含内核编译，未纳入热步统计。

| 阶段 | Step 18 | Step 19 |
| --- | ---: | ---: |
| 验证整套预处理缓存 | 5,068 ms | 5,098 ms |
| 保存上一步状态到内存 | 5,449 ms | 5,416 ms |
| 恢复独立候选状态 | 5,581 ms | 5,571 ms |
| 组装一对训练帧 | 14 ms | 14 ms |
| 前向、六项损失、反向和 Adam 更新 | 951 ms | 974 ms |
| 验证新状态 | 5,412 ms | 5,421 ms |
| 采样 loss 并写入日志 | 25 ms | 21 ms |

每步的整套预处理缓存共读四次：准备数据时一次，完整 Burn 状态的保存、恢复、更新后检查各一次。由此每步重复读取并计算约 4.3 GiB 的哈希。11 月 19 日的修复版复用启动时验证的缓存身份、音频特征和帧索引，仍在每步检查抽样帧哈希及候选状态的 Burn 记录，保存检查点时也再次全量验证预处理缓存。

| 续训 step 16 → 22 | 原版 release | 修复版 release |
| --- | ---: | ---: |
| 任务启动到 batch 就绪 | 30.051 s | 25.456 s |
| 新 worker 的第一个训练步 | 38.781 s | 17.857 s |
| 5 个热步 | 22.450, 22.416, 22.503, 22.443, 22.485 s | 1.867, 1.519, 1.513, 1.503, 1.504 s |
| 热步中位数 | **22.450 s** | **1.513 s** |
| 末次指标到取消检查点提交 | 27.430 s | 11.503 s |

热步耗时下降 **93.3%**，吞吐约为原来的 **14.8 倍**。两个 worker 使用相同的已验证资源、CUDA FP32、固定 batch size 1、种子 42、学习率 `0.001`、预取容量 2 以及原损失配置。热步从同一 run 的连续 `TrainingMetrics.elapsed_ms` 相减得到，计入上一条指标的读回和写入，不含首次编译和最后的检查点发布。取消延迟包含 worker 收到取消并结束当时的准备工作，以及完整保存；冷启动和保存仍会受内核及系统文件缓存影响。

两版的 6 个总 loss 完全一致，分项最大绝对偏差为 `1.86e-9`。step 22 的模型文件哈希、采样器状态、Adam 的 504 个动量张量与 504 个标量均相同；比较 Adam 按参数读取记录，因为文件内排列不同也会改变文件哈希。修复版从这个检查点重启到 step 25，与同版连续训练到 step 25 的 loss 和状态完全一致。从同一视频的预处理缓存开始从头训练的 3 个热步为 1.945、1.505、1.491 秒，成功取消并保存第 4 步。仅在这台 Windows／RTX 2060 上计时。

本地证据在 `.artifacts/cuda-speed/`：`baseline-events-v2.jsonl`、`baseline-stages.log`、`optimized-events.jsonl`、`compare.json` 和 `adam-diff.json`。原版 worker 的 SHA-256 为 `6370af9ca3c8a3d998b8b81b11bbd123a3216e780beebb4b55c6c29a90256712`，修复版为 `9d3437a751198f4de7895dca462843f029a34b13961a481aecac5aa2b1df25de`。复测时用独立项目副本绑定相同检查点与缓存，选固定 batch size 1，发出 `Start`，收到第 6 条 `Metrics` 后 `Cancel` 并等待检查点；下一次从刚发布的检查点续训。下节 `benchmark` 的单步调用单独验证预处理数据，测生产热步时应使用连续 worker。视频和本地记录不随仓库分发。

### 每次训练启动时校验一次

2026-11-20：从头训练直接使用预处理返回的已校验数据；续训直接使用加载检查点时验证的那一代缓存。进入训练后按路径和长度读取帧，复用内存中的音频特征和采样顺序。模型与 Adam 在设备上更新，保留有限 loss 和梯度检查，成功后一起提交模型、采样器和步数。保存时验证新生成的 Burn 记录并原子提交，不重扫输入。Auto batch 的探测进程独立加载数据，每次探测只做一次全量校验，发生在实际训练循环之前。

训练期间视缓存为不可变。同一任务内，等长的外部修改可能影响随后读取的帧；新任务从磁盘续训会重新验证并拒绝变更，截断的帧在读取时仍会报错。

使用相同 RTX 2060、素材、检查点和固定 batch size 1，从 step 16 续训到 step 22：

| 指标 | 11 月 19 日修复版 | 每任务校验一次 |
| --- | ---: | ---: |
| 任务启动到 batch 就绪 | 25.456 s | 19.270 s |
| 新 worker 的第一个训练步 | 17.857 s | 20.321 s |
| 任务启动到首次 loss | 43.336 s | 39.621 s |
| 5 个热步 | 1.867, 1.519, 1.513, 1.503, 1.504 s | 1.080, 0.727, 0.719, 0.728, 0.722 s |
| 热步中位数 | **1.513 s** | **0.727 s** |
| 末次指标到取消检查点提交 | 11.503 s | 0.931 s |

热步比上一版缩短 52.0%，吞吐约为原始 22.450 s/步的 30.9 倍。首步包含 CUDA 内核编译，有独立进程间波动。取消耗时包含等待当前更新和新检查点的校验、写盘与原子发布。

重跑两份相同输入的项目副本：11 月 19 日 worker 的热步中位数分别为 1.575、1.544 s，取消保存为 11.493、11.447 s；新版分别为 0.753、0.836 s 和 0.954、1.097 s。两轮对照的总 loss 一致，分项最大绝对偏差为 `3.73e-9`；检查点采样器、447 个模型张量和 Adam 的 504 个动量张量及 504 个标量逐项一致。新版从其中一个 step 22 检查点重启训到 step 25，与不中断训练的 7 项 loss、模型和 Adam 状态完全一致。从相同缓存开始从头训练也完成了 4 步并保存，3 个热步为 1.278、0.696、0.691 s。

首轮和对照轮从同一 checkpoint 开始，首步总 loss 相同，但首轮 CUDA 进程在后续步出现不同的浮点结果：到 step 22 的总 loss 最大绝对偏差为 `1.71e-4`，采样器和 Adam 标量步数一致。续训可用；跨进程的 CUDA 浮点结果不承诺逐位相同。

本地证据在 `.artifacts/cuda-speed/`：`first-only-events.jsonl`、`control-*-events.jsonl`、`first-only-compare.json`、`first-only-controls.json` 和 `first-only-replay-control.json`。新版 release worker 的 SHA-256 为 `e6cbdff698eb9aa957d4e8399fed07fbbc56d1eaf10398a7ee5845a4b9c88276`。

## 重跑

在已有项目成功训练、保存检查点后，把 [示例](../../tools/benchmark/spec.example.json) 复制到 `.artifacts` 并改为本机的项目、素材和已校验资源路径。`reference` 指向开发参考张量包（见 [参考工具](../../tools/reference/README.md)），检查点 ID 取项目 `project.json` 的 `checkpoints[].model_ref.checkpoint_id`。运行前退出其他训练、生成和资源密集型任务。

```sh
cargo build --locked --release -p feathertalk-engine --no-default-features --features flex,cuda,vulkan --bin benchmark
target/release/benchmark --spec .artifacts/spec-cuda.json --output .artifacts/reports/benchmark-cuda.json
target/release/benchmark --spec .artifacts/spec-vulkan.json --output .artifacts/reports/benchmark-vulkan.json
target/release/benchmark --spec .artifacts/spec-flex.json --output .artifacts/reports/benchmark-flex.json
target/release/benchmark --compare .artifacts/reports/benchmark-cuda.json .artifacts/reports/benchmark-vulkan.json .artifacts/reports/benchmark-flex.json
```

Windows 用 `target/release/benchmark.exe`。CUDA 构建设置 `CUDARC_CUDA_VERSION=12060`；运行时仅在基准进程环境中把 `CUDA_PATH` 指向验证过的 `cuda-math` 目录。每个 spec 只改变 `device`，保持模型、输入、工具、实际 batch 与设置相同。使用存在但位于源项目之外的 scratch 目录；工具复制已提交记录和准备数据，不改变素材、项目检查点或训练历史。需要先创建报告目录，重复执行使用新报告名。其他平台可用本机支持的 feature 组合，例如 CPU `flex` 或 Apple Silicon `flex,metal`。

目前生产自动选择维持原策略，设置中可以选实际设备。`--compare` 供本次对照使用。保存的结果需要匹配实际硬件、驱动和完整输入才能用于持续选择；Burn 未给出驱动值时，本工具不将历史报告作为可复用结果。Arc B580、AMD、Mac、Linux、最大可用 batch 与长素材峰值尚无测量。批量推理、静态编码缓存、融合和低精度仍需同输入评估并验证输出与恢复语义；这些改动尚未默认启用。
