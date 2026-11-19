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
