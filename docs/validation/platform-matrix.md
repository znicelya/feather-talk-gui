# 平台验收状态

目标来自 [`resources/targets.json`](../../resources/targets.json)。以下是 2026-11-19 的本机证据。当前主机是 Windows 10 22H2 x64（10.0.19045）、Intel Core i5-10400、RTX 2060，NVIDIA 驱动 560.94。Windows 10 结果属于开发验证；Windows 发布目标仍按 Win11 的版本核验。

| 构建目标 | 目标下限 | 产物 | 证据 |
| --- | --- | --- | --- |
| `x86_64-pc-windows-msvc` | Windows 11 | NSIS | CPU、CUDA、Vulkan 构建；新版 NSIS 独立安装，空缓存首启导入全部 9 个组件（Windows 10） |
| `x86_64-unknown-linux-gnu` | Ubuntu 24.04 | deb、AppImage | 已配置构建流程，待 Linux 构建和安装 |
| `aarch64-apple-darwin` | macOS 13 | app、dmg | 已配置构建流程，待 Apple Silicon 构建、安装、签名与公证 |
| `x86_64-apple-darwin` | macOS 13 | app、dmg | 已配置构建流程，待 Intel Mac 构建、安装、签名与公证 |

| 必测路径 | 计算后端 | 本机已有证据 | 完整发布验收 |
| --- | --- | --- | --- |
| Windows / NVIDIA | CUDA 或 Vulkan；分别比较 | RTX 2060 CUDA/Vulkan 均用安装包资源从头训练、取消保存、重开续训和导出 | 待 Win11、整机离线 |
| Windows / AMD | Vulkan | 无目标显卡 | 待实机 |
| Windows / Intel Arc | Vulkan | 已备妥便携验收包；在 RTX 2060 验证可移植流程 | 待 Win11 / Arc B580 |
| Windows / CPU | Flex | 用安装包资源训练并发布 loss，取消保存 step 1、重开续训 step 2、H.264/AAC 导出 | 待 Win11、整机离线 |
| Linux / NVIDIA | CUDA 或 Vulkan；分别比较 | 无目标环境 | 待实机 |
| Linux / AMD | Vulkan | 无目标环境 | 待实机 |
| Linux / Intel Arc | Vulkan | 无目标环境 | 待实机 |
| Linux / CPU | Flex | 无目标环境 | 待实机 |
| macOS / Apple Silicon | Metal | 无目标环境 | 待实机 |
| macOS / Intel | Flex CPU | 无目标环境 | 待实机 |

此前只包含 CPU 组件的开发包见迁移工作树 `.artifacts/task18/windows-release-final/`。修复后的包与证据见 [Task 18b](../superpowers/plans/2026-11-19-bundled-runtime-assets.md)。新包包含公共模型、FFmpeg/FFprobe、CPU/CUDA/Vulkan 的 worker/probe 和 libdevice，首次启动导入全部 9 个组件，无需手动准备资源。

`.artifacts/task18/cpu-release-smoke/` 和 `.artifacts/task18/relocated kit/result/` 分别为 Flex CPU 与 RTX 2060 Vulkan 的 release worker 证据。两者均从头训练，发布有限 loss、取消保存 step 1，新会话读取历史并续训至 step 2；FFprobe 读出 640×600、29 帧、25 FPS、H.264/AAC。Vulkan 验收从另一个目录解压的归档启动，PATH 中无 Rust、Node 或 Python。Windows CUDA 桌面播放见 [桌面验收](desktop-ui.md)；把同一归档复制到 B580 的操作见 [发布验收方法](release-checklist.md)。

`desktop-checks.yml` 配置了各平台的 Rust、前端、打包与报告测试。渲染数值比较另需本地生成的 Python 参考张量；干净检出运行自包含的帧序检查。`component-builds.yml` 复用本地构建入口，在原生 runner 接收已准备的资源输入并生成安装包；也可单独编译组件。当前完整发布检查的唯一通过项仍是 Windows x64 安装器构建。真人素材相对 Python 的效果、整机断网和其余平台待验收。
