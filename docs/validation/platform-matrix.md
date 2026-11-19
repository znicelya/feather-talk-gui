# 平台验收状态

目标来自 [`resources/targets.json`](../../resources/targets.json)。以下是 2026-11-19 的本机证据。当前主机是 Windows 10 22H2 x64（10.0.19045）、Intel Core i5-10400、RTX 2060，NVIDIA 驱动 560.94。Windows 10 结果属于开发验证；Windows 发布目标仍按 Win11 的版本核验。

| 构建目标 | 目标下限 | 产物 | 证据 |
| --- | --- | --- | --- |
| `x86_64-pc-windows-msvc` | Windows 11 | NSIS | Release 应用与 CPU worker/probe 构建、独立安装和空缓存首启通过（Windows 10） |
| `x86_64-unknown-linux-gnu` | Ubuntu 24.04 | deb、AppImage | 已配置构建流程，待 Linux 构建和安装 |
| `aarch64-apple-darwin` | macOS 13 | app、dmg | 已配置构建流程，待 Apple Silicon 构建、安装、签名与公证 |
| `x86_64-apple-darwin` | macOS 13 | app、dmg | 已配置构建流程，待 Intel Mac 构建、安装、签名与公证 |

| 必测路径 | 计算后端 | 本机已有证据 | 完整发布验收 |
| --- | --- | --- | --- |
| Windows / NVIDIA | CUDA 或 Vulkan；分别比较 | RTX 2060 CUDA 训练与续训、WebView2 播放；Vulkan release 从头训练、取消保存、重开续训和导出 | 待 Win11、生产首启和整机离线 |
| Windows / AMD | Vulkan | 无目标显卡 | 待实机 |
| Windows / Intel Arc | Vulkan | 已备妥便携验收包；在 RTX 2060 验证可移植流程 | 待 Win11 / Arc B580 |
| Windows / CPU | Flex | release worker 训练并发布 loss，取消保存 step 1、重开续训 step 2、H.264/AAC 导出；随包 CPU 组件哈希一致 | 待 Win11、生产首启和整机离线 |
| Linux / NVIDIA | CUDA 或 Vulkan；分别比较 | 无目标环境 | 待实机 |
| Linux / AMD | Vulkan | 无目标环境 | 待实机 |
| Linux / Intel Arc | Vulkan | 无目标环境 | 待实机 |
| Linux / CPU | Flex | 无目标环境 | 待实机 |
| macOS / Apple Silicon | Metal | 无目标环境 | 待实机 |
| macOS / Intel | Flex CPU | 无目标环境 | 待实机 |

Windows release 开发安装包在 `.artifacts/task18/windows-release-final/bundle/nsis/`。安装器 SHA-256 `45feb746af82b076ea55c4ae243f985ac78c6f0fe065f33f8e8ef77640a79c37`；同一 `windows-release-final/` 下的 `installation-evidence.json` 记录实际安装与版本。安装用独立应用标识和目录。启动后缓存中的 worker、probe 均与安装目录和各自 manifest 一致，应用正常关闭。该包用开发 catalog，CPU 组件随包，模型与媒体组件另行导入。

`.artifacts/task18/cpu-release-smoke/` 和 `.artifacts/task18/relocated kit/result/` 分别为 Flex CPU 与 RTX 2060 Vulkan 的 release worker 证据。两者均从头训练，发布有限 loss、取消保存 step 1，新会话读取历史并续训至 step 2；FFprobe 读出 640×600、29 帧、25 FPS、H.264/AAC。Vulkan 验收从另一个目录解压的归档启动，PATH 中无 Rust、Node 或 Python。Windows CUDA 桌面播放见 [桌面验收](desktop-ui.md)；把同一归档复制到 B580 的操作见 [发布验收方法](release-checklist.md)。

`desktop-checks.yml` 配置了各平台的 Rust、前端与报告测试。渲染数值比较另需本地生成的 Python 参考张量；干净检出运行自包含的帧序检查。`component-builds.yml` 配置在原生 runner 生成安装包，按需编译 GPU 组件，归档保留 macOS/Linux 权限。当前本地发布检查的唯一通过项是 Windows x64 安装器构建。生产 catalog 下载、真人素材相对 Python 的效果、整机断网和其余平台待验收。
