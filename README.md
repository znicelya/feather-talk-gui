# FeatherTalk 桌面版

FeatherTalk 使用 Rust、Burn、Tauri 和 React，在本地从一段有音轨的人物视频训练模型，再用单独的驱动音频导出视频。界面采用 shadcn/ui。训练与生成共用一个任务槽；应用运行不需要 Python、PyTorch 或在线推理服务。

1. 在 **资源与设备** 准备当前设备的模型和组件。配置有下载地址时可联网准备，开发包也可导入已配置的本地 `manifest.json`。准备完成后的训练、续训和导出使用本地缓存。
2. 在 **人物项目** 新建项目，选一段人物视频。保留原素材，预处理结果写到项目目录。连续画面里应有清晰、可检测的人脸和音轨。
3. 在 **训练** 选一键训练，或在高级设置调整总轮数、batch size、学习率等。实时查看 loss 和实际设备，可取消并保存到最近完成的训练步，之后从 Burn 检查点继续。
4. 在 **视频生成** 选择已训练的检查点、驱动音频和保存位置。完成后可播放并打开成品目录。输出是 25 FPS H.264/AAC MP4。实时音频驱动与旧人物权重导入属于后续范围。

开发版已在 Windows 10 x64 实测 Flex CPU、RTX 2060 CUDA 和 Vulkan 的训练、续训与视频导出，并备有可在 Win11 / Arc B580 上直接运行的本地验收包。目标是 Windows 11 x64、Ubuntu 24.04 x64、Apple Silicon macOS 13+ Metal 和 Intel Mac macOS 13+ CPU。Windows/Linux 计划支持 NVIDIA CUDA/Vulkan、AMD 与 Arc Vulkan，以及 Flex CPU。各行的构建、安装和硬件实测状态见 [平台矩阵](docs/validation/platform-matrix.md)。当前随包目录是开发配置，需要已校验的本地资源；正式首启下载地址和对应发布验收尚未完成。

构建需要 Node 24、pnpm 12.21.0、Rust 1.100.0 和 [Tauri 的本机依赖](https://v2.tauri.app/start/prerequisites/)。Python 3.12 供开发阶段打包、转换和验收，安装后的应用无需 Python。把公用资产与组件保留在 Git 外。

```sh
pnpm install --frozen-lockfile
pnpm tauri dev
```

构建可安装开发包（替换示例 target 和 Python 命令）：

```sh
node tools/package/build-runtime.mjs --target x86_64-pc-windows-msvc --backend cpu --output .artifacts/staged
node tools/package/build-catalog.mjs --artifacts .artifacts/staged --output .artifacts/desktop/catalog.json --bundle-target x86_64-pc-windows-msvc
pnpm tauri build --ci --no-sign --target x86_64-pc-windows-msvc --config .artifacts/desktop/tauri-overlay.json -- --locked
```

需要其他硬件时，另建对应后端的 worker/probe，再为同一 target 生成目录和安装包。没有生成的 overlay 时，`tauri build` 只构建应用本体。使用已知来源的 FFmpeg/FFprobe（需支持 H.264/AAC）；CUDA 还需要对应驱动与已校验的 `libdevice` 组件。应用从随包目录导入 CPU worker/probe，其他资源通过 **资源与设备** 准备。

- [资源配置、开发导入和多平台暂存](resources/README.md)
- [Win11 + Arc B580 等设备的实测步骤与发布检查项](docs/validation/release-checklist.md)
- [训练与导出的实测耗时及复跑方法](docs/validation/performance.md)
