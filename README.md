# FeatherTalk 桌面版

FeatherTalk 使用 Rust、Burn、Tauri 和 React，在本地从一段有音轨的人物视频训练模型，再用单独的驱动音频导出视频。界面采用 shadcn/ui。训练与生成共用一个任务槽；应用运行不需要 Python、PyTorch 或在线推理服务。

1. 首次启动从安装目录校验并导入公共模型、FFmpeg 和计算组件。在 **资源与设备** 查看就绪状态和检测到的设备即可开始；显卡需要与系统匹配的驱动，Windows 的计算组件需要 [Visual C++ x64 运行库](https://learn.microsoft.com/cpp/windows/latest-supported-vc-redist)。训练、续训和导出使用本地缓存。
2. 在 **人物项目** 新建项目，选一段人物视频。保留原素材，预处理结果写到项目目录。连续画面里应有清晰、可检测的人脸和音轨。
3. 在 **训练** 选一键训练，或在高级设置调整总轮数、batch size、学习率等。实时查看 loss 和实际设备，可取消并保存到最近完成的训练步，之后从 Burn 检查点继续。
4. 在 **视频生成** 选择已训练的检查点、驱动音频和保存位置。完成后可播放并打开成品目录。输出是 25 FPS H.264/AAC MP4。实时音频驱动与旧人物权重导入属于后续范围。

开发版已在 Windows 10 x64 实测 Flex CPU、RTX 2060 CUDA 和 Vulkan 的训练、续训与视频导出，并备有可在 Win11 / Arc B580 上直接运行的本地验收包。目标是 Windows 11 x64、Ubuntu 24.04 x64、Apple Silicon macOS 13+ Metal 和 Intel Mac macOS 13+ CPU。Windows/Linux 的安装包包含 CPU、CUDA 和 Vulkan；AMD 与 Arc 使用 Vulkan。各行的构建、安装和硬件实测状态见 [平台矩阵](docs/validation/platform-matrix.md)。

构建需要 Node 24、pnpm 12.21.0、Rust 1.100.0 和 [Tauri 的本机依赖](https://v2.tauri.app/start/prerequisites/)。Python 3.12 供开发阶段打包、转换和验收，安装后的应用无需 Python。把公用资产与组件保留在 Git 外。

按 [资源说明](resources/README.md#build-an-installer-or-run-the-desktop-in-development) 准备 `.artifacts/bundle-inputs`，也可用 `FT_BUNDLE_INPUTS` 指向已有目录。开发启动和构建使用同一套输入：

```sh
pnpm install --frozen-lockfile
pnpm tauri dev
pnpm tauri build --ci --no-sign
```

构建默认使用 Rust 宿主目标，也可传 `--target <triple>`。脚本编译该目标的 worker/probe，校验并附加全部资源；缺件时给出具体路径并中止。安装包输出到 `target/<triple>/release/bundle`。构建机上 `python` 不是可用的 Python 3 时，设置 `FT_PACKAGING_PYTHON` 为解释器的绝对路径。现有用户在应用数据目录中手动放置的 `resource-catalog.json` 仍然优先；想改用新版随包资源时，可在退出应用后备份并移走该覆盖文件。

- [资源配置、开发导入和多平台暂存](resources/README.md)
- [Win11 + Arc B580 等设备的实测步骤与发布检查项](docs/validation/release-checklist.md)
- [训练与导出的实测耗时及复跑方法](docs/validation/performance.md)
