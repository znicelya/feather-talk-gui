# 安装包自带运行资源

目标：`pnpm tauri build` 生成的安装包在空缓存下可以发现计算设备、训练并导出视频。`pnpm tauri dev` 使用同一资源目录。启动期间只从本地导入，不改变用户选择的自定义 catalog 或已安装的历史版本。

- [x] 使用 `resources/targets.json` 选择后端。Windows/Linux x64 打包 CPU、Vulkan、CUDA，Apple Silicon 打包 CPU、Metal，Intel Mac 打包 CPU。每个后端随包提供 worker/probe；同一目标只附一份模型和媒体，CUDA 附 libdevice。
- [x] 从构建机提供的输入目录读取已校验的公共模型、目标 FFmpeg/FFprobe 和 CUDA 组件。打包前校验平台、清单和全部文件，缺件时停止构建。安装包与 CI 使用同一个构建入口。八项打包回归覆盖四种目标、缺件、错架构和损坏文件。
- [x] 启动时通过现有校验缓存导入此平台的所有随包资源；同一 pin 只导入一次。Rust 回归验证其他平台的资源不会导入，损坏资源不会成为可用版本，修复后可重新导入。
- [x] 在 Windows 用独立应用标识安装新版 NSIS，验证空缓存首启。使用安装目录内的组件，实测 CPU、CUDA、Vulkan 的设备发现、从头训练、取消保存、续训和导出。同步主迁移计划与平台矩阵。

Windows 10 22H2 / RTX 2060 本地开发验收：`pnpm tauri build --ci --no-sign` 生成 128.26 MiB NSIS。独立标识 `com.ft.bundledvalidation.20261119` 启动前无资源目录，首次启动自动导入 9 个组件。`pnpm tauri dev` 也在隔离缓存中导入同一组组件，安装目录与两处缓存里的文件 SHA-256 均匹配对应清单。

安装包位于 `.artifacts/acceptance/app-target/x86_64-pc-windows-msvc/release/bundle/nsis/FeatherTalk Bundled Validation_0.1.0_x64-setup.exe`，SHA-256 为 `43e38e47f9b67471e0f1d9bc65b120038e1fa631116c87be82af5a6802a30807`。同一工作树的 `.artifacts/acceptance/` 保留构建与安装证据。Linux、macOS、Win11/Arc B580 尚未做实机验收。

`installed-smoke-{cpu,cuda,vulkan}/smoke-result.json` 记录桌面任务服务直接使用 NSIS 安装目录中的资源，在无开发工具的 PATH 下验收。三种后端均有真实设备、有限 loss、取消时保存第 1 步、重开续训至第 2 步，导出 640×600、29 帧、25 FPS 的 H.264/AAC MP4。新版图形应用验证到启动和自动准备资源；完整流程由桌面任务服务验收程序执行。整机断网、发布签名和其余设备仍按平台矩阵验收。
