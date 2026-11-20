# 发布验收方法

## 先跑实机工作流

**Win11 / Arc B580：** 开发工作树已有 `.artifacts/task18/arc-b580-local-kit.zip`（约 137 MiB），SHA-256 `7c5b90be7121ea0434736436b2bd6f7cb4a9c4a8eb498b8008e2c6f50ff4b8a1`。它含 Windows x64 release Vulkan/CPU 组件、已校验的本地公用资产和 FFmpeg/FFprobe，不含人物素材。供本地开发验收；公用资产的分发状态见 [资源说明](../../resources/README.md)。将归档复制到测试机并解压，安装 Intel Vulkan 驱动，准备一段含可见人脸和音轨的视频，再准备一段独立驱动音频。在解压后的 `arc-b580-local-kit` 目录打开 PowerShell：

```powershell
& .\run.ps1 -Runner .\hardware_acceptance.exe -Packages .\packages `
  -Video 'D:\素材\人物.mp4' -Audio 'D:\素材\驱动.wav' `
  -Backend vulkan -Device 'B580' -Output .\b580-smoke
```

`b580-smoke` 必须是新目录。验收程序无需 Rust、Node 或 Python，也不依赖构建机的路径。用 `B580` 可匹配带 `(TM)` 等标记的设备名。完成后保留 `system.json`、`smoke-result.json`、`task-events.jsonl` 和 `smoke.log`；视频、检查点和缓存可留在本机。CPU 复测使用 `-Backend cpu -Device 'Host CPU'` 和另一个输出目录。该归档已解压到新路径，在无开发工具的 PATH 上通过 RTX 2060 的完整流程；B580 仍待实机结果。

**从源码准备包：** 按 [资源说明](../../resources/README.md) 准备允许本地使用的公用资产（含均值脸）、目标 FFmpeg/FFprobe、worker/probe 和 catalog。CUDA 另需同架构的 `cuda-math`。在 Windows 开发机上构建同一验收程序后，可用下列命令生成可搬移目录：

```powershell
node tools/package/build-runtime.mjs --target x86_64-pc-windows-msvc --backend cpu --output .artifacts/staged
node tools/package/build-runtime.mjs --target x86_64-pc-windows-msvc --backend vulkan --output .artifacts/staged
node tools/package/build-catalog.mjs --artifacts .artifacts/staged --output .artifacts/b580/catalog.json --base-imports .artifacts/model-media-imports.json
$json = cargo test --locked -p feather-talk-gui --test hardware_acceptance --no-run --message-format=json
$runner = $json | ForEach-Object { $_ | ConvertFrom-Json } | Where-Object { $_.target.name -eq 'hardware_acceptance' -and $_.executable } | Select-Object -ExpandProperty executable
python -B tools/acceptance/stage_smoke_kit.py --packages .artifacts/b580/packages --runner $runner --output .artifacts/b580-kit
```

没有名为 `python` 的解释器时，给 `build-catalog.mjs` 传 `--python <解释器路径>`，并用同一解释器执行 `stage_smoke_kit.py`。在 Windows 开发机也可省略 `-Runner`，脚本通过 `cargo --offline` 执行测试，需要先缓存 Rust 依赖。Linux 和 Mac 使用同一测试，参数依次是 packages 路径、人物视频、驱动音频、后端、设备名中的唯一子串及全新目录：

```sh
bash tools/acceptance/run.sh .artifacts/local/packages ./person.mp4 ./drive.wav metal 'Apple M2' .artifacts/apple-smoke
```

CPU 使用 `cpu 'Host CPU'`，CUDA 示例为 `cuda 'RTX 2060'`，Linux 的 AMD 或 Arc 选 `vulkan` 并填实际设备名。脚本要求训练、续训和生成都报告同一执行设备，验证取消发布的 Burn 检查点、历史回读、续训推进、导出帧数和编码。保存 `system.*`、`task-events.jsonl`、`smoke.log`、`smoke-result.json` 和输出视频。播放器和网络隔离由下面的安装验收确认。

## 使用安装包验证

- 按 [资源说明](../../resources/README.md#build-an-installer-or-run-the-desktop-in-development) 构建对应目标的包，并用新用户数据目录安装。记录安装包 SHA-256、实际系统版本、签名、驱动、组件版本与模型清单。确认本机所有后端的随包资源通过校验、实际设备可被探测，无需下载或手动导入。单独验证有下载地址的更新资源。Windows 准备 WebView2 和 Visual C++ x64 运行库；Linux 验证 WebKitGTK/GStreamer 能解码 H.264/AAC。
- 在专用验收环境断开网络并重新打开应用。用一段视频创建人物，从头一键训练，检查实时曲线，取消保存；重开恢复历史，从检查点续训，然后选驱动音频导出并播放。确认实际 GPU 或 CPU、画面和音轨。用中文/空格输入输出路径；验证损坏资源、OOM、驱动和编码失败的提示，编码失败保留已有成品。
- 用同一人物视频和驱动音频生成 Python 参考成品，记录素材、参考和实际输出的哈希。对照同一规范化输入检查口型、画质、回贴边界及帧间稳定性。
- macOS 在清单生成前给 worker、probe 和媒体工具签名；随后签主应用和安装包、公证并 stapling。组件随包放在 `Contents/Resources/bootstrap`。在构建机设置 `FT_COMPONENT_SIGNING_IDENTITY`，并提供已签名的媒体工具。验证 `codesign --verify --deep --strict`、Gatekeeper、公证票据、缓存中的执行权限和实际启动。

## 汇总证据

将 `build-*.json` 和 `acceptance-*.json` 放在同一个报告目录，所引用的日志用相对路径随报告留存。构建报告可由 `build_report.py` 从实际安装包目录生成。硬件报告使用 [示例](../../tools/acceptance/report.example.json)，从实测摘要、系统记录与人工步骤填写。CPU 的硬件类和后端写 `cpu`（应用显示 `flex`）；`components.<role>.target` 用 Rust triple，系统版本用 `10.0.<build>`、`24.04` 或 `13.x` 等格式。下载、断网、签名、播放和画质需要各自的实测记录。

```sh
python -B tools/acceptance/check_reports.py --matrix resources/targets.json --reports .artifacts/reports --require-complete
```

退出码 0 表示所有目标与设备行均有符合格式、保留日志的证据。退出码 1 表示缺少必需结果，2 表示报告或清单无效。省略 `--require-complete` 可整理进度。检查器校验覆盖和基本身份，结果仍需对照实际下载、设备和播放记录。
