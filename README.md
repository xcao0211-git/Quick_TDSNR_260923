# Quick_TDSNR_260923

2026-09-23 本地软件快照，应用版本 0.2.1。保留参考 Quick_TDSNR 仓库的 src、tests、docs、scripts、resources 结构，同时内置本次实际运行的公共核心源码。

## 安装与启动（Windows）

推荐 Python 3.14；本快照在 Windows / Python 3.14.7 验证。首次安装需要联网下载第三方依赖。克隆或解压后，在仓库根目录运行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.\scripts\setup_dev.ps1
qts --version
qts
```

如果 PowerShell 阻止脚本执行，可直接调用虚拟环境 Python：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-migration.txt
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\qts.exe
```

## 本地核心与算法

本仓库的发行包同时安装 src/quick_tdsnr 和 src/sipi_sparam_core；无需另行下载 sipi-sparam-core。核心保留原 0.1.0 标记及本地连续时间梯形波修正，不冒充外部 0.2 版本。

本快照使用本地 traveling-wave S 参数 VTF 接口，与参考仓库的 Z 域端接 VTF 版本不同。它是独立的本地版本备份，不覆盖参考仓库或其他核心仓库。请使用新虚拟环境，避免同时安装另一个核心包。

- 支持可选加载的 XLSX/TXT 波形导出；不支持 TR0。
- XLSX 数值显示、TXT 数值输出均使用科学计数法小数点后三位；TXT 时间仅为秒。
- 梯形波按真实 UI/rise 计算各均匀采样点，不再将边沿参数取整到 dt。

## 示例数据

examples/single_channel 包含人工生成的 50 Ω、500 ps、约 6 dB 单通道 S2P、HSPICE 对比网表和解析验证脚本；examples/waveform_export 包含 XLSX/TXT 示例。没有私人 S30P、用户网表或历史运行缓存。

导入 examples/single_channel/single_channel_50ohm_500ps_6dB.s2p 即可开始，详细设置见该目录使用说明。

qts -test 默认打开空白界面；如需指定自己的工程，先设置环境变量 QUICK_TDSNR_TEST_PROJECT 为工程文件路径。

## 验证与打包

```powershell
python -m pytest -q
qts --runtime-check
python examples/single_channel/generate_and_verify.py
.\scripts\build_release.ps1
```

真实 S24P 验收是可选测试；未提供 dev_samples.local.json 时自动跳过。打包需要 PyInstaller；构建产物在 dist/Quick_TDSNR。

## 文档

- [使用说明](docs/USER_GUIDE.md)
- [波形导出](docs/WAVEFORM_EXPORT.md)
- [连续时间激励](docs/TRAPEZOID_EXACT_TIME.md)
- [发布验证](docs/RELEASE_260923.md)

docs 中旧阶段验收记录用于历史追溯，当前安装方式和发布结果以本 README 和 RELEASE_260923 为准。
