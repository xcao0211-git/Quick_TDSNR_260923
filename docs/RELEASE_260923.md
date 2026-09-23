# 2026-09-23 本地快照验证

应用版本：0.2.1。仓库：xcao0211-git/Quick_TDSNR_260923（私有）。

## 范围

以本地可运行源码为准，参考 Quick_TDSNR 的 src/tests/docs/scripts/resources 结构。
将本地 src/sipi_sparam_core 与应用一并安装；不依赖无法取得的外部核心 0.2。
核心全部 Python 文件及应用 services/domain 文件与原工作目录相同（仅清理部分文件末尾空行），本次整理没有改动计算公式。
原工作目录的软件、Git 远端和历史均未修改。

包装调整：修复 qts 命令注册；安装脚本不再引用相邻核心目录；清除启动器默认私人绝对路径；
qts -test 通过 QUICK_TDSNR_TEST_PROJECT 指定用户工程，未设置时打开空白界面。

## 验证结果

在新建独立虚拟环境安装 requirements-migration.txt 和本项目 .[dev] 后验证：

- Windows / Python 3.14.7。
- `python -m pytest -q`：**239 passed, 3 skipped**，27.64 秒。
- 3 项跳过为需要私人 S24P 和 dev_samples.local.json 的可选真实样本验收，不是算法测试失败。
- `qts --version`：Quick_TDSNR 0.2.1。
- `qts --runtime-check`：退出码 0，完整主窗口运行依赖可导入。
- `python -m pip check`：No broken requirements found。
- 已确认 quick_tdsnr 与 sipi_sparam_core 均从本仓库 src 下加载。
- 原有单通道示例重新生成并验证：Rx 峰值 0.25 V，半高中心 560 ps，最大解析波形误差 5.127842594987442e-15 V。
- XLSX/TXT 示例重新生成；自动测试覆盖格式、精度、完整采样点、可选加载及非整数 UI/rise。

## 示例同步

原示例网表沿用旧离散脉冲在 119 ps 结束的下降沿。本快照的算法采用连续时间取样，
因此将两份人工示例网表、独立 PWL 参考和说明同步为 120 ps，重新生成 CSV/JSON/XLSX/TXT。
没有执行 HSPICE；示例网表尚需在用户的 HSPICE 环境验证。

## 交付边界

- 包含应用、核心、测试、图标、安装/构建脚本、文档与人工示例数据。
- 不包含虚拟环境、私人 S 参数/网表、旧电脑路径注册表、备份、TR0 文件或运行缓存。
- 这是源码仓库；第三方库首次安装需联网。保留 EXE 构建脚本，本次未生成或验证 EXE。
- 本次没有取得 Quick_Sparam 源码，因此未执行其应用级回归；本次整理没有新增核心接口或公式变更。
- 不覆盖 maoduhu823-code/Quick_TDSNR，也不将本地核心冒充该仓库依赖的 Z 域核心 0.2。

仓库后续修改应在提交前重新运行测试。要恢复本次版本，可重新克隆仓库或检出本次初始提交。
