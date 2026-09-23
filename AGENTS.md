# Quick_TDSNR_260923 快照仓库

- 本仓库保存 2026-09-23 本地可运行应用及其实际使用的核心快照。
- 核心仅位于 src/sipi_sparam_core，不依赖外部核心 0.2；不得重复实现算法。
- 核心禁止 import PyQt、PySide、qtpy 或 matplotlib；应用 domain/services 禁止 import Qt。
- 公共端口号为 1-based；数值公式变更前先增加解析或黄金用例。
- 不在模块级调用 matplotlib.use；耗时任务通过 worker 调用服务层。
- 提交前运行 python -m pytest -q；公共接口变化还需 Quick_Sparam 回归。
- 不提交私人仿真数据、本机样例注册表、凭据、备份或缓存。
