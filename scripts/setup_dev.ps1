$ErrorActionPreference = "Stop"
$appRoot = Split-Path -Parent $PSScriptRoot
python -m pip install -r (Join-Path $appRoot "requirements-migration.txt")
if ($LASTEXITCODE -ne 0) { throw "运行依赖安装失败" }
python -m pip install -e "$appRoot[dev]"
if ($LASTEXITCODE -ne 0) { throw "应用安装失败" }
Write-Host "已安装应用及仓库内置核心。启动：qts；测试：python -m pytest -q"
