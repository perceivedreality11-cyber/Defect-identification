$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$taskPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    throw '请先按 README 创建 .venv 并安装 .[ui] 依赖。'
}
& $taskPython -m streamlit run app.py
