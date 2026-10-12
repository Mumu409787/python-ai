# 只停止 start_demo.ps1 记录的本项目 uvicorn，不停止其他 Python 或 MCP 进程。
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent $PSScriptRoot
$pythonPath = [System.IO.Path]::GetFullPath((Join-Path $projectPath '.venv\Scripts\python.exe'))
$pidPath = Join-Path $projectPath '.demo-run\server.pid'
if (-not (Test-Path -LiteralPath $pidPath)) {
    Write-Output 'No launcher PID file was found.'
    exit 0
}
[int]$serverId = 0
if (-not [int]::TryParse((Get-Content -LiteralPath $pidPath -Raw).Trim(), [ref]$serverId)) {
    throw 'The launcher PID file is invalid.'
}
$serverInfo = Get-CimInstance Win32_Process -Filter "ProcessId = $serverId"
if ($null -eq $serverInfo) {
    Write-Output 'The recorded application process has already exited.'
    exit 0
}
if (-not [string]::Equals($serverInfo.ExecutablePath, $pythonPath, [System.StringComparison]::OrdinalIgnoreCase) `
    -or $serverInfo.CommandLine -notmatch '-m\s+uvicorn\s+python_ai\.main:app') {
    throw 'The recorded PID no longer belongs to this project application; no process was stopped.'
}
Stop-Process -Id $serverId
Write-Output "Stopped this project application (PID $serverId)."
