# 本地演示启动器：只启动本项目服务，不重建或清空数据库。
param([ValidateRange(1, 65535)][int]$Port = 8000)
$ErrorActionPreference = 'Stop'
$projectPath = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectPath '.venv\Scripts\python.exe'
$runtimePath = Join-Path $projectPath '.demo-run'
$url = "http://127.0.0.1:$Port/"

try {
    $response = Invoke-WebRequest -Uri $url -TimeoutSec 2 -UseBasicParsing
    if ($response.Content -match 'PYTHON AI / WORKSPACE') {
        Write-Output "Project is already running: $url"
        exit 0
    }
    throw "Port $Port is occupied by another application."
} catch {
    if ($_.Exception.Message -like '*occupied*') { throw }
}
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw 'Virtual environment missing. Run uv sync in the project directory first.'
}
New-Item -ItemType Directory -Path $runtimePath -Force | Out-Null
$previousUtf8 = $env:PYTHONUTF8
try {
    $env:PYTHONUTF8 = '1'
    $serverProcess = Start-Process -FilePath $pythonPath -ArgumentList @(
        '-m', 'uvicorn', 'python_ai.main:app', '--host', '127.0.0.1', '--port', "$Port"
    ) -WorkingDirectory $projectPath -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $runtimePath 'stdout.log') `
        -RedirectStandardError (Join-Path $runtimePath 'stderr.log')
} finally {
    $env:PYTHONUTF8 = $previousUtf8
}
$serverProcess.Id | Set-Content -LiteralPath (Join-Path $runtimePath 'server.pid')
for ($attempt = 0; $attempt -lt 120; $attempt++) {
    if ($serverProcess.HasExited) {
        throw "Application startup failed. See $runtimePath\stderr.log"
    }
    try {
        $response = Invoke-WebRequest -Uri $url -TimeoutSec 1 -UseBasicParsing
        if ($response.StatusCode -eq 200 -and $response.Content -match 'PYTHON AI / WORKSPACE') {
            Write-Output "Project is ready: $url (PID $($serverProcess.Id))"
            Write-Output "Logs: $runtimePath"
            exit 0
        }
    } catch { }
    Start-Sleep -Milliseconds 500
}
throw "Startup is taking longer than expected. See $runtimePath\stderr.log"
