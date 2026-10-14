# 同时启动 Python 后端 API 和 Bun 前端；Ctrl+C 一起停止。（run.sh 的 Windows 版）
#
# 用法（PowerShell）：  .\run.ps1          或   .\run.ps1 -Mode prod
# 用法（CMD）：         run.cmd            或   run.cmd -Mode prod
# 若提示脚本被禁止运行：powershell -ExecutionPolicy Bypass -File .\run.ps1
#
# 环境变量（与 run.sh 相同）：
#   BACKEND_HOST / BACKEND_PORT   后端地址（默认 127.0.0.1 / 8765）
#   FRONTEND_HOST / FRONTEND_PORT 前端地址（默认 127.0.0.1 / 5173）
#   PYTHON                        Python 解释器（默认依次尝试 py -3、python、python3）
#   MODE                          dev（热重载，默认）或 prod
#
# 兼容 Windows PowerShell 5.1 和 PowerShell 7+。本文件须保存为 UTF-8 with BOM，
# 否则 Windows PowerShell 5.1 会把中文读成乱码。
[CmdletBinding()]
param(
    [ValidateSet('dev', 'prod')]
    [string]$Mode = $(if ($env:MODE) { $env:MODE } else { 'dev' })
)

$ErrorActionPreference = 'Stop'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch { }

$Root = $PSScriptRoot
$BackendHost = if ($env:BACKEND_HOST) { $env:BACKEND_HOST } else { '127.0.0.1' }
$BackendPort = if ($env:BACKEND_PORT) { [int]$env:BACKEND_PORT } else { 8765 }
$FrontendHost = if ($env:FRONTEND_HOST) { $env:FRONTEND_HOST } else { '127.0.0.1' }
$FrontendPort = if ($env:FRONTEND_PORT) { [int]$env:FRONTEND_PORT } else { 5173 }
$OnWindows = $env:OS -eq 'Windows_NT'

function Fail([string]$Message) {
    Write-Host "错误：$Message" -ForegroundColor Red
    exit 1
}

# 只接受真正的可执行文件（排除 .ps1 包装脚本）。
function Find-Exe([string]$Name) {
    $cmd = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cmd) { return $cmd.Source }
    return $null
}

# 找到一个能用的 Python ≥ 3.12。Windows 应用商店的 python.exe 占位程序会被版本检查排除。
function Find-Python {
    # 显式指定了 PYTHON 就只用它，与 run.sh 一致
    if ($env:PYTHON) {
        $candidates = , @($env:PYTHON)
    } else {
        $candidates = @(, @('py', '-3'))
        $candidates += , @('python')
        $candidates += , @('python3')
    }
    foreach ($c in $candidates) {
        $exe = Find-Exe $c[0]
        if (-not $exe) { continue }
        $prefix = @($c | Select-Object -Skip 1)
        try {
            # 不在参数里用双引号：Windows PowerShell 5.1 传给原生程序时会把它们吞掉
            $out = & $exe @prefix -c 'import sys; print(*sys.version_info[:2], sep=chr(46))' 2>$null
        } catch { continue }
        if ($LASTEXITCODE -ne 0 -or -not $out) { continue }
        $ver = [version]("$out".Trim())
        if ($ver -ge [version]'3.12') { return @{ Exe = $exe; Prefix = $prefix; Version = $ver } }
    }
    return $null
}

function Test-Port([string]$TargetHost, [int]$Port) {
    $client = New-Object System.Net.Sockets.TcpClient
    try {
        $task = $client.ConnectAsync($TargetHost, $Port)
        return ($task.Wait(300) -and $client.Connected)
    } catch {
        return $false
    } finally {
        $client.Dispose()
    }
}

function Stop-Tree($Proc) {
    if (-not $Proc -or $Proc.HasExited) { return }
    if ($OnWindows) {
        # /T 连同子进程一起结束
        & taskkill.exe /PID $Proc.Id /T /F 2>&1 | Out-Null
    } else {
        Stop-Process -Id $Proc.Id -Force -ErrorAction SilentlyContinue
    }
}

$python = Find-Python
if (-not $python) {
    if ($env:PYTHON) { Fail "PYTHON=$($env:PYTHON) 不可用或版本低于 3.12" }
    Fail '找不到 Python ≥ 3.12。请安装 Python，或用环境变量 PYTHON 指定解释器路径'
}
$bun = Find-Exe 'bun'
if (-not $bun) { Fail '找不到 bun，请先安装：https://bun.sh' }

if (Test-Port $BackendHost $BackendPort) { Fail "端口 $BackendPort 已被占用（后端是否已在运行？）" }
if (Test-Port $FrontendHost $FrontendPort) { Fail "端口 $FrontendPort 已被占用" }

$FrontendDir = Join-Path $Root 'frontend'
if (-not (Test-Path (Join-Path $FrontendDir 'node_modules'))) {
    Write-Host '安装前端依赖……'
    Push-Location $FrontendDir
    try {
        & $bun install
        if ($LASTEXITCODE -ne 0) { Fail 'bun install 失败' }
    } finally {
        Pop-Location
    }
}

# 子进程继承这些环境变量；结束时恢复，避免污染当前 PowerShell 会话。
$envNames = 'PYTHONPATH', 'COGNITIVE_AI_API', 'HOST', 'PORT', 'NODE_ENV'
$savedEnv = @{}
foreach ($n in $envNames) { $savedEnv[$n] = [Environment]::GetEnvironmentVariable($n, 'Process') }

$backend = $null
$frontend = $null
try {
    # 不要求先 pip install：直接把 src 加入 PYTHONPATH
    $sep = [System.IO.Path]::PathSeparator
    $src = Join-Path $Root 'src'
    $env:PYTHONPATH = if ($savedEnv['PYTHONPATH']) { "$src$sep$($savedEnv['PYTHONPATH'])" } else { $src }

    $backendArgs = @($python.Prefix) + @('-m', 'cognitive_ai.cli', 'ui', '--host', $BackendHost, '--port', "$BackendPort")
    $backend = Start-Process -FilePath $python.Exe -ArgumentList $backendArgs -WorkingDirectory $Root -NoNewWindow -PassThru

    # 等后端就绪（最多约 10 秒）
    $ready = $false
    for ($i = 0; $i -lt 50; $i++) {
        if (Test-Port $BackendHost $BackendPort) { $ready = $true; break }
        if ($backend.HasExited) { break }
        Start-Sleep -Milliseconds 200
    }
    if (-not $ready) { Fail '后端启动失败' }

    $env:COGNITIVE_AI_API = "http://${BackendHost}:$BackendPort"
    $env:HOST = $FrontendHost
    $env:PORT = "$FrontendPort"
    if ($Mode -eq 'prod') {
        $env:NODE_ENV = 'production'
        $frontendArgs = @('server.js')
    } else {
        $frontendArgs = @('--hot', 'server.js')
    }
    $frontend = Start-Process -FilePath $bun -ArgumentList $frontendArgs -WorkingDirectory $FrontendDir -NoNewWindow -PassThru

    Write-Host "打开 http://${FrontendHost}:$FrontendPort  （Ctrl+C 停止）" -ForegroundColor Green
    while (-not $backend.HasExited -and -not $frontend.HasExited) {
        Start-Sleep -Milliseconds 500
    }
    if ($backend.HasExited) { Write-Host '后端已退出' -ForegroundColor Yellow }
    if ($frontend.HasExited) { Write-Host '前端已退出' -ForegroundColor Yellow }
} finally {
    # Ctrl+C 时 PowerShell 也会执行 finally
    Write-Host ''
    Write-Host '正在停止……'
    Stop-Tree $frontend
    Stop-Tree $backend
    foreach ($n in $envNames) { [Environment]::SetEnvironmentVariable($n, $savedEnv[$n], 'Process') }
}
