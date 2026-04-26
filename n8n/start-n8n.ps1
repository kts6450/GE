# n8n launcher — handles N8N_DISABLE_UI override, picks free port, kills stale node, waits for UI assets.
$ErrorActionPreference = "Continue"
Set-Location -LiteralPath $PSScriptRoot

# 1) Strip hostile env overrides (session + User-scope), then force defaults
foreach ($name in @("N8N_DISABLE_UI", "N8N_PATH", "N8N_EDITOR_BASE_URL")) {
  Remove-Item "Env:\$name" -ErrorAction SilentlyContinue
  try { [System.Environment]::SetEnvironmentVariable($name, $null, 'User') } catch { }
}
$env:N8N_DISABLE_UI = "false"
$env:N8N_PATH = "/"

# 2) Pick a free port (16888 first; TCube usually holds 5678)
function Test-PortHasListener([int]$Port) {
  $null -ne (Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue)
}
function Get-RandomFreePort {
  $l = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, 0)
  try { $l.Start(); return $l.LocalEndpoint.Port } finally { $l.Stop() }
}

$candidates = @(16888, 5678, 17888, 18888, 20888, 25678, 33888)
$chosen = $null
foreach ($p in $candidates) { if (-not (Test-PortHasListener $p)) { $chosen = $p; break } }
if ($null -eq $chosen) { $chosen = Get-RandomFreePort }

# 3) If a stale n8n already owns that port, terminate it
$owner = Get-NetTCPConnection -LocalPort $chosen -State Listen -ErrorAction SilentlyContinue |
  Select-Object -First 1 -ExpandProperty OwningProcess
if ($owner) {
  $proc = Get-Process -Id $owner -ErrorAction SilentlyContinue
  if ($proc -and $proc.ProcessName -ieq 'node') {
    Write-Host "[n8n] Stopping stale node PID $owner on port $chosen" -ForegroundColor Yellow
    Stop-Process -Id $owner -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 700
  }
}

$env:N8N_PORT = "$chosen"
Set-Content -LiteralPath (Join-Path $PSScriptRoot ".n8n-ui-port") -Value $chosen -Encoding utf8 -ErrorAction SilentlyContinue

Write-Host ""
Write-Host "  [n8n] Open: http://127.0.0.1:$chosen/" -ForegroundColor Cyan
Write-Host ""

if (-not (Get-Command n8n -ErrorAction SilentlyContinue)) {
  Write-Host "  [n8n] 'n8n' not in PATH. Run:  npm install -g n8n" -ForegroundColor Red
  exit 1
}

# 4) Background watcher: open browser when UI assets become available (max ~60s)
$port = $chosen
$watcher = Start-Job -ScriptBlock {
  param($port)
  $deadline = (Get-Date).AddSeconds(60)
  while ((Get-Date) -lt $deadline) {
    try {
      $r = Invoke-WebRequest -Uri "http://127.0.0.1:$port/" -UseBasicParsing -MaximumRedirection 0 -TimeoutSec 2
      if ($r.StatusCode -eq 200 -and $r.Content -match '<html') {
        Start-Process "http://127.0.0.1:$port/"
        return "ready"
      }
    } catch {
      $code = $_.Exception.Response.StatusCode.value__
      if ($code -eq 302 -or $code -eq 301) {
        Start-Process "http://127.0.0.1:$port/"
        return "ready-redirect"
      }
    }
    Start-Sleep -Seconds 2
  }
  return "timeout"
} -ArgumentList $port

try {
  & n8n start
}
finally {
  if ($watcher) { Stop-Job $watcher -ErrorAction SilentlyContinue | Out-Null; Remove-Job $watcher -ErrorAction SilentlyContinue | Out-Null }
}
