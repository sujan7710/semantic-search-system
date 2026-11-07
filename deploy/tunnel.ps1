# Public demo from this PC through a Cloudflare quick tunnel (no account needed).
#
#   powershell -ExecutionPolicy Bypass -File deploy\tunnel.ps1
#
# Starts CacheGuard on 127.0.0.1 only (reachable just through the tunnel),
# opens the tunnel, prints the public https://*.trycloudflare.com link, and
# keeps running until you press Ctrl+C. The link changes on every start and
# works only while this PC and this script are running.
#
# Needs: .env with GROQ_API_KEY, and cloudflared.exe in %LOCALAPPDATA%\cloudflared
# (or on PATH). Protection stays on: 30 requests/min per visitor, 300 AI calls/day.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
$port = if ($env:CACHEGUARD_PORT) { $env:CACHEGUARD_PORT } else { "8000" }
New-Item -ItemType Directory -Force -Path "artifacts" | Out-Null

$cloudflared = (Get-Command cloudflared -ErrorAction SilentlyContinue).Source
if (-not $cloudflared) { $cloudflared = Join-Path $env:LOCALAPPDATA "cloudflared\cloudflared.exe" }
if (-not (Test-Path $cloudflared)) { throw "cloudflared not found. Download it from https://github.com/cloudflare/cloudflared/releases" }
if (-not (Test-Path ".env")) { throw ".env with GROQ_API_KEY is missing (see .env.example)" }

# Admin token for /admin/whoami and cache admin endpoints. Kept in artifacts/ (git-ignored), never printed.
$tokenFile = "artifacts\admin-token.txt"
if (-not (Test-Path $tokenFile)) {
    $bytes = New-Object byte[] 32
    [System.Security.Cryptography.RandomNumberGenerator]::Create().GetBytes($bytes)
    [Convert]::ToBase64String($bytes).TrimEnd("=").Replace("+", "-").Replace("/", "_") | Set-Content -NoNewline -Encoding ascii $tokenFile
}

$env:CACHEGUARD_ADMIN_TOKEN = (Get-Content $tokenFile -Raw).Trim()
$env:CACHEGUARD_CLIENT_IP_HEADER = "cf-connecting-ip"   # set by Cloudflare; safe because we bind to 127.0.0.1
$env:CACHEGUARD_RATE_LIMIT_PER_MINUTE = "30"
$env:CACHEGUARD_DAILY_BACKEND_CALLS = "300"

$serverLog = "artifacts\server.log"
$tunnelLog = "artifacts\tunnel.log"
Remove-Item $tunnelLog -ErrorAction SilentlyContinue

Write-Host "Starting CacheGuard on http://127.0.0.1:$port (first start loads the models, ~20-40 s)..."
$server = Start-Process -FilePath "python" -WindowStyle Hidden -PassThru `
    -ArgumentList "-m", "uvicorn", "cacheguard.api:create_default_app", "--factory", "--host", "127.0.0.1", "--port", $port `
    -RedirectStandardOutput $serverLog -RedirectStandardError "$serverLog.err"

$ready = $false
for ($i = 0; $i -lt 90; $i++) {
    if ($server.HasExited) { throw "CacheGuard exited during startup; see $serverLog.err" }
    try { Invoke-RestMethod "http://127.0.0.1:$port/health" -TimeoutSec 2 | Out-Null; $ready = $true; break } catch { Start-Sleep -Seconds 2 }
}
if (-not $ready) { Stop-Process -Id $server.Id -Force; throw "CacheGuard did not become ready; see $serverLog.err" }
Write-Host "CacheGuard is up. Opening the Cloudflare tunnel..."

$tunnel = Start-Process -FilePath $cloudflared -WindowStyle Hidden -PassThru `
    -ArgumentList "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:$port", "--logfile", $tunnelLog

$url = $null
for ($i = 0; $i -lt 60 -and -not $url; $i++) {
    Start-Sleep -Seconds 1
    if (Test-Path $tunnelLog) {
        $m = Select-String -Path $tunnelLog -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" | Select-Object -First 1
        if ($m) { $url = $m.Matches[0].Value }
    }
}
if (-not $url) { Stop-Process -Id $tunnel.Id, $server.Id -Force; throw "No tunnel URL; see $tunnelLog" }

Set-Content -Path "artifacts\public-url.txt" -Value $url -Encoding ascii
Write-Host ""
Write-Host "PUBLIC URL: $url"
Write-Host "Keep this window open. Press Ctrl+C to stop both the tunnel and CacheGuard."
try {
    while (-not $server.HasExited -and -not $tunnel.HasExited) { Start-Sleep -Seconds 5 }
    if ($server.HasExited) { Write-Host "CacheGuard stopped; see $serverLog.err" }
    if ($tunnel.HasExited) { Write-Host "The tunnel stopped; see $tunnelLog" }
} finally {
    foreach ($p in @($tunnel, $server)) { if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force } }
    Write-Host "Stopped."
}
