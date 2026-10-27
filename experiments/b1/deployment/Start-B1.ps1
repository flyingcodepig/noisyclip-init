param(
    [ValidatePattern('^[A-Za-z0-9_-]+$')][string]$RunId = '',
    [switch]$MonitorOnly,
    [switch]$Resume,
    [switch]$SkipPrediction
)
$ErrorActionPreference = 'Stop'
if (-not $RunId) {
    if ($MonitorOnly -or $Resume) { throw 'MonitorOnly/Resume requires -RunId.' }
    $RunId = 'b1_' + (Get-Date -Format 'yyyyMMdd_HHmmss')
}
$repo = '/mnt/noisyclip-b1-20261027'
$py = '/root/b1-env-20261027/bin/python'
$deploy = "$repo/experiments/b1/deployment"
$code = "$repo/experiments/b1/code"
$session = "b1_$RunId"
$localRepo = (Resolve-Path "$PSScriptRoot/../../..").Path
$revision = (& git -C $localRepo rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) { throw 'Cannot determine local Git revision.' }
if (-not $MonitorOnly) {
    & git -C $localRepo diff --quiet HEAD -- experiments/b1/code experiments/b1/deployment
    if ($LASTEXITCODE -ne 0) { throw 'Commit and push code/config changes before launching.' }
}
$options = ''
if ($Resume) { $options += ' --resume' }
if ($SkipPrediction) { $options += ' --skip-prediction' }
$launch = ''
if (-not $MonitorOnly) {
    $launch = @"
if tmux has-session -t '$session' 2>/dev/null; then
    echo 'Existing session found; monitoring only.'
else
    if tmux list-sessions -F '#{session_name}' 2>/dev/null | grep -q '^b1_'; then
        echo 'Another managed B1 session exists. Refusing concurrent launch.'; exit 2
    fi
    git -C '$repo' pull --ff-only
    test "`$(git -C '$repo' rev-parse HEAD)" = '$revision' || { echo 'Remote revision differs from local HEAD; push first.'; exit 2; }
    mkdir -p '$code/runs'
    after=`$(date +%s)
    tmux new-session -d -s '$session' "exec '$py' -u '$deploy/pipeline_a100.py' --run '$RunId'$options > '$code/runs/$RunId.launch.log' 2>&1"
fi
"@
}
$command = @"
set -euo pipefail
after=0
$launch
exec '$py' -u '$deploy/pipeline_a100.py' --run '$RunId' --monitor --after "`$after"
"@
$encoded = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($command.Replace("`r", '')))
Write-Host "Run: $RunId | Server: px-cloud2.matpool.com:28770"
Write-Host "Ctrl+C closes monitoring only; server tmux continues."
Write-Host "Reconnect: .\experiments\b1\deployment\Start-B1.ps1 -MonitorOnly -RunId $RunId"
& ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -p 28770 root@px-cloud2.matpool.com "echo '$encoded' | base64 -d | bash"
if ($LASTEXITCODE -ne 0) { throw "SSH/pipeline exited with code $LASTEXITCODE. Inspect or reconnect using RunId $RunId; do not blindly launch a new run." }
