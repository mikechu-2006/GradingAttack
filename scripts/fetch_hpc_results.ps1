param(
    [string]$User = "pwang176",
    [string]$HostName = "hpc2login.hpc.hkust-gz.edu.cn",
    [string]$RemoteProject = "~/GradingAttack",
    [string]$SshBin = "D:\Git\usr\bin\ssh.exe",
    [string]$ScpBin = "D:\Git\usr\bin\scp.exe"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Remote = "$User@$HostName"

$ResultDest = Join-Path $RepoRoot "result_from_hpc_current\RolePlay\Llama-3.1-8B-Instruct"
$ConfigDest = Join-Path $RepoRoot "configs"
$HpcDest = Join-Path $RepoRoot "hpc"

New-Item -ItemType Directory -Force -Path $ResultDest | Out-Null
New-Item -ItemType Directory -Force -Path $ConfigDest | Out-Null
New-Item -ItemType Directory -Force -Path $HpcDest | Out-Null

Write-Host "[fetch] Checking SSH route..."
& $SshBin -o BatchMode=yes -o ConnectTimeout=8 "$Remote" "hostname" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[fetch] SSH may require password; continuing with scp."
}

Write-Host "[fetch] Copying result files..."
& $ScpBin "$Remote`:$RemoteProject/result/RolePlay/Llama-3.1-8B-Instruct/*_metrics.json" "$ResultDest\"
& $ScpBin "$Remote`:$RemoteProject/result/RolePlay/Llama-3.1-8B-Instruct/*.jsonl" "$ResultDest\"

Write-Host "[fetch] Copying experiment configs..."
& $ScpBin "$Remote`:$RemoteProject/configs/*nodefense*.yaml" "$ConfigDest\"
& $ScpBin "$Remote`:$RemoteProject/configs/*paraphrase*.yaml" "$ConfigDest\"
& $ScpBin "$Remote`:$RemoteProject/configs/*smooth*.yaml" "$ConfigDest\"

Write-Host "[fetch] Copying HPC job scripts when present..."
& $ScpBin "$Remote`:$RemoteProject/hpc/overnight_*.sh" "$HpcDest\"
& $ScpBin "$Remote`:$RemoteProject/hpc/run_*.sh" "$HpcDest\"

Write-Host "[fetch] Done. Latest local result files:"
Get-ChildItem -Path $ResultDest -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 12 FullName, Length, LastWriteTime
