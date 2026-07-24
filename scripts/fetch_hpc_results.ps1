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

$ResultDestAs = Join-Path $RepoRoot "artifacts\hpc\roleplay\attention_sharpening\runs"
$ResultDestHs = Join-Path $RepoRoot "artifacts\hpc\roleplay\hijacking_suppression\runs"
$ConfigDest = Join-Path $RepoRoot "configs\roleplay"
$HpcDest = Join-Path $RepoRoot "hpc"

New-Item -ItemType Directory -Force -Path $ResultDestAs | Out-Null
New-Item -ItemType Directory -Force -Path $ResultDestHs | Out-Null
New-Item -ItemType Directory -Force -Path $ConfigDest | Out-Null
New-Item -ItemType Directory -Force -Path $HpcDest | Out-Null

Write-Host "[fetch] Checking SSH route..."
& $SshBin -o BatchMode=yes -o ConnectTimeout=8 "$Remote" "hostname" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "[fetch] SSH may require password; continuing with scp."
}

Write-Host "[fetch] Copying AS results (new + legacy paths)..."
& $ScpBin "$Remote`:$RemoteProject/result/roleplay/attention_sharpening/*" "$ResultDestAs\" 2>$null
& $ScpBin "$Remote`:$RemoteProject/result/RolePlay/Llama-3.1-8B-Instruct/*attention-sharpening*" "$ResultDestAs\" 2>$null

Write-Host "[fetch] Copying HS results (new + legacy paths)..."
& $ScpBin "$Remote`:$RemoteProject/result/roleplay/hijacking_suppression/*" "$ResultDestHs\" 2>$null
& $ScpBin "$Remote`:$RemoteProject/result/RolePlay/Llama-3.1-8B-Instruct/*hijacking-suppression*" "$ResultDestHs\" 2>$null

Write-Host "[fetch] Copying grid summaries..."
& $ScpBin "$Remote`:$RemoteProject/result/roleplay/*/grid_search/summary.json" "$RepoRoot\artifacts\hpc\roleplay\" 2>$null

Write-Host "[fetch] Copying roleplay configs..."
& $ScpBin -r "$Remote`:$RemoteProject/configs/roleplay" "$ConfigDest\.." 2>$null

Write-Host "[fetch] Copying HPC job scripts when present..."
& $ScpBin "$Remote`:$RemoteProject/hpc/overnight_*.sh" "$HpcDest\"
& $ScpBin "$Remote`:$RemoteProject/hpc/run_*.sh" "$HpcDest\"

Write-Host "[fetch] Done. Latest HS results:"
Get-ChildItem -Path $ResultDestHs -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 8 FullName, Length, LastWriteTime
Write-Host "[fetch] Latest AS results:"
Get-ChildItem -Path $ResultDestAs -File -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 8 FullName, Length, LastWriteTime
