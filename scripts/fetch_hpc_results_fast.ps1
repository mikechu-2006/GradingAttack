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
$ArchiveName = "gradingattack_results_export.tgz"
$LocalArchive = Join-Path $RepoRoot $ArchiveName
$ExtractDir = Join-Path $RepoRoot ".hpc_results_extract"

$ResultDest = Join-Path $RepoRoot "result_from_hpc_current\RolePlay\Llama-3.1-8B-Instruct"
$ConfigDest = Join-Path $RepoRoot "configs"
$HpcDest = Join-Path $RepoRoot "hpc"

New-Item -ItemType Directory -Force -Path $ResultDest | Out-Null
New-Item -ItemType Directory -Force -Path $ConfigDest | Out-Null
New-Item -ItemType Directory -Force -Path $HpcDest | Out-Null

Write-Host "[fetch] Creating result archive on HPC..."
$PackCommand = "bash -lc 'cd $RemoteProject && tar --ignore-failed-read -czf ~/$ArchiveName result/RolePlay/Llama-3.1-8B-Instruct/*_metrics.json result/RolePlay/Llama-3.1-8B-Instruct/*.jsonl configs/*nodefense*.yaml configs/*paraphrase*.yaml configs/*smooth*.yaml hpc/overnight_*.sh hpc/run_*.sh'"
& $SshBin "$Remote" $PackCommand
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create archive on HPC."
}

Write-Host "[fetch] Copying archive to local repo..."
& $ScpBin "$Remote`:~/$ArchiveName" "$LocalArchive"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to copy archive from HPC."
}

if (Test-Path $ExtractDir) {
    Remove-Item -Recurse -Force $ExtractDir
}
New-Item -ItemType Directory -Force -Path $ExtractDir | Out-Null

Write-Host "[fetch] Extracting archive..."
tar -xzf $LocalArchive -C $ExtractDir

$ExtractedResult = Join-Path $ExtractDir "result\RolePlay\Llama-3.1-8B-Instruct"
if (Test-Path $ExtractedResult) {
    Copy-Item -Force -Path (Join-Path $ExtractedResult "*") -Destination $ResultDest
}

$ExtractedConfigs = Join-Path $ExtractDir "configs"
if (Test-Path $ExtractedConfigs) {
    Copy-Item -Force -Path (Join-Path $ExtractedConfigs "*") -Destination $ConfigDest
}

$ExtractedHpc = Join-Path $ExtractDir "hpc"
if (Test-Path $ExtractedHpc) {
    Copy-Item -Force -Path (Join-Path $ExtractedHpc "*") -Destination $HpcDest
}

Remove-Item -Force $LocalArchive
Remove-Item -Recurse -Force $ExtractDir

Write-Host "[fetch] Done. Latest local result files:"
Get-ChildItem -Path $ResultDest -File |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 12 FullName, Length, LastWriteTime