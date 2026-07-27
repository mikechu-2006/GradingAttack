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
$ArchiveName = "gradingattack_current_results_export.tgz"
$LocalArchive = Join-Path $RepoRoot $ArchiveName
$ExtractDir = Join-Path $RepoRoot ".hpc_results_extract"

Write-Host "[fetch] Creating current result archive on HPC..."
$PackCommand = @"
bash -lc 'cd $RemoteProject && tar --ignore-failed-read -czf ~/$ArchiveName \
  result/RolePlay/Llama-3.1-8B-Instruct \
  result/gcg_suffix_bank/Llama-3.1-8B-Instruct \
  result/experiments/scientsbank_2c \
  configs/experiments/scientsbank_2c \
  logs/experiments/scientsbank_2c \
  hpc/logs/merged_pipeline_*.out \
  hpc/logs/merged_pipeline_*.err \
  output_*.txt err_*.txt'
"@
& $SshBin "$Remote" $PackCommand
if ($LASTEXITCODE -ne 0) { throw "Failed to create archive on HPC." }

Write-Host "[fetch] Copying archive to local repo..."
& $ScpBin "$Remote`:~/$ArchiveName" "$LocalArchive"
if ($LASTEXITCODE -ne 0) { throw "Failed to copy archive from HPC." }

if (Test-Path $ExtractDir) { Remove-Item -Recurse -Force $ExtractDir }
New-Item -ItemType Directory -Force -Path $ExtractDir | Out-Null

Write-Host "[fetch] Extracting archive..."
tar -xzf $LocalArchive -C $ExtractDir

foreach ($dir in @("result", "configs", "logs", "hpc")) {
    $src = Join-Path $ExtractDir $dir
    $dst = Join-Path $RepoRoot $dir
    if (Test-Path $src) {
        New-Item -ItemType Directory -Force -Path $dst | Out-Null
        Copy-Item -Recurse -Force -Path (Join-Path $src "*") -Destination $dst
    }
}

foreach ($pattern in @("output_*.txt", "err_*.txt")) {
    Get-ChildItem -Path $ExtractDir -Filter $pattern -File -ErrorAction SilentlyContinue |
        Copy-Item -Force -Destination $RepoRoot
}

Remove-Item -Force $LocalArchive
Remove-Item -Recurse -Force $ExtractDir

Write-Host "[fetch] Done. Latest metrics:"
Get-ChildItem -Path (Join-Path $RepoRoot "result") -Recurse -File -Filter "*metrics.json" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 16 FullName, Length, LastWriteTime
