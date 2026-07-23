param(
    [string]$Message = "Add HPC experiment results"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

Write-Host "[git] Current status:"
git status --short

Write-Host "[git] Pulling latest remote changes with rebase..."
git pull --rebase origin main

Write-Host "[git] Staging experiment files..."
git add result_from_hpc_current configs hpc scripts/fetch_hpc_results.ps1 scripts/commit_hpc_results.ps1

Write-Host "[git] Staged diff summary:"
git diff --cached --stat

if (-not (git diff --cached --quiet)) {
    Write-Host "[git] Committing..."
    git commit -m $Message
    Write-Host "[git] Pushing..."
    git push origin main
} else {
    Write-Host "[git] Nothing to commit."
}

Write-Host "[git] Done."
