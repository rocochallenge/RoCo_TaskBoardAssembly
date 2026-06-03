param(
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"

function Require-Command {
    param([string]$Name)

    if (-not (Get-Command $Name -ErrorAction SilentlyContinue)) {
        throw "Required command '$Name' was not found on PATH."
    }
}

Require-Command git

$repoRoot = git rev-parse --show-toplevel 2>$null
if (-not $repoRoot) {
    throw "This script must be run from inside a Git repository."
}

Set-Location $repoRoot

git lfs version *> $null
if ($LASTEXITCODE -ne 0) {
    throw "Git LFS is not available. Install Git LFS first, then rerun this script."
}

$patterns = @(
    "*.usd",
    "*.usdc",
    "*.obj",
    "*.mp4"
)

Write-Host "Configuring conservative Git LFS tracking in $repoRoot"
Write-Host "Patterns:"
$patterns | ForEach-Object { Write-Host "  $_" }

if ($DryRun) {
    Write-Host "Dry run only. No files were changed."
    exit 0
}

git lfs install
foreach ($pattern in $patterns) {
    git lfs track $pattern
}

Write-Host ""
Write-Host "Done. Review .gitattributes, then stage it before adding matching assets:"
Write-Host "  git add .gitattributes"
Write-Host "  git add <files>"
