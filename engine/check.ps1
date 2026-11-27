# Acceptance check for the whole engine workspace (contract section 3.4).
# Lead / acceptance only: everybody else uses the crate-scoped cargo commands.
#
# Runs the steps below in order, stops at the first one that fails and exits with that step's
# exit code (0 when all pass). Written for Windows PowerShell 5.1; keep this file ASCII-only.
#
# If script execution is disabled on this machine, start it with
#     powershell -NoProfile -ExecutionPolicy Bypass -File check.ps1

$steps = @(
    'fmt --all -- --check',
    'clippy --workspace --all-targets -- -D warnings',
    'test --workspace'
)

# cargo reports its progress on stderr. When a caller captures stderr while the error preference
# is 'Stop', Windows PowerShell turns the first progress line into a terminating error. Failures
# are therefore detected through $LASTEXITCODE only.
$ErrorActionPreference = 'Continue'

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    Write-Host 'check.ps1: cargo was not found on PATH'
    exit 1
}

Push-Location -LiteralPath $PSScriptRoot
if (-not $?) {
    Write-Host "check.ps1: cannot enter the script directory '$PSScriptRoot'"
    exit 1
}

$exitCode = 0
try {
    foreach ($step in $steps) {
        Write-Host "==> cargo $step"
        $cargoArgs = $step.Split(' ')
        # Stays in place if cargo cannot even be started, so a stale 0 never counts as a pass.
        $global:LASTEXITCODE = 1
        & cargo $cargoArgs
        if ($LASTEXITCODE -ne 0) {
            $exitCode = $LASTEXITCODE
            Write-Host "check.ps1: FAILED at 'cargo $step' (exit code $exitCode)"
            break
        }
    }
}
finally {
    Pop-Location
}

if ($exitCode -eq 0) {
    Write-Host 'check.ps1: all checks passed'
}
exit $exitCode
