param([switch]$Check)

$ErrorActionPreference = 'Stop'

function To-WslPath([string]$path) {
    $full = [IO.Path]::GetFullPath($path)
    if ($full -match '^([A-Za-z]):\\(.*)$') {
        return '/mnt/' + $matches[1].ToLowerInvariant() + '/' +
            $matches[2].Replace('\', '/')
    }
    return $full.Replace('\', '/')
}

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$script = Join-Path $root 'scan_corpus.sh'
$output = & wsl -d Ubuntu -- bash (To-WslPath $script) 2>&1
$output | ForEach-Object { Write-Host $_ }

$ok = @($output |
    Where-Object { $_ -match '^OK ' } |
    ForEach-Object { ($_ -replace '^OK ', '').Trim() } |
    Sort-Object)
$fail = @($output |
    Where-Object { $_ -match '^FAIL ' } |
    ForEach-Object { ($_ -replace '^FAIL ', '').Trim() } |
    Sort-Object)
$failNames = @($fail | ForEach-Object {
    if ($_ -match '^([^:]+):') { $matches[1].Trim() } else { $_.Trim() }
})

if ($Check) {
    $baselineOkPath = Join-Path $root 'corpus_success.txt'
    $baselineFailPath = Join-Path $root 'corpus_failure.txt'
    if (-not (Test-Path $baselineOkPath) -or
        -not (Test-Path $baselineFailPath)) {
        Write-Error 'Corpus baselines are missing; run scan_corpus.bat first.'
        exit 2
    }
    $baselineOk = @(Get-Content $baselineOkPath | Where-Object { $_ })
    $baselineFail = @(Get-Content $baselineFailPath | ForEach-Object {
        if ($_ -match '^([^:]+):') { $matches[1].Trim() } else { $_.Trim() }
    })

    $regressions = @($baselineOk | Where-Object { $_ -notin $ok })
    $fixed = @($baselineFail | Where-Object { $_ -notin $failNames })
    $newFailures = @($failNames | Where-Object { $_ -notin $baselineFail })

    Write-Host "Corpus check: ok=$($ok.Count) fail=$($fail.Count)"
    if ($regressions.Count -gt 0) {
        Write-Host "Regressions (were OK, now fail):"
        $regressions | ForEach-Object { Write-Host "  $_" }
    }
    if ($newFailures.Count -gt 0) {
        Write-Host "New failures (not in baseline):"
        $newFailures | ForEach-Object { Write-Host "  $_" }
    }
    if ($fixed.Count -gt 0) {
        Write-Host "Fixed since baseline:"
        $fixed | ForEach-Object { Write-Host "  $_" }
    }
    if ($regressions.Count -gt 0) {
        exit 1
    }
    Write-Host 'No corpus regressions.'
    exit 0
}

$manifest = Join-Path $root 'corpus_success.txt'
$ok | Set-Content $manifest
Write-Host "Wrote $manifest ($($ok.Count) entries)"

$failures = Join-Path $root 'corpus_failure.txt'
$fail | Set-Content $failures
Write-Host "Wrote $failures ($($fail.Count) entries)"
