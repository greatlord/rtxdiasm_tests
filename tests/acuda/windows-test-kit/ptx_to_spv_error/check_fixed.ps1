$ErrorActionPreference = 'Continue'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent $root
$exe = Join-Path $repo 'bin\Debug\net8.0\ConsoleApp2.dll'
$casesRoot = Join-Path $root 'cases'
if (-not (Test-Path $casesRoot)) {
    $casesRoot = $root
}
$successRoot = Join-Path $repo 'ptx_to_spv_success\cases'

$moved = @()
$fixedNoMeta = @()
$stillFailing = @()

Get-ChildItem $casesRoot -Directory | Sort-Object Name | ForEach-Object {
    $caseDir = $_.FullName
    $name = $_.Name
    if ($name -in @('common', 'trap')) {
        return
    }
    $test = Join-Path $caseDir 'test.ps1'
    if (-not (Test-Path $test)) {
        return
    }

    # The case's own test passes while the expected failure still occurs.
    & powershell -NoProfile -ExecutionPolicy Bypass -File $test *> $null
    if ($LASTEXITCODE -eq 0) {
        $stillFailing += $name
        return
    }

    $meta = Join-Path $caseDir 'success.txt'
    if (-not (Test-Path $meta)) {
        $fixedNoMeta += "$name (no success.txt)"
        return
    }

    $config = @{}
    Get-Content $meta | ForEach-Object {
        if ($_ -match '^\s*([^#=]+)=(.*)$') {
            $config[$matches[1].Trim()] = $matches[2].Trim()
        }
    }
    if (-not $config.ContainsKey('entry') -or
        (-not $config.ContainsKey('expected') -and
         -not $config.ContainsKey('expected_regex'))) {
        $fixedNoMeta += "$name (bad success.txt)"
        return
    }

    $ptx = Get-ChildItem $caseDir -Filter *.ptx | Select-Object -First 1
    if (-not $ptx) {
        $fixedNoMeta += "$name (no PTX)"
        return
    }
    $argList = @()
    if ($config.ContainsKey('args') -and $config['args']) {
        $argList = $config['args'] -split '\s+'
    }
    $output = & dotnet $exe --run-ptx $ptx.FullName `
        --ptx-entry $config['entry'] @argList 2>&1 | Out-String

    $ok = $output -match 'validation_errors=0'
    if ($config.ContainsKey('expected_regex')) {
        $ok = $ok -and ($output -match $config['expected_regex'])
    }
    else {
        $ok = $ok -and ($output -match [regex]::Escape($config['expected']))
    }
    if (-not $ok) {
        $fixedNoMeta += "$name (success run failed)"
        return
    }

    $dest = Join-Path $successRoot $name
    New-Item -ItemType Directory -Force -Path $dest | Out-Null
    Copy-Item $ptx.FullName (Join-Path $dest "$name.ptx") -Force
    $caseText = "entry=$($config['entry'])`nargs=$($config['args'])`n"
    if ($config.ContainsKey('expected_regex')) {
        $caseText += "expected_regex=$($config['expected_regex'])`n"
    }
    else {
        $caseText += "expected=$($config['expected'])`n"
    }
    Set-Content (Join-Path $dest 'case.txt') $caseText -NoNewline
    Remove-Item $caseDir -Recurse -Force
    $moved += $name
}

if ($moved.Count -gt 0) {
    Write-Host "Moved to ptx_to_spv_success: $($moved -join ', ')"
}
if ($fixedNoMeta.Count -gt 0) {
    Write-Host "Fixed but not moved: $($fixedNoMeta -join ', ')"
}
if ($stillFailing.Count -gt 0) {
    Write-Host "Still failing: $($stillFailing -join ', ')"
}
if ($moved.Count -eq 0 -and $fixedNoMeta.Count -eq 0) {
    Write-Host 'No fixed cases found.'
}

if ($fixedNoMeta.Count -gt 0) {
    exit 1
}
exit 0
