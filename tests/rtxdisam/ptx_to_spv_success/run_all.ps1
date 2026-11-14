$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$repo = Split-Path -Parent (Split-Path -Parent $root)
$exe = Join-Path $repo 'src\rtxdisam.cli\bin\Debug\net8.0\rtxdisam.dll'

if (-not (Test-Path $exe)) {
    Write-Error "Build the app first: dotnet build $repo\src\rtxdisam.cli\rtxdisam.cli.csproj -c Debug"
    exit 1
}

$failures = @()
Get-ChildItem (Join-Path $root 'cases') -Directory | Sort-Object Name | ForEach-Object {
    $caseDir = $_.FullName
    $ptx = Get-ChildItem $caseDir -Filter *.ptx | Select-Object -First 1
    $config = @{}
    Get-Content (Join-Path $caseDir 'case.txt') | ForEach-Object {
        if ($_ -match '^\s*([^#=]+)=(.*)$') {
            $config[$matches[1].Trim()] = $matches[2].Trim()
        }
    }
    if (-not $config.ContainsKey('entry') -or
        (-not $config.ContainsKey('expected') -and
         -not $config.ContainsKey('expected_regex'))) {
        Write-Host "FAIL $($_.Name): case.txt needs entry and expected/expected_regex"
        $failures += $_.Name
        return
    }
    $entry = $config['entry']
    $expected = $config['expected']
    $expectedRegex = $config['expected_regex']
    $argList = @()
    if ($config.ContainsKey('args') -and $config['args']) {
        $argList = $config['args'] -split '\s+'
    }

    Write-Host "=== $($_.Name) ==="
    $savedErrorAction = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    $output = & dotnet $exe --run-ptx $ptx.FullName --ptx-entry $entry @argList 2>&1 | Out-String
    $ErrorActionPreference = $savedErrorAction
    Write-Host $output.Trim()
    if ($output -match 'needs 32-lane warp semantics') {
        Write-Host "SKIP $($_.Name) (device has no 32-lane subgroup support)"
        return
    }
    if ($expectedRegex) {
        $ok = $output -match $expectedRegex
    } else {
        $ok = $output -match [regex]::Escape($expected)
    }
    if ($output -notmatch 'validation_errors=0') {
        $ok = $false
    }
    if ($ok) {
        Write-Host "PASS $($_.Name)"
    } else {
        Write-Host "FAIL $($_.Name)"
        $failures += $_.Name
    }
}

if ($failures.Count -gt 0) {
    Write-Host "Failed: $($failures -join ', ')"
    exit 1
}
Write-Host "All success cases passed."
