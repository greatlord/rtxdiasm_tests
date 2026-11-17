$ErrorActionPreference = 'Continue'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$cubin = Join-Path $root 'sm35_brx.cubin'
if (-not (Test-Path $cubin)) {
    & (Join-Path $root 'build_cubin.bat') | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'FAIL: ptxas build failed'
        exit 1
    }
}

$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $root))
$exe = Join-Path $repo 'src\rtxdisam.cli\bin\Debug\net8.0\rtxdisam.dll'

function Invoke-Case([string]$selector) {
    $stdout = [IO.Path]::GetTempFileName()
    $stderr = [IO.Path]::GetTempFileName()
    try {
        $process = Start-Process -FilePath 'dotnet' -ArgumentList @(
            $exe, '--run-cubin', $cubin, '--ptx-entry', 'k',
            '--run-no-buffer', '--run-scalar', $selector
        ) -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
          -NoNewWindow -Wait -PassThru
        $output = (Get-Content $stdout -Raw) + (Get-Content $stderr -Raw)
        $code = $process.ExitCode
    } finally {
        Remove-Item $stdout, $stderr -Force -ErrorAction SilentlyContinue
    }
    Write-Host $output.Trim()
    return ($code -eq 0 -and $output -match 'validation_errors=0')
}

$ok = $true
foreach ($selector in 0, 1, 2) {
    if (-not (Invoke-Case "$selector")) {
        $ok = $false
    }
}
if ($ok) {
    Write-Host 'PASS (BRX jump table 0..2)'
    exit 0
}
Write-Host 'FAIL: BRX jump-table run mismatch'
exit 1
