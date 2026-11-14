$ErrorActionPreference = 'Continue'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$cubin = Join-Path $root 'dynamic_ldc.cubin'
if (-not (Test-Path $cubin)) {
    & (Join-Path $root 'build_cubin.bat') | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'FAIL: SASS assembly failed'
        exit 1
    }
}

$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $root))
$exe = Join-Path $repo 'src\rtxdisam.cli\bin\Debug\net8.0\rtxdisam.dll'
$stdout = [IO.Path]::GetTempFileName()
$stderr = [IO.Path]::GetTempFileName()
try {
    $process = Start-Process -FilePath 'dotnet' -ArgumentList @(
        $exe, '--lift-ptx', $cubin, '--ptx-entry', 'k',
        '--ptx-param', 'p0:u64:0x140'
    ) -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
      -NoNewWindow -Wait -PassThru
    $output = (Get-Content $stdout -Raw) + (Get-Content $stderr -Raw)
    $code = $process.ExitCode
} finally {
    Remove-Item $stdout, $stderr -Force -ErrorAction SilentlyContinue
}
Write-Host $output.Trim()

if ($code -eq 0 -and
    $output -match 'ld\.const\.u32 %r2, \[%rt' -and
    $output -match 'ld\.const\.u32 %r3, \[%rt') {
    Write-Host 'PASS (dynamic bank-0 LDC)'
    exit 0
}
Write-Host 'FAIL: expected dynamic constant-bank loads'
exit 1
