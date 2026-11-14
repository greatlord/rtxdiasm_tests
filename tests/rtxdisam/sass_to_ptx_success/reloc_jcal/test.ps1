$ErrorActionPreference = 'Continue'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$obj = Join-Path $root 'reloc_jcal.o'
if (-not (Test-Path $obj)) {
    & (Join-Path $root 'build_cubin.bat') | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'FAIL: ptxas relocatable build failed'
        exit 1
    }
}

$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $root))
$exe = Join-Path $repo 'src\rtxdisam.cli\bin\Debug\net8.0\rtxdisam.dll'
$stdout = [IO.Path]::GetTempFileName()
$stderr = [IO.Path]::GetTempFileName()
try {
    $process = Start-Process -FilePath 'dotnet' -ArgumentList @(
        $exe, '--run-cubin', $obj, '--ptx-entry', 'k',
        '--run-buffer', '4:0', '--run-scalar', '3'
    ) -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
      -NoNewWindow -Wait -PassThru
    $output = (Get-Content $stdout -Raw) + (Get-Content $stderr -Raw)
    $code = $process.ExitCode
} finally {
    Remove-Item $stdout, $stderr -Force -ErrorAction SilentlyContinue
}
Write-Host $output.Trim()

if ($code -eq 0 -and $output -match 'validation_errors=0' -and
    $output -match 'buf0\[0\]=10') {
    Write-Host 'PASS (relocatable JCAL / separate .text)'
    exit 0
}
Write-Host 'FAIL: relocatable JCAL result mismatch'
exit 1
