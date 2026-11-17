$ErrorActionPreference = 'Continue'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$cubin = Join-Path $root 'atom_add_uniform.cubin'
if (-not (Test-Path $cubin)) {
    & (Join-Path $root 'build_cubin.bat') | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'FAIL: ptxas build failed'
        exit 1
    }
}

$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $root))
$exe = Join-Path $repo 'src\rtxdisam.cli\bin\Debug\net8.0\rtxdisam.dll'
$stdout = [IO.Path]::GetTempFileName()
$stderr = [IO.Path]::GetTempFileName()
try {
    $process = Start-Process -FilePath 'dotnet' -ArgumentList @(
        $exe, '--run-cubin', $cubin, '--ptx-entry', 'k',
        '--run-buffer-hex', '8:290000000a000000',
        '--run-buffer-hex', '8:0000000000000000'
    ) -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
      -NoNewWindow -Wait -PassThru
    $output = (Get-Content $stdout -Raw) + (Get-Content $stderr -Raw)
    $code = $process.ExitCode
} finally {
    Remove-Item $stdout, $stderr -Force -ErrorAction SilentlyContinue
}
Write-Host $output.Trim()

if ($code -eq 0 -and
    $output -match [regex]::Escape('buf1[0]=51') -and
    $output -match [regex]::Escape('buf1[1]=41')) {
    Write-Host 'PASS (interleaved XMAD.MRG shared atomic add)'
    exit 0
}
Write-Host 'FAIL: expected buf1[0]=51 and buf1[1]=41'
exit 1
