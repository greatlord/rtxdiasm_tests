$ErrorActionPreference = 'Continue'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$cubin = Join-Path $root 'sm50_pred_atom.cubin'
if (-not (Test-Path $cubin)) {
    & (Join-Path $root 'build_cubin.bat') | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'FAIL: ptxas build failed'
        exit 1
    }
}

$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $root))
$exe = Join-Path $repo 'src\rtxdisam.cli\bin\Debug\net8.0\rtxdisam.dll'

function Invoke-Case(
    [string]$inputHex, [string]$expectedValue, [string]$expectedOld) {
    $stdout = [IO.Path]::GetTempFileName()
    $stderr = [IO.Path]::GetTempFileName()
    try {
        $process = Start-Process -FilePath 'dotnet' -ArgumentList @(
            $exe, '--run-cubin', $cubin, '--ptx-entry', 'k',
            '--run-block', '4',
            '--run-buffer-hex', "4:$inputHex",
            '--run-buffer-hex', '8:0a00000000000000'
        ) -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
          -NoNewWindow -Wait -PassThru
        $output = (Get-Content $stdout -Raw) + (Get-Content $stderr -Raw)
        $code = $process.ExitCode
    } finally {
        Remove-Item $stdout, $stderr -Force -ErrorAction SilentlyContinue
    }
    Write-Host $output.Trim()
    return ($code -eq 0 -and
        $output -match [regex]::Escape($expectedValue) -and
        $output -match [regex]::Escape($expectedOld))
}

$ok1 = Invoke-Case '2a000000' 'buf1[0]=42' 'buf1[1]=10'
$ok2 = Invoke-Case '05000000' 'buf1[0]=10' 'buf1[1]=10'
if ($ok1 -and $ok2) {
    Write-Host 'PASS (predicated atomic max and old value)'
    exit 0
}
Write-Host 'FAIL: predicated atomic result mismatch'
exit 1
