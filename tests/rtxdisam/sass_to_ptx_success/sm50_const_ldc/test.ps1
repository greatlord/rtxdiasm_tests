$ErrorActionPreference = 'Continue'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$cubin = Join-Path $root 'sm50_const_ldc.cubin'
if (-not (Test-Path $cubin)) {
    & (Join-Path $root 'build_cubin.bat') | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'FAIL: SASS assembly failed'
        exit 1
    }
}

$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $root))
$exe = Join-Path $repo 'src\rtxdisam.cli\bin\Debug\net8.0\rtxdisam.dll'
$lifted = Join-Path ([IO.Path]::GetTempPath()) `
    ('rtxdisam_const_ldc_' + [Guid]::NewGuid().ToString('N') + '.ptx')
try {
    & dotnet $exe --lift-ptx $cubin --ptx-entry k `
        --ptx-param p0:u32:0x140 --ptx-param p1:u64:0x148 `
        --ptx-output $lifted | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'FAIL: SASS -> PTX lift failed'
        exit 1
    }

    function Invoke-Case([string]$scalar, [string]$expected) {
        $stdout = [IO.Path]::GetTempFileName()
        $stderr = [IO.Path]::GetTempFileName()
        try {
            $process = Start-Process -FilePath 'dotnet' -ArgumentList @(
                $exe, '--run-ptx', $lifted, '--ptx-entry', 'k',
                '--run-scalar', $scalar,
                '--run-buffer-hex', '4:00000000'
            ) -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
              -NoNewWindow -Wait -PassThru
            $output = (Get-Content $stdout -Raw) + (Get-Content $stderr -Raw)
            $code = $process.ExitCode
        } finally {
            Remove-Item $stdout, $stderr -Force -ErrorAction SilentlyContinue
        }
        Write-Host $output.Trim()
        return ($code -eq 0 -and $output -match [regex]::Escape($expected))
    }

    $ok1 = Invoke-Case '42' 'buf0[0]=42'
    $ok2 = Invoke-Case '7' 'buf0[0]=7'
    if ($ok1 -and $ok2) {
        Write-Host 'PASS (constant-offset LDC 42 and 7)'
        exit 0
    }
    Write-Host 'FAIL: constant-offset LDC result mismatch'
    exit 1
} finally {
    Remove-Item $lifted -Force -ErrorAction SilentlyContinue
}
