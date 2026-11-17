$ErrorActionPreference = 'Continue'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$cubin = Join-Path $root 'sm50_dynamic_ptr_ldc.cubin'
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
    ('rtxdisam_dynamic_ptr_ldc_' + [Guid]::NewGuid().ToString('N') + '.ptx')
try {
    & dotnet $exe --lift-ptx $cubin --ptx-entry k `
        --ptx-param p0:u32:0x140 `
        --ptx-param p1:u64:0x148 --ptx-param p2:u64:0x150 `
        --ptx-param p3:u64:0x158 --ptx-param p4:u64:0x160 `
        --ptx-output $lifted | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'FAIL: SASS -> PTX lift failed'
        exit 1
    }

    $liftedText = Get-Content $lifted -Raw
    if ($liftedText -notmatch 'ld\.shared\.u32 %r2' -or
        $liftedText -notmatch 'ld\.global\.u32') {
        Write-Host 'FAIL: pointer-table LDC was not lowered as expected'
        exit 1
    }

    function Invoke-Case([string]$selector, [string]$expected) {
        $stdout = [IO.Path]::GetTempFileName()
        $stderr = [IO.Path]::GetTempFileName()
        try {
            $process = Start-Process -FilePath 'dotnet' -ArgumentList @(
                $exe, '--run-ptx', $lifted, '--ptx-entry', 'k',
                '--run-scalar', $selector,
                '--run-buffer-hex', '4:0a000000',
                '--run-buffer-hex', '4:14000000',
                '--run-buffer-hex', '4:1e000000',
                '--run-buffer-hex', '4:00000000'
            ) -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
              -NoNewWindow -Wait -PassThru
            $output = (Get-Content $stdout -Raw) + (Get-Content $stderr -Raw)
            $code = $process.ExitCode
        } finally {
            Remove-Item $stdout, $stderr -Force -ErrorAction SilentlyContinue
        }
        Write-Host $output.Trim()
        return ($code -eq 0 -and $output -match 'validation_errors=0' -and
            $output -match [regex]::Escape($expected))
    }

    $ok1 = Invoke-Case '0' 'buf3[0]=10'
    $ok2 = Invoke-Case '1' 'buf3[0]=20'
    $ok3 = Invoke-Case '2' 'buf3[0]=30'
    $ok4 = Invoke-Case '3' 'buf3[0]=10'
    if ($ok1 -and $ok2 -and $ok3 -and $ok4) {
        Write-Host 'PASS (dynamic pointer LDC 0..2, clamped out-of-range)'
        exit 0
    }
    Write-Host 'FAIL: dynamic pointer LDC result mismatch'
    exit 1
} finally {
    Remove-Item $lifted -Force -ErrorAction SilentlyContinue
}
