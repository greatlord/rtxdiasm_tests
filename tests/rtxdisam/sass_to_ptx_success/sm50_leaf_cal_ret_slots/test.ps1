$ErrorActionPreference = 'Continue'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$cubin = Join-Path $root 'sm50_leaf_cal_ret_slots.cubin'
if (-not (Test-Path $cubin)) {
    & (Join-Path $root 'build_cubin.bat') | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'FAIL: ptxas build failed'
        exit 1
    }
}

$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $root))
$exe = Join-Path $repo 'src\rtxdisam.cli\bin\Debug\net8.0\rtxdisam.dll'
$lifted = Join-Path ([IO.Path]::GetTempPath()) `
    ('rtxdisam_leaf_cal_ret_slots_' + [Guid]::NewGuid().ToString('N') + '.ptx')
try {
    & dotnet $exe --lift-ptx $cubin --ptx-entry k `
        --ptx-param x:u32:0x140 --ptx-param out:u64:0x148 `
        --ptx-output $lifted | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'FAIL: SASS -> PTX lift failed'
        exit 1
    }

    $liftedText = Get-Content $lifted -Raw
    if ($liftedText -notmatch 'st\.local\.u32 \[__callee_frame\+' -or
        $liftedText -notmatch 'ld\.local\.u32' ) {
        Write-Host 'FAIL: caller return slots were not lowered to local memory'
        exit 1
    }

    $stdout = [IO.Path]::GetTempFileName()
    $stderr = [IO.Path]::GetTempFileName()
    try {
        $process = Start-Process -FilePath 'dotnet' -ArgumentList @(
            $exe, '--run-ptx', $lifted, '--ptx-entry', 'k',
            '--run-scalar', '1',
            '--run-buffer-hex', '4:00000000'
        ) -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
          -NoNewWindow -Wait -PassThru
        $output = (Get-Content $stdout -Raw) + (Get-Content $stderr -Raw)
        $code = $process.ExitCode
    } finally {
        Remove-Item $stdout, $stderr -Force -ErrorAction SilentlyContinue
    }
    Write-Host $output.Trim()

    if ($code -eq 0 -and $output -match 'validation_errors=0' -and
        $output -match 'buf0\[0\]=230') {
        Write-Host 'PASS (leaf CAL return slots)'
        exit 0
    }
    Write-Host 'FAIL: return-slot leaf CAL mismatch'
    exit 1
} finally {
    Remove-Item $lifted -Force -ErrorAction SilentlyContinue
}
