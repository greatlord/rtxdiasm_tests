$ErrorActionPreference = 'Continue'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$cubin = Join-Path $root 'sm50_func_ptr.cubin'
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
    ('rtxdisam_func_ptr_' + [Guid]::NewGuid().ToString('N') + '.ptx')
try {
    & dotnet $exe --lift-ptx $cubin --ptx-entry k `
        --ptx-param pin:u64:0x140 --ptx-param pout:u64:0x148 `
        --ptx-param out:u64:0x150 --ptx-functions `
        --ptx-output $lifted | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'FAIL: SASS -> PTX lift failed'
        exit 1
    }

    $liftedText = Get-Content $lifted -Raw
    if ($liftedText -notmatch '\.func \(\.param \.u32 __ret\)' -or
        $liftedText -notmatch '\.param \.u64 __a0, \.param \.u64 __a1' -or
        $liftedText -notmatch 'ld\.global\.u32' -or
        $liftedText -notmatch 'st\.global\.u32' -or
        $liftedText -match 'L_inline_') {
        Write-Host 'FAIL: pointer function call was not emitted as .func/call'
        exit 1
    }

    $stdout = [IO.Path]::GetTempFileName()
    $stderr = [IO.Path]::GetTempFileName()
    try {
        $process = Start-Process -FilePath 'dotnet' -ArgumentList @(
            $exe, '--run-ptx', $lifted, '--ptx-entry', 'k',
            '--run-buffer-hex', '4:29000000',
            '--run-buffer-hex', '4:00000000',
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
        $output -match 'buf1\[0\]=42' -and
        $output -match 'buf2\[0\]=43') {
        Write-Host 'PASS (function pointer args and global memory)'
        exit 0
    }
    Write-Host 'FAIL: pointer .func/call result mismatch'
    exit 1
} finally {
    Remove-Item $lifted -Force -ErrorAction SilentlyContinue
}
