$ErrorActionPreference = 'Continue'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$cubin = Join-Path $root 'sm50_func_ret2.cubin'
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
    ('rtxdisam_func_ret2_' + [Guid]::NewGuid().ToString('N') + '.ptx')
try {
    & dotnet $exe --lift-ptx $cubin --ptx-entry k `
        --ptx-param a:u32:0x140 --ptx-param out:u64:0x148 `
        --ptx-functions --ptx-output $lifted | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'FAIL: SASS -> PTX lift failed'
        exit 1
    }

    $liftedText = Get-Content $lifted -Raw
    if ($liftedText -notmatch '\.func \(\.param \.u64 __ret\)' -or
        $liftedText -notmatch 'st\.param\.u64 \[__ret\]' -or
        $liftedText -notmatch 'mov\.b64 %rdt0, \{%r0, %r2\}' -or
        $liftedText -match 'L_inline_') {
        Write-Host 'FAIL: two-register return was not packed into a .func return'
        exit 1
    }

    $stdout = [IO.Path]::GetTempFileName()
    $stderr = [IO.Path]::GetTempFileName()
    try {
        $process = Start-Process -FilePath 'dotnet' -ArgumentList @(
            $exe, '--run-ptx', $lifted, '--ptx-entry', 'k',
            '--run-scalar', '3',
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
        $output -match 'buf0\[0\]=7') {
        Write-Host 'PASS (two-register function return)'
        exit 0
    }
    Write-Host 'FAIL: two-register .func return mismatch'
    exit 1
} finally {
    Remove-Item $lifted -Force -ErrorAction SilentlyContinue
}
