$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$cubin = Join-Path $root 'sm50_func_carry.cubin'
if (-not (Test-Path $cubin)) {
    & (Join-Path $root 'build_cubin.bat') | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Host 'FAIL: ptxas build failed'; exit 1 }
}
$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $root))
$exe = Join-Path $repo 'src\rtxdisam.cli\bin\Debug\net8.0\rtxdisam.dll'
$lifted = Join-Path ([IO.Path]::GetTempPath()) ('rtxdisam_func_carry_' + [Guid]::NewGuid().ToString('N') + '.ptx')
try {
    & dotnet $exe --lift-ptx $cubin --ptx-entry k --ptx-param a0:u32:0x140 --ptx-param a1:u32:0x144 --ptx-param b0:u32:0x148 --ptx-param b1:u32:0x14c --ptx-param out:u64:0x150 --ptx-functions --ptx-output $lifted | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Host 'FAIL: SASS -> PTX lift failed'; exit 1 }
    $text = Get-Content $lifted -Raw
    if ($text -notmatch '\.func ' -or $text -notmatch 'add\.u64' -or $text -match 'L_inline_') {
        Write-Host 'FAIL: carry-chain function was not emitted'; exit 1
    }
    $stdout = [IO.Path]::GetTempFileName(); $stderr = [IO.Path]::GetTempFileName()
    try {
        $p = Start-Process -FilePath 'dotnet' -ArgumentList @($exe, '--run-ptx', $lifted, '--ptx-entry', 'k', '--run-scalar', '1', '--run-scalar', '0', '--run-scalar', '2', '--run-scalar', '0', '--run-buffer-hex', '8:0000000000000000') -RedirectStandardOutput $stdout -RedirectStandardError $stderr -NoNewWindow -Wait -PassThru
        $out = (Get-Content $stdout -Raw) + (Get-Content $stderr -Raw); $code = $p.ExitCode
    } finally { Remove-Item $stdout, $stderr -Force -ErrorAction SilentlyContinue }
    Write-Host $out.Trim()
    if ($code -eq 0 -and $out -match 'validation_errors=0' -and $out -match 'buf0\[0\]=3') { Write-Host 'PASS (carry chain in .func)'; exit 0 }
    Write-Host 'FAIL: carry-chain .func mismatch'; exit 1
} finally { Remove-Item $lifted -Force -ErrorAction SilentlyContinue }
