$ErrorActionPreference = 'Continue'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$cubin = Join-Path $root 'sm50_func_shared.cubin'
if (-not (Test-Path $cubin)) {
    & (Join-Path $root 'build_cubin.bat') | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Host 'FAIL: ptxas build failed'; exit 1 }
}
$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $root))
$exe = Join-Path $repo 'src\rtxdisam.cli\bin\Debug\net8.0\rtxdisam.dll'
$lifted = Join-Path ([IO.Path]::GetTempPath()) ('rtxdisam_func_shared_' + [Guid]::NewGuid().ToString('N') + '.ptx')
try {
    & dotnet $exe --lift-ptx $cubin --ptx-entry k --ptx-param out:u64:0x140 --ptx-functions --ptx-output $lifted | Out-Null
    if ($LASTEXITCODE -ne 0) { Write-Host 'FAIL: SASS -> PTX lift failed'; exit 1 }
    $text = Get-Content $lifted -Raw
    if ($text -notmatch '\.func ' -or $text -notmatch 'st\.shared\.u32' -or $text -notmatch 'ld\.shared\.u32' -or $text -match 'L_inline_') {
        Write-Host 'FAIL: shared-memory function was not emitted'; exit 1
    }
    $stdout = [IO.Path]::GetTempFileName(); $stderr = [IO.Path]::GetTempFileName()
    try {
        $p = Start-Process -FilePath 'dotnet' -ArgumentList @($exe, '--run-ptx', $lifted, '--ptx-entry', 'k', '--run-buffer-hex', '4:00000000') -RedirectStandardOutput $stdout -RedirectStandardError $stderr -NoNewWindow -Wait -PassThru
        $out = (Get-Content $stdout -Raw) + (Get-Content $stderr -Raw); $code = $p.ExitCode
    } finally { Remove-Item $stdout, $stderr -Force -ErrorAction SilentlyContinue }
    Write-Host $out.Trim()
    if ($code -eq 0 -and $out -match 'validation_errors=0' -and $out -match 'buf0\[0\]=6') { Write-Host 'PASS (shared memory in .func)'; exit 0 }
    Write-Host 'FAIL: shared-memory .func mismatch'; exit 1
} finally { Remove-Item $lifted -Force -ErrorAction SilentlyContinue }
