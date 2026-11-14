$ErrorActionPreference = 'Continue'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$cubin = Join-Path $root 'sm30_shared_add.cubin'
if (-not (Test-Path $cubin)) {
    & (Join-Path $root 'build_cubin.bat') | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'FAIL: ptxas build failed'
        exit 1
    }
}

$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $root))
$exe = Join-Path $repo 'src\rtxdisam.cli\bin\Debug\net8.0\rtxdisam.dll'

function Invoke-Case([string]$block, [string]$expectedOld,
    [string]$expectedCurrent) {
    $stdout = [IO.Path]::GetTempFileName()
    $stderr = [IO.Path]::GetTempFileName()
    try {
        $process = Start-Process -FilePath 'dotnet' -ArgumentList @(
            $exe, '--run-cubin', $cubin, '--ptx-entry', 'k',
            '--run-block', $block,
            '--run-buffer-hex', '8:0000000000000000'
        ) -RedirectStandardOutput $stdout -RedirectStandardError $stderr `
          -NoNewWindow -Wait -PassThru
        $output = (Get-Content $stdout -Raw) + (Get-Content $stderr -Raw)
        $code = $process.ExitCode
    } finally {
        Remove-Item $stdout, $stderr -Force -ErrorAction SilentlyContinue
    }
    Write-Host $output.Trim()
    $oldOk = [string]::IsNullOrEmpty($expectedOld) -or
        ($output -match [regex]::Escape($expectedOld))
    return ($code -eq 0 -and $oldOk -and
        $output -match [regex]::Escape($expectedCurrent))
}

$ok1 = Invoke-Case '1' 'buf0[0]=0' 'buf0[1]=1'
$ok2 = Invoke-Case '4' '' 'buf0[1]=4'
if ($ok1 -and $ok2) {
    Write-Host 'PASS (LL/SC shared atomic add, one and four threads)'
    exit 0
}
Write-Host 'FAIL: shared atomic result mismatch'
exit 1
