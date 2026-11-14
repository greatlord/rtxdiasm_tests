$ErrorActionPreference = 'Continue'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$cubin = Join-Path $root 'sm50_icmp.cubin'
if (-not (Test-Path $cubin)) {
    & (Join-Path $root 'build_cubin.bat') | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'FAIL: ptxas build failed'
        exit 1
    }
}

$repo = Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $root))
$exe = Join-Path $repo 'src\rtxdisam.cli\bin\Debug\net8.0\rtxdisam.dll'

function Invoke-Case([string]$inputHex, [string]$expected) {
    $stdout = [IO.Path]::GetTempFileName()
    $stderr = [IO.Path]::GetTempFileName()
    try {
        $process = Start-Process -FilePath 'dotnet' -ArgumentList @(
            $exe, '--run-cubin', $cubin, '--ptx-entry', 'k',
            '--run-buffer-hex', "4:$inputHex",
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

$ok1 = Invoke-Case '01000000' 'buf1[0]=2'
$ok2 = Invoke-Case 'ffffffff' 'buf1[0]=1'
if ($ok1 -and $ok2) {
    Write-Host 'PASS (ICMP positive and negative)'
    exit 0
}
Write-Host 'FAIL: ICMP result mismatch'
exit 1
