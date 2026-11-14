$ErrorActionPreference = 'Continue'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$cubin = Join-Path $root 'sm50_fadd32i.cubin'
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
    ('rtxdisam_fadd32i_' + [Guid]::NewGuid().ToString('N') + '.ptx')
try {
    & dotnet $exe --lift-ptx $cubin --ptx-entry k `
        --ptx-param p0:u64:0x140 --ptx-param p1:u64:0x148 `
        --ptx-output $lifted | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host 'FAIL: SASS -> PTX lift failed'
        exit 1
    }

    function Invoke-Case([string]$inputHex, [string]$expected) {
        $stdout = [IO.Path]::GetTempFileName()
        $stderr = [IO.Path]::GetTempFileName()
        try {
            $process = Start-Process -FilePath 'dotnet' -ArgumentList @(
                $exe, '--run-ptx', $lifted, '--ptx-entry', 'k',
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

    $ok1 = Invoke-Case '00000040' 'buf1[0]=1080033280'
    $ok2 = Invoke-Case '0000003f' 'buf1[0]=1073741824'
    if ($ok1 -and $ok2) {
        Write-Host 'PASS (FADD32I 2.0+1.5 and 0.5+1.5)'
        exit 0
    }
    Write-Host 'FAIL: FADD32I result mismatch'
    exit 1
} finally {
    Remove-Item $lifted -Force -ErrorAction SilentlyContinue
}
