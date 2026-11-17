@echo off
setlocal
set "CUDA102=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2"
"%CUDA102%\bin\ptxas.exe" -arch=sm_50 ^
    -o "%~dp0sm50_icmp.cubin" ^
    "%~dp0sm50_icmp.ptx"
exit /b %ERRORLEVEL%
