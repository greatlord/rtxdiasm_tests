@echo off
setlocal
set "CUDA102=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2"
"%CUDA102%\bin\ptxas.exe" -arch=sm_30 ^
    -o "%~dp0sm30_shared_add.cubin" ^
    "%~dp0sm30_shared_add.ptx"
exit /b %ERRORLEVEL%
