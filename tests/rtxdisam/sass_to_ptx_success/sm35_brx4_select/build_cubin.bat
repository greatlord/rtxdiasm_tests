@echo off
setlocal
set "CUDA102=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2"
"%CUDA102%\bin\ptxas.exe" -arch=sm_35 ^
    -o "%~dp0sm35_brx4_select.cubin" ^
    "%~dp0sm35_brx4_select.ptx"
exit /b %ERRORLEVEL%
