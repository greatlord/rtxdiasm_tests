@echo off
setlocal
set "CUDA102=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2"
"%CUDA102%\bin\ptxas.exe" -O0 -arch=sm_35 ^
    -o "%~dp0sm35_leaf_cal_branchy.cubin" ^
    "%~dp0sm35_leaf_cal_branchy.ptx"
exit /b %ERRORLEVEL%
