@echo off
setlocal
set "CUDA102=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2"
"%CUDA102%\bin\ptxas.exe" -O0 -arch=sm_50 ^
    -o "%~dp0sm50_leaf_cal_structured.cubin" ^
    "%~dp0sm50_leaf_cal_structured.ptx"
exit /b %ERRORLEVEL%
