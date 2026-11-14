@echo off
setlocal
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin\ptxas.exe" -O0 -arch=sm_50 -o "%~dp0sm50_leaf_cal_nested.cubin" "%~dp0sm50_leaf_cal_nested.ptx"
exit /b %ERRORLEVEL%
