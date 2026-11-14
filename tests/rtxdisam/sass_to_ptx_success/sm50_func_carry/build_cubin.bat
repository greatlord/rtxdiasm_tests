@echo off
setlocal
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin\ptxas.exe" -O0 -arch=sm_50 -o "%~dp0sm50_func_carry.cubin" "%~dp0sm50_func_carry.ptx"
exit /b %ERRORLEVEL%
