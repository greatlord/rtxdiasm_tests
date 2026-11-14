@echo off
setlocal
"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.2\bin\ptxas.exe" -O0 -arch=sm_50 -c -o "%~dp0reloc_jcal.o" "%~dp0reloc_jcal.ptx"
exit /b %ERRORLEVEL%
