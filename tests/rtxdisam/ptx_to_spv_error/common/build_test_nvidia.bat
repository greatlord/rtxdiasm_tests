@echo off
setlocal
set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.4"
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
pushd "%~dp0"
cl /nologo /EHsc /std:c++17 /I "%CUDA_ROOT%\include" test_nvidia.cpp ^
    /link /LIBPATH:"%CUDA_ROOT%\lib\x64" cuda.lib
popd
