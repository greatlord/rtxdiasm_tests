@echo off
setlocal
set "HERE=%~dp0"
if not exist "%HERE%..\common\test_nvidia.exe" call "%HERE%..\common\build_test_nvidia.bat"
"%HERE%..\common\test_nvidia.exe" "%HERE%trap.ptx" "%HERE%case.txt"
