@echo off
setlocal
set "HERE=%~dp0"
if not exist "%HERE%..\..\..\ptx_to_spv_error\common\test_nvidia.exe" call "%HERE%..\..\..\ptx_to_spv_error\common\build_test_nvidia.bat"
"%HERE%..\..\..\ptx_to_spv_error\common\test_nvidia.exe" "%HERE%bar_red_and_pred.ptx" "%HERE%nvidia_case.txt"
