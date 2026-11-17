@echo off
setlocal
set "ROOT=%~dp0..\..\.."
dotnet "%ROOT%\src\rtxdisam.cli\bin\Debug\net8.0\rtxdisam.dll" -asmfile "%~dp0sm50_dynamic_ldc.sass" -arch sm_50 -o "%~dp0sm50_dynamic_ldc.cubin"
exit /b %ERRORLEVEL%
