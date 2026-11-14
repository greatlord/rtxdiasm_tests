@echo off
setlocal
set "ROOT=%~dp0..\..\.."
dotnet "%ROOT%\src\rtxdisam.cli\bin\Debug\net8.0\rtxdisam.dll" -asmfile "%~dp0dynamic_ldc.sass" -arch sm_35 -o "%~dp0dynamic_ldc.cubin"
exit /b %ERRORLEVEL%
