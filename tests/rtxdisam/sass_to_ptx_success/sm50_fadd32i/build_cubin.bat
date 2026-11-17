@echo off
setlocal
set "ROOT=%~dp0..\..\.."
dotnet "%ROOT%\src\rtxdisam.cli\bin\Debug\net8.0\rtxdisam.dll" -asmfile "%~dp0sm50_fadd32i.sass" -arch sm_50 -o "%~dp0sm50_fadd32i.cubin"
exit /b %ERRORLEVEL%
