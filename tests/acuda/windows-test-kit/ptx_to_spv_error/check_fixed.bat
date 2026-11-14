@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0check_fixed.ps1"
exit /b %ERRORLEVEL%
