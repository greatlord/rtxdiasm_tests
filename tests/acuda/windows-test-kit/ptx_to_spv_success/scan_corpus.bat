@echo off
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0scan_corpus.ps1"
exit /b %ERRORLEVEL%
