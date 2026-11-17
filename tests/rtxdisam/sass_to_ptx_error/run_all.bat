@echo off
setlocal enabledelayedexpansion
set "ROOT=%~dp0"
set FAIL=0
for /d %%D in ("%ROOT%*") do (
    if exist "%%D\run.bat" (
        echo === %%~nxD ===
        call "%%D\run.bat"
        if errorlevel 1 set FAIL=1
    )
)
if %FAIL% neq 0 (
    echo Some lifter error cases failed.
    exit /b 1
)
echo All lifter error cases passed.
exit /b 0
