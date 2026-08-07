@echo off
rem Modern Art AI - double-click to start. Opens your browser automatically.
rem Keep this window open while you play. Close it to quit.
chcp 65001 >nul
cd /d "%~dp0"
set PYTHONUTF8=1

where python >nul 2>nul
if %errorlevel%==0 (set PY=python) else (set PY=py)

%PY% -X utf8 -m modernart.server --open %*

if errorlevel 1 (
  echo.
  echo [!] Could not start. Read the message above.
  echo     If it says "No module named modernart", this file was moved out of its folder.
)
echo.
pause
