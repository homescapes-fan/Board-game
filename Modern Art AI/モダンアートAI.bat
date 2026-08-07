@echo off
rem ダブルクリックで起動する。ブラウザが自動で開く。
rem 日本語が化けないよう、コンソールとPythonの両方をUTF-8にする。
chcp 65001 >nul
cd /d "%~dp0"

set PYTHONUTF8=1
where python >nul 2>nul && (set PY=python) || (set PY=py)

echo モダンアート AI アドバイザー を起動しています...
start "" http://localhost:8765/
%PY% -X utf8 -m modernart.server %*

echo.
echo 終了しました。このウィンドウは閉じて構いません。
pause
