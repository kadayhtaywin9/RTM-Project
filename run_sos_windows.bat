@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 goto failed
if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Run run_windows.bat first to set up .venv.
    goto failed
)
".venv\Scripts\python.exe" "scripts\check_runtime.py"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" -m uvicorn sos_service.app:app --host 0.0.0.0 --port 8000
if errorlevel 1 goto failed
exit /b 0

:failed
echo SOS did not start. Check the error above and README.md.
pause
exit /b 1
