@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 goto directory_failed

echo GeoVision AI - Windows launcher
echo.
if not exist "app.py" goto files_missing
if not exist "requirements.txt" goto files_missing
if not exist "scripts\check_runtime.py" goto files_missing
if exist ".venv\Scripts\python.exe" goto check_environment
if exist ".venv" goto incomplete_environment

echo Looking for Python 3.12 or 3.11...
py -3.12 -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) and sys.maxsize > 2**32 else 1)" >nul 2>&1
if not errorlevel 1 goto create_py312
py -3.11 -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 11) and sys.maxsize > 2**32 else 1)" >nul 2>&1
if not errorlevel 1 goto create_py311
python -c "import sys; sys.exit(0 if sys.version_info[:2] in ((3, 11), (3, 12)) and sys.maxsize > 2**32 else 1)" >nul 2>&1
if not errorlevel 1 goto create_python
echo ERROR: Python 3.11 or 3.12 could not be found.
echo Install 64-bit Python 3.12 from python.org with the Python launcher enabled,
echo then run this file again.
goto failed

:create_py312
echo Creating .venv with Python 3.12...
py -3.12 -m venv ".venv"
if errorlevel 1 goto creation_failed
goto check_environment

:create_py311
echo Creating .venv with Python 3.11...
py -3.11 -m venv ".venv"
if errorlevel 1 goto creation_failed
goto check_environment

:create_python
echo Creating .venv with the compatible Python on PATH...
python -m venv ".venv"
if errorlevel 1 goto creation_failed
goto check_environment

:check_environment
echo Checking the project Python environment...
".venv\Scripts\python.exe" -c "import sys; print('Python ' + sys.version.split()[0]); sys.exit(0 if sys.version_info[:2] in ((3, 11), (3, 12)) and sys.maxsize > 2**32 else 1)"
if errorlevel 1 goto incompatible_environment
".venv\Scripts\python.exe" "scripts\check_runtime.py"
if not errorlevel 1 goto launch

echo.
echo Dependencies need setup or repair. Installing requirements into .venv...
echo This step needs an internet connection on first setup. Errors will appear below.
".venv\Scripts\python.exe" -m pip --version
if not errorlevel 1 goto install_requirements
".venv\Scripts\python.exe" -m ensurepip --upgrade
if errorlevel 1 goto dependency_failed

:install_requirements
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r "requirements.txt"
if errorlevel 1 goto dependency_failed
".venv\Scripts\python.exe" "scripts\check_runtime.py"
if errorlevel 1 goto dependency_failed

:launch
echo.
echo Starting GeoVision AI. Open the Local URL printed by Streamlit below.
echo KEEP THIS WINDOW OPEN while using the dashboard.
echo Closing this window stops the app and causes a browser connection error.
echo Press Ctrl+C here when you want to stop the app.
echo.
".venv\Scripts\python.exe" -m streamlit run "app.py"
if errorlevel 1 goto server_failed
exit /b 0

:directory_failed
echo ERROR: Cannot open the GeoVision project directory.
goto failed

:files_missing
echo ERROR: Project files are missing. Extract the entire ZIP before launching.
echo Keep run_windows.bat beside app.py, requirements.txt, and the scripts folder.
goto failed

:incomplete_environment
echo ERROR: An existing .venv was found, but its Python executable is missing.
echo It has been preserved. Repair it, or rename it before creating a new environment.
goto failed

:creation_failed
echo ERROR: Python could not create the project environment. See the error above.
echo Any partially created .venv has been preserved for inspection.
goto failed

:incompatible_environment
echo ERROR: The existing .venv is broken or does not use 64-bit Python 3.11 or 3.12.
echo It has been preserved. Repair it, or rename it before running this launcher again.
goto failed

:dependency_failed
echo ERROR: Dependency setup failed or the installed packages still cannot load.
echo Check the error above and your internet connection. The existing .venv is preserved.
goto failed

:server_failed
echo ERROR: Streamlit stopped before a normal shutdown. Read its error above.
echo If the port is already in use, use the existing app or stop it before retrying.
goto failed

:failed
echo.
echo GeoVision AI did not start successfully. Keep or copy the error message above.
pause
exit /b 1
