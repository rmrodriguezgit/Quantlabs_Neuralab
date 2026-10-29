@echo off
setlocal
cd /d "%~dp0"
where node >nul 2>nul
if errorlevel 1 (echo Instala Node.js 22 LTS o superior. & exit /b 1)
where npm >nul 2>nul
if errorlevel 1 (echo npm no disponible. & exit /b 1)
node -e "if(Number(process.versions.node.split('.')[0])<22) process.exit(1)"
if errorlevel 1 (echo Se necesita Node.js 22+. & exit /b 1)
if exist .venv\Scripts\python.exe goto dependencies
if defined FORECAST_PYTHON goto custompython
py -3.12 -m venv .venv
if not errorlevel 1 goto dependencies
py -3.11 -m venv .venv
if not errorlevel 1 goto dependencies
py -3.10 -m venv .venv
if errorlevel 1 (echo Instala Python 3.10, 3.11 o 3.12 y Python Launcher. & exit /b 1)
goto dependencies
:custompython
"%FORECAST_PYTHON%" -c "import sys; assert (3,10)<=sys.version_info[:2]<=(3,12)"
if errorlevel 1 exit /b 1
"%FORECAST_PYTHON%" -m venv .venv
if errorlevel 1 exit /b 1
:dependencies
.venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 exit /b 1
set REQ=python\requirements.txt
if "%FORECAST_ADVANCED%"=="1" set REQ=python\requirements-advanced.txt
.venv\Scripts\python.exe -m pip install -r "%REQ%"
if errorlevel 1 exit /b 1
if exist package-lock.json (call npm ci) else (call npm install)
if errorlevel 1 exit /b 1
set FORECAST_PYTHON=%CD%\.venv\Scripts\python.exe
call npm run dev
exit /b %errorlevel%
