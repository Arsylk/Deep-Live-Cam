@echo off
setlocal
cd /d "%~dp0"

if not exist "runtime\network-live" mkdir "runtime\network-live"
set "PYTHON=%CD%\venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo [%DATE% %TIME%] Missing project interpreter: %PYTHON% >> "runtime\network-live\service.log"
  exit /b 3
)

rem Each client has a fixed slot.  Selection is changed through the native
rem manager/API; this supervisor never needs endpoint arguments or a restart.
:loop
echo [%DATE% %TIME%] Starting five-slot network router >> "runtime\network-live\service.log"
"%PYTHON%" run_network.py --android-host 192.168.1.12 --arch-host 192.168.1.11 --bitrate 10M --swapper-model inswapper-128 >> "runtime\network-live\service.log" 2>&1
set EXIT_CODE=%ERRORLEVEL%
echo [%DATE% %TIME%] Network router exited with %EXIT_CODE%; restarting in 2s >> "runtime\network-live\service.log"
timeout /t 2 /nobreak >nul
goto loop
