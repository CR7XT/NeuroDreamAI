@echo off
REM NeuroDreamAI Launcher

set "CONDA_CMD=C:/Users/pruth/.cache/mine/Scripts/conda.exe run -p c:/Users/pruth/ai/NeuroDreamAI/.conda --no-capture-output python src/app.py"

:MENU
echo.
echo Select NeuroDreamAI mode:
echo 1. Offline/Template (no API)
echo 2. API (OpenAI GPT)
echo 3. Exit
set /p mode=Enter choice (1/2/3): 

if "%mode%"=="1" goto OFFLINE
if "%mode%"=="2" goto API
if "%mode%"=="3" exit

echo Invalid choice. Try again.
goto MENU

:OFFLINE
echo Running in Offline/Template mode...
set USE_API=
%CONDA_CMD%
goto END

:API
echo Running in API (OpenAI GPT) mode...
set USE_API=1
%CONDA_CMD%
goto END

:END
echo.
pause
