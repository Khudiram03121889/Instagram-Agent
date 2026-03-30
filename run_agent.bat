@echo off
echo Starting Instagram Agent...
cd /d "D:\Python\Programs\instagram agent"

:: Set Python to use UTF-8 for I/O to handle emojis in logs
set PYTHONUTF8=1

if /I "%~1"=="dry" goto dryrun

echo Running in Flow LIVE mode.
echo This will open Google Flow in Chrome after local validation passes.
echo Use: run_agent.bat dry
echo for a no-browser validation run.

:: Run the agent
python main.py --flow-live
goto end

:dryrun
echo Running in Flow DRY-RUN mode.
echo This validates script, prompts, and preflight checks without opening Flow.
python main.py --flow-dry-run

:end
pause
