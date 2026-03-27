@echo off
echo Starting Instagram Agent...
cd /d "D:\Python\Programs\instagram agent"

:: Set Python to use UTF-8 for I/O to handle emojis in logs
set PYTHONUTF8=1

echo Running in Flow dry-run mode by default.
echo Use: python main.py --flow-live
echo to launch Google Flow and spend credits after local validation passes.

:: Run the agent
python main.py
pause
