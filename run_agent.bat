@echo off
echo Starting Instagram Agent...
cd /d "D:\Python\Programs\instagram agent"

:: Set Python to use UTF-8 for I/O to handle emojis in logs
set PYTHONUTF8=1

:: Run the agent
python main.py
pause
