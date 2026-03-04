@echo off
echo Starting Instagram Agent...
cd /d "D:\Python\Programs\instagram agent"

:: Set Python to use UTF-8 for I/O to handle emojis in logs
set PYTHONUTF8=1

:: Run the agent
python main.py >> "daily_log.txt" 2>&1

:: Optional: keep window open for a few seconds if user is watching
:: timeout /t 5
