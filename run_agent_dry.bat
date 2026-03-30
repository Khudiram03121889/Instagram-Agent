@echo off
echo Starting Instagram Agent in dry-run mode...
cd /d "D:\Python\Programs\instagram agent"

:: Set Python to use UTF-8 for I/O to handle emojis in logs
set PYTHONUTF8=1

python main.py --flow-dry-run
pause
