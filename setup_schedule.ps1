# PowerShell script to schedule the Instagram Agent
$TaskName = "InstagramAgentDaily"
$ScriptPath = "D:\Python\Programs\instagram agent\run_agent.bat"

Write-Host "Creating Scheduled Task: $TaskName"

# 1. Action
$Action = New-ScheduledTaskAction -Execute $ScriptPath

# 2. Trigger
$Trigger = New-ScheduledTaskTrigger -Daily -At 7pm

# 3. Settings
$Settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

# 4. Register
try {
    Register-ScheduledTask -Action $Action -Trigger $Trigger -Settings $Settings -TaskName $TaskName -Description "Runs the Instagram Agent daily using topics.txt" -Force
    Write-Host "SUCCESS! Task scheduled."
} catch {
    Write-Host "ERROR: Could not create task."
    Write-Host $_ 
}