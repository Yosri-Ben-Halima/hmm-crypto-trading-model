# PowerShell script to create a daily scheduled task for predict.py

# --- CONFIGURATION ---
# IMPORTANT: Please update these variables to match your environment.

# 1. Full path to your Python executable (e.g., in your venv)
$pythonPath = "C:\Users\Lenovo\Desktop\Git Uploads\hmm-btc-trading-model\venv\Scripts\python.exe"

# 2. Full path to the directory containing your project
$projectPath = "C:\Users\Lenovo\Desktop\Git Uploads\hmm-btc-trading-model"

# 3. Full path to your predict.py script
$scriptPath = "$projectPath\predict.py"

# 4. Name for the scheduled task
$taskName = "Daily_HMM_BTC_Signal"

# 5. Time to run the task every day (24-hour format)

$runTime = "17:15"

# --- SCRIPT --- 

# Action to be performed by the task
$action = New-ScheduledTaskAction -Execute $pythonPath -Argument $scriptPath -WorkingDirectory $projectPath

# Trigger to run the task daily at the specified time
$trigger = New-ScheduledTaskTrigger -Daily -At $runTime

# Settings for the scheduled task
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable

# Principal for the task (who runs it)
$principal = New-ScheduledTaskPrincipal -UserId (Get-CimInstance -ClassName Win32_ComputerSystem).UserName -LogonType Interactive

# Register the scheduled task
Write-Host "Creating scheduled task: $taskName"
Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Force

Write-Host "Task created successfully!"
Write-Host "It will run daily at $runTime."
Write-Host "To view the task, open Task Scheduler and look for '$taskName'."
