Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "   Attendance AI - PowerShell Launcher" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/2] Starting FastAPI Backend (Port 8000)..." -ForegroundColor Yellow
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectRoot\backend'; `$host.ui.RawUI.WindowTitle = 'Backend Server'; `$py = 'python'; if (Test-Path '$ProjectRoot\venv\Scripts\python.exe') { `$py = '$ProjectRoot\venv\Scripts\python.exe' } elseif (Test-Path '$ProjectRoot\.cuda\Scripts\python.exe') { `$py = '$ProjectRoot\.cuda\Scripts\python.exe' }; Write-Host `"Using Python: `$py`" -ForegroundColor Cyan; & `$py -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload"

Write-Host "[2/2] Starting React Frontend (Port 5173)..." -ForegroundColor Yellow
Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd '$ProjectRoot\frontend'; `$host.ui.RawUI.WindowTitle = 'Frontend Server'; npm install; npm run dev"

Write-Host ""
Write-Host "All services are launching in new windows!" -ForegroundColor Green
Write-Host "You can now close this launcher window." -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Cyan
