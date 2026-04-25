# ============================================================
# Aura Project Orchestrator (Root)
# ============================================================
# Usage:
#   .\run.ps1 setup     → Complete one-time installation
#   .\run.ps1 dev       → Start BOTH frontend & backend
#   .\run.ps1 demo      → Start File Analysis Demo
#   .\run.ps1 backend   → Start FastAPI backend (port 8000)
#   .\run.ps1 frontend  → Start static frontend (port 3000)
#   .\run.ps1 clean     → Purge caches and temp files
# ============================================================

param(
    [Parameter(Position=0)]
    [ValidateSet("backend", "frontend", "dev", "install", "clean", "models", "setup")] # , "demo"
    [string]$Target = "dev"
)

$RootDir = Get-Location
$VenvPath = "$RootDir\.venv"
$PythonPath = if (Test-Path "$VenvPath\Scripts\python.exe") { "$VenvPath\Scripts\python.exe" } else { "python" }

function Start-Backend {
    Write-Host "--- Starting Backend ---" -ForegroundColor Green
    cd emotion-ai-backend
    # Ensure we use the root venv if backend doesn't have its own
    if (Test-Path ".\run.ps1") {
        .\run.ps1 dev
    } else {
        & $PythonPath -m uvicorn services.orchestrator.main:app --host 0.0.0.0 --port 8000 --reload
    }
}

function Start-Frontend {
    Write-Host "--- Starting Frontend (Port 3000) ---" -ForegroundColor Cyan
    cd aura-frontend
    # Inject env vars before serving
    if (Test-Path "scripts\inject-env.js") {
        node scripts\inject-env.js
    }
    npx -y serve . -p 3000
}

switch ($Target) {

    "setup" {
        Write-Host "--- Initializing Aura Environment ---" -ForegroundColor Yellow
        
        if (-not (Test-Path $VenvPath)) {
            Write-Host "Creating Virtual Environment..." -ForegroundColor Gray
            python -m venv .venv
        }

        Write-Host "Installing Backend Dependencies..." -ForegroundColor Gray
        & "$VenvPath\Scripts\pip" install -r emotion-ai-backend\requirements.txt
        
        Write-Host "Downloading AI Models..." -ForegroundColor Gray
        cd emotion-ai-backend
        .\run.ps1 models
        cd $RootDir

        Write-Host "Setup complete! Run '.\run.ps1 dev' to start." -ForegroundColor Green
    }

    "install" {
        Write-Host "--- Installing Dependencies ---" -ForegroundColor Cyan
        & "$VenvPath\Scripts\pip" install -r emotion-ai-backend\requirements.txt
    }

    "backend" {
        Start-Backend
    }

    "frontend" {
        Start-Frontend
    }

    "dev" {
        Write-Host "--- Launching Aura Development Environment ---" -ForegroundColor Yellow
        
        $BackendCmd = "cd emotion-ai-backend; if (Test-Path '..\.venv\Scripts\activate') { ..\.venv\Scripts\activate }; .\run.ps1 dev"
        $FrontendCmd = "cd aura-frontend; if (Test-Path 'scripts\inject-env.js') { node scripts\inject-env.js }; npx -y serve . -p 3000"

        Start-Process powershell -ArgumentList "-NoExit", "-Command", $BackendCmd
        Start-Process powershell -ArgumentList "-NoExit", "-Command", $FrontendCmd
        
        Write-Host "Dev environment ready. Check the new windows for logs." -ForegroundColor Green
    }

    # "demo" {
    #     Write-Host "--- Launching File Analysis Demo ---" -ForegroundColor Yellow
    #     Write-Host "1. Starting Backend in new window..." -ForegroundColor Gray
    #     Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd emotion-ai-backend; if (Test-Path '..\.venv\Scripts\activate') { ..\.venv\Scripts\activate }; .\run.ps1 dev"
    #     
    #     Write-Host "2. Starting Demo Frontend (Port 3001)..." -ForegroundColor Gray
    #     Start-Process powershell -ArgumentList "-NoExit", "-Command", "cd demo-frontend; npx -y serve . -p 3001"
    #     
    #     Write-Host "Demo ready at http://localhost:3001" -ForegroundColor Green
    # }

    "models" {
        cd emotion-ai-backend
        .\run.ps1 models
    }

    "clean" {
        Write-Host "--- Cleaning Project ---" -ForegroundColor Yellow
        
        # Remove Python caches
        Get-ChildItem -Path . -Recurse -Directory -Filter "__pycache__" | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem -Path . -Recurse -Filter "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue
        
        # Remove runtime artifacts
        if (Test-Path "emotion-ai-backend\recordings") {
            Remove-Item -Path "emotion-ai-backend\recordings\*.wav" -Force -ErrorAction SilentlyContinue
        }
        Remove-Item -Path "emotion-ai-backend\services\livekit_gateway\.livekit_usage.json" -ErrorAction SilentlyContinue
        
        Write-Host "Clean complete." -ForegroundColor Green
    }
}

