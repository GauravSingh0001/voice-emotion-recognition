# ============================================================
# run.ps1 — Windows PowerShell equivalent of the Makefile
# ============================================================
# Usage (from emotion-ai-backend/ directory):
#   .\run.ps1 run       → start orchestrator
#   .\run.ps1 dev       → start with hot-reload
#   .\run.ps1 mock      → offline / no API keys needed
#   .\run.ps1 dashboard → open Rich terminal UI
#   .\run.ps1 install   → install all dependencies
#   .\run.ps1 schema    → print Supabase SQL to stdout
#   .\run.ps1 clean     → remove __pycache__ etc.
# ============================================================

param(
    [Parameter(Position=0)]
    [ValidateSet("run","dev","dashboard","install","schema","clean","models")] # , "mock"
    [string]$Target = "run"
)

$Port = 8000

switch ($Target) {

    "install" {
        Write-Host "Installing CPU-only PyTorch first..." -ForegroundColor Cyan
        pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
        Write-Host "Installing remaining dependencies..." -ForegroundColor Cyan
        pip install python-dotenv pydantic livekit livekit-api librosa noisereduce numpy onnxruntime soundfile aiohttp groq supabase fastapi "uvicorn[standard]" rich
        Write-Host "Done!" -ForegroundColor Green
    }

    "run" {
        Write-Host "Starting Orchestrator on port $Port ..." -ForegroundColor Green
        python -m uvicorn services.orchestrator.main:app --host 0.0.0.0 --port $Port --log-level info
    }

    "dev" {
        Write-Host "Starting in development mode (hot reload)..." -ForegroundColor Yellow
        python -m uvicorn services.orchestrator.main:app --host 0.0.0.0 --port $Port --reload --log-level debug
    }

    # "mock" {
    #     Write-Host "Starting in MOCK mode (no internet required)..." -ForegroundColor Magenta
    #     $env:MOCK_APIS = "true"
    #     python -m uvicorn services.orchestrator.main:app --host 0.0.0.0 --port $Port --log-level info
    # }

    "dashboard" {
        Write-Host "Opening TUI dashboard..." -ForegroundColor Cyan
        python -m services.dashboard.tui
    }

    "schema" {
        python -c "from services.orchestrator.supabase_client import create_schema_sql; print(create_schema_sql())"
    }

    "models" {
        Write-Host "Downloading ONNX models..." -ForegroundColor Cyan
        $modelsDir = "models"
        if (-not (Test-Path $modelsDir)) { New-Item -ItemType Directory -Path $modelsDir | Out-Null }

        $senseVoicePath = "$modelsDir\sensevoice_small.onnx"
        if (-not (Test-Path $senseVoicePath)) {
            Write-Host "Downloading SenseVoice-Small ONNX (~150MB)..." -ForegroundColor Yellow
            $url = "https://huggingface.co/csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17/resolve/main/model.int8.onnx"
            Invoke-WebRequest -Uri $url -OutFile $senseVoicePath -UseBasicParsing
            Write-Host "SenseVoice downloaded." -ForegroundColor Green
        } else {
            Write-Host "SenseVoice-Small already exists, skipping." -ForegroundColor Gray
        }

        $sileroPath = "$modelsDir\silero_vad.onnx"
        if (-not (Test-Path $sileroPath)) {
            Write-Host "Downloading Silero VAD ONNX (~2MB)..." -ForegroundColor Yellow
            $url = "https://raw.githubusercontent.com/snakers4/silero-vad/master/src/silero_vad/data/silero_vad.onnx"
            Invoke-WebRequest -Uri $url -OutFile $sileroPath -UseBasicParsing
            Write-Host "Silero VAD downloaded." -ForegroundColor Green
        } else {
            Write-Host "Silero VAD already exists, skipping." -ForegroundColor Gray
        }
        Write-Host "All models ready in .\models\" -ForegroundColor Green
    }

    "clean" {
        Write-Host "Cleaning up..." -ForegroundColor Cyan
        Get-ChildItem -Recurse -Filter "__pycache__" -Directory | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue
        Get-ChildItem -Recurse -Filter "*.pyc" | Remove-Item -Force -ErrorAction SilentlyContinue
        Remove-Item -Path "services\livekit_gateway\.livekit_usage.json" -ErrorAction SilentlyContinue
        Write-Host "Clean complete." -ForegroundColor Green
    }
}
