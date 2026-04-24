# Makefile — Real-Time Emotion Intelligence Backend
# ====================================================
# Targets:
#   make run       - Start orchestrator with real API keys
#   make dev       - Start in development mode (hot reload)
#   make mock      - Start in mock/offline mode (no internet required)
#   make dashboard - Open the Rich terminal UI
#   make models    - Download ONNX models
#   make install   - Install all Python dependencies
#   make clean     - Remove .pyc, logs, recordings
#   make schema    - Print Supabase SQL schema to stdout

PYTHON  := python
UVICORN := uvicorn
PORT    := 8000

.PHONY: run dev mock dashboard models install clean schema

# ── Real mode (needs .env with API keys) ─────────────────────────────────────
run:
	@echo "🚀 Starting Orchestrator on port $(PORT)..."
	$(UVICORN) services.orchestrator.main:app \
		--host 0.0.0.0 --port $(PORT) \
		--log-level info

# ── Dev mode (auto-reload on file changes) ────────────────────────────────────
dev:
	@echo "🔧 Starting in development mode (hot reload)..."
	$(UVICORN) services.orchestrator.main:app \
		--host 0.0.0.0 --port $(PORT) \
		--reload --log-level debug

# ── Mock / offline mode ───────────────────────────────────────────────────────
mock:
	@echo "🎭 Starting in MOCK mode (no internet required)..."
	MOCK_APIS=true $(UVICORN) services.orchestrator.main:app \
		--host 0.0.0.0 --port $(PORT) \
		--log-level info

# ── Rich terminal dashboard ───────────────────────────────────────────────────
dashboard:
	@echo "📊 Opening TUI dashboard..."
	$(PYTHON) -m services.dashboard.tui

# ── Download ONNX models ──────────────────────────────────────────────────────
models:
	@echo "📦 Downloading ONNX models..."
	@bash scripts/download_models.sh

# ── Install all dependencies ──────────────────────────────────────────────────
install:
	@echo "📦 Installing Python dependencies..."
	pip install -r requirements.txt

# ── Clean up ──────────────────────────────────────────────────────────────────
clean:
	@echo "🧹 Cleaning up..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	rm -f services/livekit_gateway/.livekit_usage.json
	rm -f *.log *.log.*
	@echo "✅ Clean complete."

# ── Print Supabase SQL schema ─────────────────────────────────────────────────
schema:
	@$(PYTHON) -c "from services.orchestrator.supabase_client import create_schema_sql; print(create_schema_sql())"
