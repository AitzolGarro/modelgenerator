.PHONY: help run stop build logs clean setup run-local dev

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# ── Docker (recommended for Bazzite/immutable OS) ───────────

run: ## Start with Docker (recommended)
	@bash start.sh

stop: ## Stop Docker containers
	@docker compose down

build: ## Rebuild Docker image
	@docker compose build

logs: ## View Docker logs
	@docker compose logs -f

# ── Local mode (needs Python headers + CUDA toolkit) ────────

setup: ## First-time setup for local mode
	@bash setup.sh

run-local: ## Start in local venv mode (no Docker)
	@bash start.sh --local

dev: ## Dev mode: backend on :8000 (reload), frontend on :3000 (hot-reload)
	@bash -c 'source backend/venv/bin/activate && cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &'
	@cd frontend && npm run dev

# ── Cleanup ──────────────────────────────────────────────────

clean: ## Remove generated files
	rm -rf storage/images/* storage/models/* storage/exports/* storage/*.db
	rm -rf frontend/out frontend/.next
	@echo "Cleaned. Run 'make clean-all' for full reset."

clean-all: clean ## Remove everything (deps, images, volumes)
	rm -rf backend/venv frontend/node_modules
	docker compose down -v 2>/dev/null || true
	@echo "Full clean. Run 'make run' to rebuild."
