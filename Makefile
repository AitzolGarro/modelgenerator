.PHONY: setup run dev stop build clean help

ROOT_DIR := $(shell pwd)
VENV := $(ROOT_DIR)/backend/venv
PYTHON := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

setup: ## First-time setup (installs everything)
	@bash setup.sh

run: ## Start the app (single process, http://localhost:8000)
	@bash start.sh

dev: ## Start in dev mode (frontend hot-reload on :3000, backend on :8000)
	@echo "Starting backend + worker on :8000..."
	@bash -c 'source $(VENV)/bin/activate && cd backend && uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload &'
	@echo "Starting frontend dev server on :3000..."
	@cd frontend && npm run dev

build: ## Rebuild the frontend
	@cd frontend && npm run build
	@echo "Frontend built to frontend/out/"

clean: ## Remove generated files (storage, venv, node_modules, builds)
	rm -rf storage/images/* storage/models/* storage/exports/* storage/*.db
	rm -rf frontend/out frontend/.next
	@echo "Cleaned generated files. Run 'make clean-all' to also remove deps."

clean-all: clean ## Remove everything including dependencies
	rm -rf backend/venv frontend/node_modules
	@echo "All clean. Run 'make setup' to reinstall."
