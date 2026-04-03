"""
FastAPI application entry point.

Single-process mode: API + Worker + Static frontend, all in one.
"""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.db.database import init_db
from app.api import jobs, files, health
from app.workers.background import start_worker, stop_worker

settings = get_settings()
logger = get_logger(__name__)

# Path to the built frontend (next export or standalone)
FRONTEND_BUILD_DIR = Path(__file__).resolve().parent.parent.parent / "frontend" / "out"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events."""
    # Startup
    setup_logging(settings.DEBUG)
    settings.ensure_dirs()
    init_db()

    # Start background worker in a daemon thread
    logger.info("Starting embedded worker...")
    start_worker()

    yield

    # Shutdown
    logger.info("Shutting down embedded worker...")
    stop_worker()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Local 3D model generation from text prompts",
    lifespan=lifespan,
)

# CORS — needed for dev mode (next dev on :3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes — all under /api/v1
app.include_router(health.router, prefix="/api/v1")
app.include_router(jobs.router, prefix="/api/v1")
app.include_router(files.router, prefix="/api/v1")


# Serve the built frontend as static files if available
if FRONTEND_BUILD_DIR.exists() and (FRONTEND_BUILD_DIR / "index.html").exists():
    logger.info(f"Serving frontend from {FRONTEND_BUILD_DIR}")

    # Mount static assets (JS, CSS, images)
    app.mount(
        "/_next",
        StaticFiles(directory=str(FRONTEND_BUILD_DIR / "_next")),
        name="next-assets",
    )

    # Catch-all: serve index.html for all non-API routes (SPA fallback)
    from fastapi.responses import FileResponse

    @app.get("/{full_path:path}")
    async def serve_frontend(full_path: str):
        """Serve the frontend SPA. Falls back to index.html for client-side routing."""
        # Try to serve the exact file first (e.g. /jobs.html, /favicon.ico)
        file_path = FRONTEND_BUILD_DIR / full_path
        if file_path.is_file():
            return FileResponse(str(file_path))

        # Try with .html extension (Next.js static export convention)
        html_path = FRONTEND_BUILD_DIR / f"{full_path}.html"
        if html_path.is_file():
            return FileResponse(str(html_path))

        # Try as directory with index.html
        index_path = FRONTEND_BUILD_DIR / full_path / "index.html"
        if index_path.is_file():
            return FileResponse(str(index_path))

        # Fallback to root index.html (SPA routing)
        return FileResponse(str(FRONTEND_BUILD_DIR / "index.html"))

else:
    @app.get("/")
    def root():
        return {
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "docs": "/docs",
            "api": "/api/v1",
            "frontend": "Not built. Run: cd frontend && npm run build",
            "dev_mode": "Run frontend separately: cd frontend && npm run dev",
        }
