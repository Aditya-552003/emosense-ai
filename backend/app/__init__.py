"""
app — FastAPI application package for EmoSense AI.

Provides the ``create_app()`` factory function which assembles the
FastAPI instance with middleware, lifespan events, routes, and
static-file serving.

Usage
-----
    # Via uvicorn factory
    uvicorn app:create_app --factory --host 127.0.0.1 --port 8080

    # Or direct import
    from app import create_app
    app = create_app()
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app import config
from app.services.model_loader import load_models, unload_models

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format=config.LOG_FORMAT,
    datefmt=config.LOG_DATE_FORMAT,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan — load models on startup, release on shutdown
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ML artifacts on startup; release on shutdown."""
    load_models()
    yield
    log.info("Shutting down ...")
    unload_models()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
def create_app() -> FastAPI:
    """Build and return the fully-configured FastAPI application."""

    app = FastAPI(
        title=config.API_TITLE,
        description=config.API_DESCRIPTION,
        version=config.API_VERSION,
        lifespan=lifespan,
    )

    # ── CORS middleware ──────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Register routers ─────────────────────────────────────────────────
    from app.routes.health import router as health_router
    from app.routes.predict import router as predict_router

    app.include_router(health_router)
    app.include_router(predict_router)

    # ── Serve frontend static files at /ui ───────────────────────────────
    frontend_dir = str(config.FRONTEND_DIR)
    if os.path.isdir(frontend_dir):
        app.mount(
            "/ui",
            StaticFiles(directory=frontend_dir, html=True),
            name="frontend",
        )
        log.info(f"Serving frontend at /ui  →  {frontend_dir}")
    else:
        log.warning(f"Frontend directory not found: {frontend_dir}")

    # ── Global exception handler ─────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        log.error(f"Unhandled error on {request.url}: {exc}")
        return JSONResponse(
            status_code=500,
            content={"success": False, "detail": "Internal server error."},
        )

    return app
