"""FastAPI application entry point."""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.api.auth_routes import router as auth_router
from app.core.config import get_settings
from app.core.logging import get_logger, setup_logging
from app.db.session import async_session_maker
from app.services.model_service import sync_models

settings = get_settings()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    setup_logging()
    
    if settings.skip_model_sync or os.getenv("PYTEST_CURRENT_TEST"):
        logger.info("Skipping model sync on startup")
    else:
        # Sync models from Chutes API on startup
        logger.info("Syncing models from Chutes API on startup...")
        try:
            async with async_session_maker() as db:
                count = await sync_models(db)
                logger.info(f"Synced {count} models from Chutes API")
        except Exception as e:
            logger.error(f"Failed to sync models on startup: {e}")
    
    yield


app = FastAPI(
    title="Chutes Bench Runner",
    description="LLM Benchmark Runner for Chutes-hosted models",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware
cors_origins = [
    "http://localhost:3000",
    "https://chutes-bench-runner-ui.onrender.com",
    "https://chutes-bench-runner-frontend.onrender.com",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router)
app.include_router(auth_router, prefix="/api")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Chutes Bench Runner",
        "version": "0.1.0",
        "docs": "/docs",
    }
