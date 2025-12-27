"""FastAPI application entry point."""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import get_settings
from app.core.logging import setup_logging

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    setup_logging()
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


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Chutes Bench Runner",
        "version": "0.1.0",
        "docs": "/docs",
    }

