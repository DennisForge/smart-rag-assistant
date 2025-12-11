"""
Main FastAPI application setup.

Loads config from .env and registers all route handlers.
Routes are organized by API version for easier maintenance.
"""

from fastapi import FastAPI

from app.core.config import get_settings
from app.api.v1.routes_health import router as health_router

# Load settings (cached singleton, safe to call multiple times)
settings = get_settings()

# Initialize FastAPI app
# Auto-generates docs at /docs and /redoc
app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
)

# Register routers
# Prefix adds /api/v1 to all routes (e.g., /health -> /api/v1/health)
app.include_router(health_router, prefix=settings.api_v1_prefix)
