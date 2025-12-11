"""
Health check endpoints for monitoring service status.

Used by load balancers, K8s probes, and deployment verification.
"""

from fastapi import APIRouter

from app.core.config import get_settings

# Tags group endpoints in the API docs
router = APIRouter(tags=["health"])


@router.get("/health")
def read_health():
    """
    Basic health-check endpoint.

    This can later be extended to include deeper checks, for example:
    - database connection status
    - vector store availability
    - disk / memory usage

    Returns:
        dict: {"status": "ok", "app_name": str, "environment": str}
    """
    settings = get_settings()

    return {
        "status": "ok",
        "app_name": settings.app_name,
        "environment": settings.environment,
    }
