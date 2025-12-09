from fastapi import FastAPI
from app.config.settings import settings

app = FastAPI(
    title=settings.app_name,
    debug=settings.debug,
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "app": settings.app_name,
        "environment": settings.environment,
    }


@app.get("/")
def root():
    return {"message": "Smart RAG Assistant backend is running."}
