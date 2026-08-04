"""NEXTGEN AI v20 — FastAPI web application.

Local dev:  uvicorn app.main:app --reload
Vercel:     vercel.json routes to this module.
"""
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .config import get_settings
from .database import init_db
from .routers import admin, auth, chat

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


app = FastAPI(
    title="NEXTGEN AI v20",
    version="20.0.0",
    description="ChatGPT conversation + opencode tools + Claude reasoning — fully local.",
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins if settings.cors_origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(chat.router)
app.include_router(admin.router)


@app.get("/api/health")
def health():
    return {"status": "ok", "app": settings.app_name}


@app.get("/api/models")
async def public_models():
    from .database import get_db, get_ollama_url
    from .services.ollama import OllamaClient

    db = get_db().__next__()
    try:
        client = OllamaClient(get_ollama_url(db))
        return {"models": await client.list_models(), "reachable": await client.check()}
    finally:
        db.close()


# ---- Static frontend ----
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index():
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/login")
def login_page():
    return FileResponse(str(STATIC_DIR / "login.html"))


@app.get("/admin")
def admin_page():
    return FileResponse(str(STATIC_DIR / "admin.html"))
