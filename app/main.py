"""NEXTGEN AI v20 — FastAPI web application.

Local dev:  uvicorn app.main:app --reload
Vercel:     vercel.json routes to this module.
"""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .config import get_settings
from .database import init_db
from .routers import admin, auth, chat, worker

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
app.include_router(worker.router)


@app.get("/api/health")
def health():
    return {"status": "ok", "app": settings.app_name}


@app.get("/api/models")
async def public_models():
    import datetime as _dt

    from .database import get_db
    from .models import ApiSetting

    db = get_db().__next__()
    try:
        row = db.query(ApiSetting).filter(ApiSetting.key == "worker_last_seen").first()
        online = False
        if row and row.value:
            try:
                last = float(row.value)
                online = (_dt.datetime.utcnow().timestamp() - last) < 180
            except Exception:
                pass
        models = []
        if online:
            models = [
                {"name": "nextgen-trained", "size_gb": None},
            ]
        return {"models": models, "reachable": online, "worker_online": online}
    finally:
        db.close()


@app.get("/api/worker/code")
def worker_code():
    """Serve the Colab/Kaggle worker script so notebooks can bootstrap with a
    short fetch+exec cell (single source of truth in this repo).
    Injects KAGGLE_ACCOUNTS from env vars so API keys stay out of source."""
    import json as _json, os
    worker_file = Path(__file__).resolve().parent.parent / "colab_nextgen.py"
    if not worker_file.exists():
        from fastapi.responses import JSONResponse

        return JSONResponse({"error": "colab_nextgen.py not found"}, status_code=404)
    from fastapi.responses import PlainTextResponse

    code = worker_file.read_text(encoding="utf-8")

    # Inject Kaggle accounts from env vars (avoids hardcoding keys in source)
    accounts_json = os.environ.get("KAGGLE_ACCOUNTS_JSON", "")
    if accounts_json:
        code = code.replace(
            "# __KAGGLE_ACCOUNTS_INJECT__",
            'KAGGLE_ACCOUNTS = ' + accounts_json,
        )

    return PlainTextResponse(code)


@app.post("/api/admin/kaggle_keepalive")
def kaggle_keepalive(request: Request):
    """Called on a schedule (Vercel cron / GitHub Actions). If the Kaggle GPU
    session is not running, re-push it so the worker fleet stays up without
    the user's machine being on."""
    from fastapi.responses import JSONResponse

    expected = os.environ.get("KAGGLE_KEEPALIVE_KEY", "")
    key = request.headers.get("x-keepalive-key", "")
    if not expected or key != expected:
        return JSONResponse({"error": "forbidden"}, status_code=401)

    username = os.environ.get("KAGGLE_USERNAME", "")
    api_key = os.environ.get("KAGGLE_KEY", "")
    if not username or not api_key:
        return JSONResponse({"error": "KAGGLE_USERNAME/KAGGLE_KEY env not set"}, status_code=500)

    try:
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = api_key
        os.environ["KAGGLE_CONFIG_DIR"] = "/tmp"
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        try:
            status = api.kernels_status("subikshan18/nextgen-gpu")
            raw = status if isinstance(status, dict) else json.loads(str(status))
            st = str(raw.get("status", "")).strip().lower()
        except Exception:
            # 404 = kernel exists but no active session (e.g. GPU quota exhausted
            # until the weekly refresh) -> treat as "needs push".
            st = "nosession"
        if st in ("running", "queued", "complete", "completed"):
            return JSONResponse({"action": "ok", "status": st})
        folder = str(Path(__file__).resolve().parent.parent / "kaggle")
        api.kernels_push_cli(folder, timeout=32400, acc="GPU")
        try:
            from kagglesdk.kernels.types.kernels_api_service import (
                ApiCreateKernelSessionRequest,
            )

            with api.build_kaggle_client() as kaggle:
                req = ApiCreateKernelSessionRequest()
                req.slug = "subikshan18/nextgen-gpu"
                req.language = "python"
                req.kernel_type = "notebook"
                req.enable_internet = True
                kaggle.kernels.kernels_api_client.create_kernel_session(req)
        except Exception:
            pass
        return JSONResponse({"action": "repushed", "status": st})
    except Exception as e:
        return JSONResponse({"action": "error", "error": str(e)[:500]}, status_code=500)


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
