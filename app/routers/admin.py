"""Admin endpoints: users, models, system status, api settings."""
from fastapi import APIRouter, Body, Depends, HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..auth import get_current_admin, hash_password
from ..database import active_driver, get_db, get_ollama_url
from ..models import ApiSetting, Conversation, Message, User
from ..schemas import (
    ModelInfo, OllamaStatus, OllamaUrlIn, SystemStatus, UserAdminUpdate, UserOut,
)
from ..services.ollama import OllamaClient

router = APIRouter(prefix="/api/admin", tags=["admin"])


def _user_out(u: User) -> UserOut:
    return UserOut(
        id=u.id, email=u.email, username=u.username,
        is_admin=u.is_admin, is_active=u.is_active,
    )


@router.get("/users", response_model=list[UserOut])
def list_users(_: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    users = db.query(User).order_by(User.id).all()
    return [_user_out(u) for u in users]


@router.patch("/users/{user_id}", response_model=UserOut)
def update_user(user_id: int, body: UserAdminUpdate, _: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    if body.is_admin is not None:
        user.is_admin = body.is_admin
    if body.is_active is not None:
        user.is_active = body.is_active
    if body.password:
        user.password_hash = hash_password(body.password)
    db.commit()
    db.refresh(user)
    return _user_out(user)


@router.delete("/users/{user_id}")
def delete_user(user_id: int, admin: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    if admin.id == user_id:
        raise HTTPException(400, "Cannot delete yourself")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    db.delete(user)
    db.commit()
    return {"ok": True}


@router.get("/users/activity")
def users_activity(_: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    """Full account overview for every user: identity, activity counters and
    what they searched. Passwords are bcrypt-hashed (one-way) so plaintext is
    never stored — use the reset endpoint to set a new one."""
    from ..models import ChatJob, Feedback, Memory, SearchResult

    rows = []
    for u in db.query(User).order_by(User.id).all():
        convs = db.query(Conversation).filter(Conversation.user_id == u.id).count()
        msgs = (
            db.query(func.count(Message.id))
            .join(Conversation, Message.conversation_id == Conversation.id)
            .filter(Conversation.user_id == u.id)
            .scalar()
            or 0
        )
        jobs = db.query(ChatJob).filter(ChatJob.user_id == u.id).count()
        searches = (
            db.query(func.count(SearchResult.id))
            .join(ChatJob, SearchResult.job_id == ChatJob.id)
            .filter(ChatJob.user_id == u.id)
            .scalar()
            or 0
        )
        mems = db.query(Memory).filter(Memory.user_id == u.id).count()
        fb = db.query(Feedback).filter(Feedback.user_id == u.id).count()
        # Distinct things this user asked the AI (their prompts) — the search
        # terms they used are the job prompts that produced search results.
        prompts = [
            r[0]
            for r in db.query(ChatJob.prompt)
            .filter(ChatJob.user_id == u.id)
            .order_by(ChatJob.created_at.desc())
            .limit(25)
            .all()
            if r[0]
        ]
        rows.append({
            "id": u.id,
            "email": u.email,
            "username": u.username,
            "is_admin": bool(u.is_admin),
            "is_active": bool(u.is_active),
            "created_at": u.created_at.isoformat() if u.created_at else None,
            "last_login": u.last_login.isoformat() if u.last_login else None,
            "conversations": convs,
            "messages": int(msgs),
            "jobs": jobs,
            "searches": int(searches),
            "memories": mems,
            "feedback": fb,
            "recent_searches": prompts,
            "password_storage": "bcrypt (hashed — not reversible)",
        })
    return {"users": rows}


@router.get("/searches")
def all_searches(limit: int = 100, _: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    """Every web-search result the AI fetched, with who asked for it."""
    from ..models import ChatJob, SearchResult

    q = (
        db.query(SearchResult, ChatJob, User)
        .join(ChatJob, SearchResult.job_id == ChatJob.id)
        .join(User, ChatJob.user_id == User.id)
        .order_by(SearchResult.id.desc())
        .limit(max(1, min(limit, 500)))
        .all()
    )
    out = []
    for sr, job, user in q:
        out.append({
            "id": sr.id,
            "user": user.username,
            "email": user.email,
            "query": (job.prompt or "")[:300],
            "title": sr.title,
            "url": sr.url,
            "snippet": (sr.snippet or "")[:220],
            "rank": sr.rank,
        })
    return {"searches": out}


@router.post("/users/{user_id}/reset-password")
def reset_password(
    user_id: int,
    payload: dict = Body(...),
    _: User = Depends(get_current_admin),
    db: Session = Depends(get_db),
):
    """Set a new password for any account (bcrypt-hashed at rest).

    Body: {"new_password": "..."}  — the plaintext is only ever seen in this
    request; it is hashed immediately and never stored or logged.
    """
    new_password = (payload or {}).get("new_password") or ""
    if len(new_password) < 8:
        raise HTTPException(422, "new_password must be at least 8 characters")
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(404, "User not found")
    user.password_hash = hash_password(new_password)
    db.commit()
    return {
        "ok": True,
        "user": user.username,
        "note": "password updated (stored as a bcrypt hash, never in plaintext)",
    }


@router.get("/models", response_model=list[ModelInfo])
async def list_models(_: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    client = OllamaClient(get_ollama_url(db))
    return [ModelInfo(**m) for m in await client.list_models()]


@router.get("/status", response_model=SystemStatus)
async def system_status(_: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    url = get_ollama_url(db)
    client = OllamaClient(url)
    reachable = await client.check()
    models = [ModelInfo(**m) for m in await client.list_models()]
    return SystemStatus(
        app="NEXTGEN AI v20",
        version="20.0.0",
        database=active_driver(),
        ollama=OllamaStatus(reachable=reachable, message=f"OK ({url})" if reachable else "Ollama unreachable"),
        models=models,
        total_users=db.query(User).count(),
        total_conversations=db.query(Conversation).count(),
    )


@router.post("/ollama_url")
def set_ollama_url(body: OllamaUrlIn, _: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    """Called by the Colab/Kaggle GPU notebooks on startup so the app always
    points at the current tunnel URL."""
    row = db.query(ApiSetting).filter(ApiSetting.key == "ollama_url").first()
    if not row:
        row = ApiSetting(key="ollama_url", value=body.url)
        db.add(row)
    else:
        row.value = body.url
    db.commit()
    return {"ok": True, "ollama_url": body.url}


@router.get("/settings")
def get_settings(_: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    rows = db.query(ApiSetting).all()
    return {r.key: r.value for r in rows}


@router.put("/settings/{key}")
def set_setting(key: str, value: str, _: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    row = db.query(ApiSetting).filter(ApiSetting.key == key).first()
    if not row:
        row = ApiSetting(key=key, value=value)
        db.add(row)
    else:
        row.value = value
    db.commit()
    return {"key": key, "value": value}
