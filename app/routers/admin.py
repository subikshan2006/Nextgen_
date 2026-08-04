"""Admin endpoints: users, models, system status, api settings."""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..auth import get_current_admin, hash_password
from ..database import active_driver, get_db, get_ollama_url
from ..models import ApiSetting, Conversation, User
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
