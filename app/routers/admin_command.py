"""Admin Command Console — lets the admin issue commands the AI worker
actually executes: self-update, remember, improve, set emotion, grant tool
access, restart, run a tool. Commands are queued durably, then consumed by
the GPU worker during its poll (it acts on them like chat jobs)."""
import datetime
import json
import os
import subprocess
import uuid

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from ..auth import get_current_admin
from ..database import get_db
from ..models import ApiSetting, Feedback, Memory, SelfImprovement, User, WorkerCommand
from ..schemas import EmotionOut, WorkerCommandIn, WorkerCommandOut

router = APIRouter(prefix="/api/admin/commands", tags=["admin-command"])


def _is_admin(user: User) -> bool:
    return bool(getattr(user, "is_admin", False))


# Kinds whose effect is a pure server-side state change. These are applied here
# and now, instead of being delegated to the remote GPU worker, because the
# worker has no way to write ApiSetting/Memory rows the system actually reads
# (it could only append a memory row, so "emotion set" never changed the
# emotion and admin instructions never reached real users).
_HOST_KINDS = ("emotion", "remember", "improve", "grant")


def _apply_host_side(db: Session, kind: str, payload: str) -> str:
    """Perform a command that only touches server state. Raises on error."""
    if kind == "grant":
        data = json.loads(payload or "{}") or {}
        return _grant_tool(db, data.get("tool", ""), bool(data.get("enabled", True)))
    if kind == "emotion":
        return _set_emotion(db, payload)
    if kind == "remember":
        return _remember(db, payload)
    if kind == "improve":
        return _improve(db, payload)
    raise ValueError(f"command type '{kind}' is not host-runnable")


@router.post("", response_model=WorkerCommandOut)
def issue_command(cmd: WorkerCommandIn, user: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    """Admin issues a command for the AI to carry out.

    Server-state commands (emotion / remember / improve / grant) are applied
    immediately and recorded as done, so the effect is real and the admin UI
    never shows a phantom success. GPU commands (self_update) stay queued for
    the remote worker to claim."""
    cid = uuid.uuid4().hex[:16]
    status = "pending"
    result = None
    if cmd.kind in _HOST_KINDS:
        try:
            result = _apply_host_side(db, cmd.kind, cmd.payload)
            status = "done"
        except Exception as e:
            result = f"error: {e}"
            status = "error"
    row = WorkerCommand(
        id=cid,
        kind=cmd.kind,
        payload=cmd.payload,
        status=status,
        result=result,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.get("", response_model=list[WorkerCommandOut])
def list_commands(user: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    return (
        db.query(WorkerCommand)
        .order_by(WorkerCommand.created_at.desc())
        .limit(50)
        .all()
    )


def _grant_tool(db: Session, name: str, enabled: bool) -> str:
    key = f"tool_{name}"
    setting = db.query(ApiSetting).filter(ApiSetting.key == key).first()
    if setting:
        setting.value = "1" if enabled else "0"
    else:
        db.add(ApiSetting(key=key, value="1" if enabled else "0"))
    db.commit()
    return f"tool '{name}' {'GRANTED' if enabled else 'REVOKED'}"


def _set_emotion(db: Session, payload: str) -> str:
    try:
        data = json.loads(payload or "{}")
    except Exception:
        data = {}
    mood = data.get("mood", "curious")
    intensity = int(data.get("intensity", 3))
    setting = db.query(ApiSetting).filter(ApiSetting.key == "emotion_mood").first()
    if setting:
        setting.value = mood
    else:
        db.add(ApiSetting(key="emotion_mood", value=mood))
    s2 = db.query(ApiSetting).filter(ApiSetting.key == "emotion_intensity").first()
    if s2:
        s2.value = str(intensity)
    else:
        db.add(ApiSetting(key="emotion_intensity", value=str(intensity)))
    db.commit()
    return f"emotion set to {mood} (intensity {intensity})"


def _remember(db: Session, payload: str) -> str:
    try:
        data = json.loads(payload or "{}")
    except Exception:
        data = {}
    content = data.get("content") or data.get("text") or payload
    kind = data.get("kind", "instruction")
    # user_id stays NULL on purpose: this is an operator instruction that must
    # reach every user, not a note about the admin's own chats.
    existing = (
        db.query(Memory)
        .filter(Memory.user_id.is_(None), Memory.content == content)
        .first()
    )
    if existing:
        return f"already remembered: {content[:80]}"
    mem = Memory(content=content, kind=kind, source="admin",
                 importance=max(1, min(5, int(data.get("importance", 5)))))
    db.add(mem)
    db.commit()
    return f"remembered for all users: {content[:80]}"


def _improve(db: Session, payload: str) -> str:
    try:
        data = json.loads(payload or "{}")
    except Exception:
        data = {}
    content = data.get("content") or data.get("text") or payload
    imp = SelfImprovement(kind="lesson", content=content)
    db.add(imp)
    db.commit()
    return f"improvement lesson stored: {content[:80]}"


def _self_update(db: Session) -> str:
    """Signal the trainer to be pushed to a free GPU slot. Enqueues the real
    push via subprocess (kaggle trainer) so the model retrains on memories +
    lessons + feedback — the deepest level of autonomous self-update."""
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    script = os.path.join(base, "kaggle", "push_train", "push_train.ps1")
    if os.path.exists(script):
        try:
            subprocess.Popen(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", script],
                cwd=os.path.dirname(script),
                creationflags=subprocess.CREATE_NO_WINDOW,
            )
            return "self-update TRIGGERED: trainer will retrain the model when a GPU slot frees"
        except Exception as e:
            return f"self-update starter failed: {e}"
    return "self-update queued for worker (trainer script not found on host)"


@router.post("/run", response_model=WorkerCommandOut)
def run_now(cmd: WorkerCommandIn, user: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    """Execute a command immediately on the host (not via the remote worker) —
    for things that must happen right now: grant tools, set emotion, remember."""
    kind = cmd.kind
    try:
        if kind in _HOST_KINDS:
            result = _apply_host_side(db, kind, cmd.payload)
        elif kind == "self_update":
            result = _self_update(db)
        else:
            result = f"command type '{kind}' is not host-runnable"
        status = "done"
    except Exception as e:
        result = f"error: {e}"
        status = "error"

    row = WorkerCommand(
        id=uuid.uuid4().hex[:16],
        kind=kind,
        payload=cmd.payload,
        status=status,
        result=result,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


@router.get("/emotion", response_model=EmotionOut)
def current_emotion(user: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    mood = db.query(ApiSetting).filter(ApiSetting.key == "emotion_mood").first()
    intensity = db.query(ApiSetting).filter(ApiSetting.key == "emotion_intensity").first()
    return EmotionOut(
        mood=(mood.value if mood else "curious"),
        intensity=int(intensity.value) if (intensity and intensity.value) else 3,
        last_change=None,
    )


@router.get("/lesson-counts")
def lesson_counts(user: User = Depends(get_current_admin), db: Session = Depends(get_db)):
    mems = db.query(Memory).count()
    lessons = db.query(SelfImprovement).count()
    fb = db.query(Feedback).count()
    cmds = db.query(WorkerCommand).count()
    return {"memories": mems, "lessons": lessons, "feedback": fb, "commands": cmds}
