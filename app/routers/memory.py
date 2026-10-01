"""Memory, feedback, and self-improvement endpoints.

These give the AI long-term memory, a feedback loop, and self-evolution:
  - GET/POST/DELETE /api/memory    — per-user persistent memories
  - POST /api/chat/job/{id}/feedback — rate a response (thumbs up/down)
  - GET /api/feedback               — list feedback (admin)
  - GET /api/improvements           — list self-learned lessons
  - GET /api/alive                  — alive-status snapshot (memories, lessons, feedback count)
"""
import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..database import get_db
from ..models import (
    ApiSetting,
    ChatJob,
    Feedback,
    Memory,
    SelfImprovement,
    User,
    WorkerCommand,
)
from ..schemas import FeedbackIn, FeedbackOut, MemoryIn, MemoryOut, SelfImprovementOut

router = APIRouter(prefix="/api", tags=["memory"])


def _require_admin(user: User):
    if not user.is_admin:
        raise HTTPException(403, "Admin only")
    return user


def _mem_out(m: Memory) -> MemoryOut:
    return MemoryOut(
        id=m.id, content=m.content, kind=m.kind, source=m.source,
        importance=m.importance,
        created_at=m.created_at.isoformat() if m.created_at else "",
    )


# ── Memory CRUD ──────────────────────────────────────────────────────

@router.get("/memory", response_model=list[MemoryOut])
def list_memories(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = (
        db.query(Memory)
        .filter(Memory.user_id == user.id)
        .order_by(Memory.importance.desc(), Memory.created_at.desc())
        .limit(200)
        .all()
    )
    return [_mem_out(m) for m in rows]


@router.post("/memory", response_model=MemoryOut)
def add_memory(
    body: MemoryIn,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    m = Memory(
        user_id=user.id,
        content=body.content.strip(),
        kind=body.kind,
        source="manual",
        importance=max(1, min(5, body.importance)),
    )
    db.add(m)
    db.commit()
    db.refresh(m)
    return _mem_out(m)


@router.delete("/memory/{mem_id}")
def delete_memory(
    mem_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    m = db.query(Memory).filter(Memory.id == mem_id, Memory.user_id == user.id).first()
    if not m:
        raise HTTPException(404, "Memory not found")
    db.delete(m)
    db.commit()
    return {"ok": True}


# ── Feedback ─────────────────────────────────────────────────────────

@router.post("/chat/job/{job_id}/feedback", response_model=FeedbackOut)
def submit_feedback(
    job_id: str,
    body: FeedbackIn,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    job = db.query(ChatJob).filter(ChatJob.id == job_id, ChatJob.user_id == user.id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    if body.rating not in (-1, 1):
        raise HTTPException(422, "rating must be 1 or -1")

    fb = Feedback(
        job_id=job_id,
        user_id=user.id,
        conversation_id=job.conversation_id,
        rating=body.rating,
        comment=(body.comment or "").strip() or None,
    )
    db.add(fb)
    db.flush()

    # Self-improvement: negative feedback with a comment becomes a lesson
    # that gets injected into future system prompts.
    if body.rating == -1 and fb.comment:
        lesson = "Avoid this mistake (user reported): " + fb.comment[:500]
        existing = (
            db.query(SelfImprovement)
            .filter(SelfImprovement.content == lesson)
            .first()
        )
        if existing:
            existing.times_reinforced += 1
        else:
            db.add(SelfImprovement(kind="lesson", content=lesson, times_reinforced=1))

    db.commit()
    db.refresh(fb)
    return FeedbackOut(
        id=fb.id, job_id=fb.job_id, rating=fb.rating, comment=fb.comment,
        created_at=fb.created_at.isoformat() if fb.created_at else "",
    )


@router.get("/feedback", response_model=list[FeedbackOut])
def list_feedback(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_admin(user)
    rows = db.query(Feedback).order_by(Feedback.created_at.desc()).limit(200).all()
    return [
        FeedbackOut(
            id=f.id, job_id=f.job_id, rating=f.rating, comment=f.comment,
            created_at=f.created_at.isoformat() if f.created_at else "",
        )
        for f in rows
    ]


# ── Self-improvements ────────────────────────────────────────────────

@router.get("/improvements", response_model=list[SelfImprovementOut])
def list_improvements(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_admin(user)
    rows = (
        db.query(SelfImprovement)
        .filter(SelfImprovement.active == True)  # noqa: E712
        .order_by(SelfImprovement.times_reinforced.desc(), SelfImprovement.created_at.desc())
        .limit(100)
        .all()
    )
    return [
        SelfImprovementOut(
            id=s.id, kind=s.kind, content=s.content,
            times_reinforced=s.times_reinforced, active=s.active,
            created_at=s.created_at.isoformat() if s.created_at else "",
        )
        for s in rows
    ]


@router.delete("/improvements/{imp_id}")
def deactivate_improvement(
    imp_id: int,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_admin(user)
    s = db.query(SelfImprovement).filter(SelfImprovement.id == imp_id).first()
    if not s:
        raise HTTPException(404, "Improvement not found")
    s.active = False
    db.commit()
    return {"ok": True}


# ── Alive status ─────────────────────────────────────────────────────

@router.get("/alive")
def alive_status(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    mem_count = db.query(Memory).filter(Memory.user_id == user.id).count()
    lesson_count = (
        db.query(SelfImprovement)
        .filter(SelfImprovement.active == True)  # noqa: E712
        .count()
    )
    fb_count = db.query(Feedback).filter(Feedback.user_id == user.id).count()
    fb_good = db.query(Feedback).filter(Feedback.user_id == user.id, Feedback.rating == 1).count()
    fb_bad = db.query(Feedback).filter(Feedback.user_id == user.id, Feedback.rating == -1).count()

    # Current emotion/mind state set by the admin Command Center.
    mood_row = db.query(ApiSetting).filter(ApiSetting.key == "emotion_mood").first()
    int_row = db.query(ApiSetting).filter(ApiSetting.key == "emotion_intensity").first()
    mood = mood_row.value if mood_row and mood_row.value else "curious"
    try:
        intensity = int(int_row.value) if int_row and int_row.value else 3
    except Exception:
        intensity = 3
    # Worker heartbeat so the UI can show if the acting brain is connected.
    seen = db.query(ApiSetting).filter(ApiSetting.key == "worker_last_seen").first()
    worker_online = False
    if seen and seen.value:
        try:
            worker_online = (
                datetime.datetime.utcnow().timestamp() - float(seen.value)
            ) < 180
        except Exception:
            worker_online = False
    cmd_count = db.query(WorkerCommand).count()

    return {
        "memories": mem_count,
        "lessons": lesson_count,
        "feedback_total": fb_count,
        "feedback_good": fb_good,
        "feedback_bad": fb_bad,
        "alive": True,
        "emotion": mood,
        "mood": mood,
        "intensity": intensity,
        "commands": cmd_count,
        "commands_pending": db.query(WorkerCommand)
        .filter(WorkerCommand.status.in_(["pending", "running"]))
        .count(),
        "worker_online": worker_online,
        "uptime_note": "I learn from every conversation. Your memories persist across chats.",
    }
