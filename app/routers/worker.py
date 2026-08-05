"""GPU worker endpoints: claim queued chat jobs and submit completions.

The remote GPU machine (Colab/Kaggle) polls these instead of exposing a
tunnel, so the deployed site never needs to reach the GPU directly.
"""
import datetime
import json

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import or_
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..config import get_settings
from ..database import get_db
from ..models import ChatJob, Conversation, Message, User
from ..schemas import WorkerCompleteIn

router = APIRouter(prefix="/api/worker", tags=["worker"])


def _require_admin(user: User):
    if not user.is_admin:
        raise HTTPException(403, "Admin only")
    return user


@router.get("/poll")
def poll_jobs(
    limit: int = 5,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Claim pending jobs (or stale running ones) for the GPU worker."""
    _require_admin(user)
    settings = get_settings()
    stale = datetime.datetime.utcnow() - datetime.timedelta(minutes=10)
    jobs = (
        db.query(ChatJob)
        .filter(
            or_(
                ChatJob.status == "pending",
                (ChatJob.status == "running") & (ChatJob.updated_at < stale),
            )
        )
        .order_by(ChatJob.created_at.asc())
        .limit(max(1, min(limit, 20)))
        .all()
    )
    out = []
    now = datetime.datetime.utcnow()
    for job in jobs:
        job.status = "running"
        job.updated_at = now
        try:
            history = json.loads(job.history or "[]")
        except Exception:
            history = []
        messages = [{"role": "system", "content": settings.default_system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": job.prompt})
        out.append({
            "job_id": job.id,
            "model": job.model or settings.default_model,
            "messages": messages,
        })
    db.commit()
    return {"jobs": out}


@router.post("/complete")
def complete_job(
    body: WorkerCompleteIn,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Submit a finished job from the GPU worker."""
    _require_admin(user)
    job = db.query(ChatJob).filter(ChatJob.id == body.job_id).first()
    if not job:
        raise HTTPException(404, "Job not found")
    now = datetime.datetime.utcnow()
    if body.error:
        job.status = "error"
        job.error = body.error
        if job.conversation_id:
            db.add(Message(
                conversation_id=job.conversation_id, role="assistant",
                content="(error: %s)" % body.error[:2000],
            ))
    else:
        job.status = "done"
        job.response = body.response or ""
        if job.conversation_id:
            db.add(Message(
                conversation_id=job.conversation_id, role="assistant",
                content=body.response or "(no response)",
            ))
            conv = db.query(Conversation).filter(
                Conversation.id == job.conversation_id
            ).first()
            if conv:
                conv.updated_at = now
    job.completed_at = now
    job.updated_at = now
    db.commit()
    return {"ok": True}
