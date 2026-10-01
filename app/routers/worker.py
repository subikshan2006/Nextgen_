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
from ..models import ApiSetting, ChatJob, Conversation, Memory, Message, SearchResult, SelfImprovement, User, WorkerCommand
from ..schemas import MemoryIn, WorkerCommandCompleteIn, WorkerCompleteIn
from ..services.search import format_search_context

router = APIRouter(prefix="/api/worker", tags=["worker"])


def _require_admin(user: User):
    if not user.is_admin:
        raise HTTPException(403, "Admin only")
    return user


def _claim_commands(db: Session, limit: int = 5):
    """Claim up to `limit` admin commands for a GPU worker.

    Also re-queues commands stranded in `running`. Kaggle sessions die
    without warning, and a worker that dies mid-command leaves the row in
    `running` forever, so those commands would never be executed."""
    stale = datetime.datetime.utcnow() - datetime.timedelta(minutes=10)
    out = []
    now = datetime.datetime.utcnow()
    rows = (
        db.query(WorkerCommand)
        .filter(
            or_(
                WorkerCommand.status == "pending",
                (WorkerCommand.status == "running")
                & (
                    or_(
                        WorkerCommand.updated_at.is_(None),
                        WorkerCommand.updated_at < stale,
                    )
                ),
            )
        )
        .order_by(WorkerCommand.created_at.asc())
        .limit(limit)
        .all()
    )
    for c in rows:
        c.status = "running"
        c.updated_at = now
        db.add(c)
        out.append({
            "command_id": c.id,
            "id": c.id,
            "kind": c.kind,
            "payload": c.payload,
        })
    db.commit()
    return out


def build_lifelong_context(db: Session, user_id: int) -> str:
    """Assemble the AI's lifelong context: per-user memories + self-learned
    lessons that were reinforced by feedback. This is injected alongside the
    base system prompt so the AI 'remembers' the user and improves over time."""
    blocks = []

    lessons = (
        db.query(SelfImprovement)
        .filter(SelfImprovement.active == True)  # noqa: E712
        .order_by(SelfImprovement.times_reinforced.desc())
        .limit(12)
        .all()
    )
    if lessons:
        lines = []
        for s in lessons:
            if s.kind == "lesson" and s.content:
                lines.append("- " + s.content[:600])
        if lines:
            blocks.append(
                "## LESSONS LEARNED (from user feedback across all chats)\n"
                + "\n".join(lines)
                + "\nFollow these in every reply. They exist to make your "
                "answers better."
            )

    # Global memories (user_id IS NULL) are admin-issued instructions from the
    # Command Center and must reach EVERY user, not just the admin who queued
    # them. Previously the filter was `== user_id`, which both hid global
    # memories from everyone and scoped admin instructions to the admin only.
    memories = (
        db.query(Memory)
        .filter(or_(Memory.user_id == user_id, Memory.user_id.is_(None)))
        .order_by(Memory.importance.desc(), Memory.created_at.desc())
        .limit(40)
        .all()
    )
    if memories:
        lines = []
        for m in memories:
            if m.content and len(m.content) > 2:
                scope = "admin" if m.user_id is None else "user"
                lines.append("- [%s/%s] %s" % (scope, m.kind, m.content[:400]))
        if lines:
            blocks.append(
                "## LONG-TERM MEMORY (remember these, they were learned from "
                "past chats and from your operator)\n" + "\n".join(lines)
            )

    # Emotion the operator set via the Command Center.
    mood_row = db.query(ApiSetting).filter(ApiSetting.key == "emotion_mood").first()
    int_row = db.query(ApiSetting).filter(ApiSetting.key == "emotion_intensity").first()
    mood = (mood_row.value if mood_row and mood_row.value else "") or ""
    try:
        intensity = int(int_row.value) if int_row and int_row.value else 0
    except Exception:
        intensity = 0
    if mood:
        blocks.append(
            "## YOUR CURRENT EMOTIONAL STATE\n"
            "You are feeling %s (intensity %d/10). Let this subtly colour your "
            "tone without announcing it or mentioning that it was set."
            % (mood, intensity or 3)
        )

    # Tools the operator has explicitly enabled/disabled.
    tool_rows = (
        db.query(ApiSetting)
        .filter(ApiSetting.key.like("tool_%"))
        .order_by(ApiSetting.key.asc())
        .all()
    )
    if tool_rows:
        lines = []
        for t in tool_rows:
            name = t.key[len("tool_"):]
            state = "ENABLED" if str(t.value) == "1" else "DISABLED"
            lines.append("- %s: %s" % (name, state))
        if lines:
            blocks.append(
                "## TOOL ACCESS GRANTED BY YOUR OPERATOR\n"
                + "\n".join(lines)
                + "\nA DISABLED tool is off limits: do not use it and do not "
                "pretend to have used it. Say it is switched off if asked."
            )

    if blocks:
        return (
            "\n\n======= LIFELONG MEMORY & SELF-IMPROVEMENT CONTEXT =======\n"
            + "\n\n".join(blocks)
        )
    return ""


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
        lesson_context = build_lifelong_context(db, job.user_id)
        messages = [{"role": "system", "content": settings.default_system_prompt + lesson_context}]
        messages.extend(history)
        messages.append({"role": "user", "content": job.prompt})
        # Attach web search results (if any) to the user message so the model
        # answers from them and cites the sources.
        srows = (
            db.query(SearchResult)
            .filter(SearchResult.job_id == job.id)
            .order_by(SearchResult.rank.asc())
            .all()
        )
        if srows:
            ctx = format_search_context(job.prompt, [
                {"title": r.title, "url": r.url, "snippet": r.snippet} for r in srows
            ])
            if ctx:
                messages[-1] = dict(messages[-1])
                messages[-1]["content"] += "\n\n" + ctx
        out.append({
            "job_id": job.id,
            "model": job.model or settings.default_model,
        "messages": messages,
        "want_zip": bool(job.want_zip),
    })
    db.commit()

    commands = _claim_commands(db)
    return {"jobs": out, "commands": commands}


@router.post("/heartbeat")
def heartbeat(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Called periodically by the GPU worker so the site knows it is alive."""
    _require_admin(user)
    ts = str(datetime.datetime.utcnow().timestamp())
    row = db.query(ApiSetting).filter(ApiSetting.key == "worker_last_seen").first()
    if row:
        row.value = ts
    else:
        db.add(ApiSetting(key="worker_last_seen", value=ts))
    db.commit()
    return {"ok": True}


@router.post("/memory")
def save_memory(
    body: MemoryIn,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """The worker (AI) can autonomously save a memory it learned about the
    user. Deduplicates near-identical memories."""
    _require_admin(user)
    from sqlalchemy import func

    probe = (body.content or "").strip().lower()
    dup = None
    for m in db.query(Memory).filter(Memory.user_id == user.id).all():
        if m.content and m.content.strip().lower() == probe:
            dup = m
            break
        # duplicate if similar starts (e.g. "loves JavaScript" vs "loves JS")
        a, b = probe[:60], (m.content or "").lower()[:60]
        if a and b and (a in b or b in a):
            dup = m
            break
    if dup:
        dup.importance = max(dup.importance, min(5, body.importance))
        db.commit()
        return {"ok": True, "dedup": True}
    m = Memory(
        user_id=user.id,
        content=(body.content or "").strip()[:4000],
        kind=body.kind,
        source="auto",
        importance=max(1, min(5, body.importance)),
    )
    db.add(m)
    db.commit()
    return {"ok": True, "dedup": False}


@router.get("/commands")
def poll_commands(
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """The GPU worker polls pending admin commands (just like jobs) and
    executes them. Returns at most 3 at a time; each is idempotent by id."""
    _require_admin(user)
    out = _claim_commands(db, limit=3)
    return {"commands": out}


@router.post("/commands/complete")
def complete_command(
    body: WorkerCommandCompleteIn,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """The worker reports back what it did for an admin command."""
    _require_admin(user)
    c = db.query(WorkerCommand).filter(WorkerCommand.id == body.command_id).first()
    if not c:
        raise HTTPException(404, "Command not found")
    c.status = body.status
    c.result = (body.result or "")[:2000]
    c.completed_at = datetime.datetime.utcnow()
    db.commit()
    return {"ok": True}


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
        job.zip_b64 = body.zip_b64
        job.zip_name = body.zip_name
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


@router.post("/command_complete")
def command_complete(
    body: WorkerCommandCompleteIn,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """The GPU worker reports back the result of an admin command it executed."""
    import uuid as _uuid

    _require_admin(user)
    c = db.query(WorkerCommand).filter(WorkerCommand.id == body.command_id).first()
    if not c:
        raise HTTPException(404, "Command not found")
    c.status = body.status
    c.result = (body.result or "")[:2000]
    c.completed_at = datetime.datetime.utcnow()
    # Side-effect on the actual AI state:
    db.commit()
    return {"ok": True}
