"""Chat endpoints: create/list conversations, stream responses via SSE, queue jobs."""
import base64
import io
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..config import get_settings
from ..database import get_db, get_ollama_url
from ..models import ChatJob, Conversation, Message, SearchResult, User
from ..schemas import ChatJobOut, ChatJobRequest, ChatRequest, ConversationOut, MessageOut, SearchSourceOut
from ..services.ollama import OllamaClient
from ..services.search import search_web

router = APIRouter(prefix="/api/chat", tags=["chat"])


def _conv_out(c: Conversation) -> ConversationOut:
    return ConversationOut(
        id=c.id, title=c.title, model=c.model,
        created_at=c.created_at.isoformat() if c.created_at else "",
    )


@router.get("/conversations", response_model=list[ConversationOut])
def list_conversations(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    convs = (
        db.query(Conversation)
        .filter(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
        .limit(100)
        .all()
    )
    return [_conv_out(c) for c in convs]


@router.get("/conversations/{conv_id}", response_model=list[MessageOut])
def get_conversation(conv_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    conv = db.query(Conversation).filter(
        Conversation.id == conv_id, Conversation.user_id == user.id
    ).first()
    if not conv:
        raise HTTPException(404, "Conversation not found")
    return [
        MessageOut(id=m.id, role=m.role, content=m.content)
        for m in conv.messages
    ]


@router.delete("/conversations/{conv_id}")
def delete_conversation(conv_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    conv = db.query(Conversation).filter(
        Conversation.id == conv_id, Conversation.user_id == user.id
    ).first()
    if not conv:
        raise HTTPException(404, "Conversation not found")
    db.delete(conv)
    db.commit()
    return {"ok": True}


@router.post("/stream")
async def stream_chat(body: ChatRequest, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    """Stream a model reply as Server-Sent Events."""
    settings = get_settings()

    # Resolve conversation
    conv_id = body.conversation_id
    if conv_id:
        conv = db.query(Conversation).filter(
            Conversation.id == conv_id, Conversation.user_id == user.id
        ).first()
        if not conv:
            raise HTTPException(404, "Conversation not found")
    else:
        title = body.message.strip()[:60] or "New chat"
        conv = Conversation(user_id=user.id, title=title, model=body.model or settings.default_model)
        db.add(conv)
        db.commit()
        db.refresh(conv)
        conv_id = conv.id

    # Build the message history from DB + the new user message
    history = []
    for m in conv.messages:
        history.append({"role": m.role, "content": m.content})
    history.append({"role": "user", "content": body.message})

    # Save the user message
    db.add(Message(conversation_id=conv.id, role="user", content=body.message))
    db.commit()

    system_prompt = body.system_prompt or settings.default_system_prompt
    ollama_messages = [{"role": "system", "content": system_prompt}] + history

    async def event_gen():
        url = get_ollama_url(db)
        client = OllamaClient(url)
        collected = []
        thinking = []
        yield "event: start\ndata: " + json.dumps({"conversation_id": conv.id, "title": conv.title}) + "\n\n"
        try:
            async for evt in client.chat(
                ollama_messages, model=body.model, temperature=0.7, max_tokens=1024,
            ):
                if evt["t"] == "think":
                    thinking.append(evt["token"])
                    yield "data: " + json.dumps({"thinking": evt["token"]}) + "\n\n"
                else:
                    collected.append(evt["token"])
                    yield "data: " + json.dumps({"token": evt["token"]}) + "\n\n"
        except Exception as e:
            yield "event: error\ndata: " + json.dumps({"error": str(e)}) + "\n\n"
            return
        # Persist the assistant message
        content = "".join(collected).strip()
        db.add(Message(conversation_id=conv.id, role="assistant", content=content or "(no response)"))
        db.commit()
        yield "event: done\ndata: " + json.dumps({"content": content}) + "\n\n"

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/job", response_model=ChatJobOut)
def create_chat_job(
    body: ChatJobRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Queue a message for a remote GPU worker (no tunnel required)."""
    settings = get_settings()
    model = body.model or settings.default_model

    conv_id = body.conversation_id
    if conv_id:
        conv = db.query(Conversation).filter(
            Conversation.id == conv_id, Conversation.user_id == user.id
        ).first()
        if not conv:
            raise HTTPException(404, "Conversation not found")
    else:
        conv = Conversation(user_id=user.id, title=body.message.strip()[:60] or "New chat", model=model)
        db.add(conv)
        db.commit()
        db.refresh(conv)
        conv_id = conv.id

    history = [
        {"role": m.role, "content": m.content}
        for m in conv.messages
    ]

    # Embed any attached images into the user message as markdown data URLs.
    # The worker extracts them and sends them to a vision model.
    content = body.message
    for img in (body.images or []):
        content += "\n\n![image](" + img + ")"
    db.add(Message(conversation_id=conv.id, role="user", content=content))
    job = ChatJob(
        id=str(uuid.uuid4()),
        user_id=user.id,
        conversation_id=conv.id,
        prompt=content,
        history=json.dumps(history),
        model=model,
        want_zip=body.want_zip,
        status="pending",
    )
    db.add(job)
    db.flush()

    # Web search runs for every message by default (ChatGPT-style browsing);
    # the UI can opt out per-message with search:false. Results are stored so
    # the worker injects them as context and the frontend renders a clickable
    # Sources list next to the reply.
    do_search = True
    if body.search is False:
        do_search = False
    if do_search:
        try:
            sources = search_web(content, max_results=6, timeout=12)
        except Exception:
            sources = []
        for i, s in enumerate(sources):
            db.add(SearchResult(
                job_id=job.id, rank=i,
                title=(s.get("title") or "")[:1000],
                url=(s.get("url") or "")[:2000],
                snippet=(s.get("snippet") or "")[:1000],
            ))

    db.commit()
    db.refresh(job)
    return ChatJobOut(job_id=job.id, conversation_id=conv.id, status=job.status)


@router.get("/job/{job_id}", response_model=ChatJobOut)
def get_chat_job(
    job_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Poll a queued job's status from the frontend."""
    job = db.query(ChatJob).filter(
        ChatJob.id == job_id, ChatJob.user_id == user.id
    ).first()
    if not job:
        raise HTTPException(404, "Job not found")
    sources = [
        SearchSourceOut(title=r.title, url=r.url, snippet=r.snippet)
        for r in db.query(SearchResult)
        .filter(SearchResult.job_id == job.id)
        .order_by(SearchResult.rank.asc())
        .all()
    ]
    return ChatJobOut(
        job_id=job.id,
        conversation_id=job.conversation_id,
        status=job.status,
        response=job.response,
        error=job.error,
        has_zip=bool(job.zip_b64),
        sources=sources,
    )


@router.get("/job/{job_id}/zip")
def download_job_zip(
    job_id: str,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Download the project archive produced for a finished job (owner only)."""
    job = db.query(ChatJob).filter(
        ChatJob.id == job_id, ChatJob.user_id == user.id
    ).first()
    if not job:
        raise HTTPException(404, "Job not found")
    if not job.zip_b64:
        raise HTTPException(404, "No project archive for this job")
    try:
        data = base64.b64decode(job.zip_b64)
    except Exception:
        raise HTTPException(400, "Stored archive is corrupt")
    name = (job.zip_name or "project.zip").replace('"', "").replace("\n", "")
    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )
