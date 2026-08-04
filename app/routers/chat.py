"""Chat endpoints: create/list conversations, stream responses via SSE."""
import json

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

from ..auth import get_current_user
from ..config import get_settings
from ..database import get_db, get_ollama_url
from ..models import Conversation, Message, User
from ..schemas import ChatRequest, ConversationOut, MessageOut
from ..services.ollama import OllamaClient

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
