"""SQLAlchemy models: users, conversations, messages, api settings, chat jobs."""
import datetime
from sqlalchemy import (
    Column, Integer, String, Text, Boolean, DateTime, ForeignKey,
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    is_admin = Column(Boolean, default=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    last_login = Column(DateTime, nullable=True)

    conversations = relationship(
        "Conversation", back_populates="user", cascade="all, delete-orphan"
    )


class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True)
    title = Column(String(255), default="New chat")
    model = Column(String(255), default=None)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

    user = relationship("User", back_populates="conversations")
    messages = relationship(
        "Message", back_populates="conversation", order_by="Message.id",
        cascade="all, delete-orphan",
    )


class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), index=True)
    role = Column(String(20), nullable=False)  # system | user | assistant
    content = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")


class ApiSetting(Base):
    __tablename__ = "api_settings"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, index=True, nullable=False)
    value = Column(Text, nullable=True)
    updated_at = Column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )


class SearchResult(Base):
    """Web search results attached to a chat job, rendered as a Sources list."""
    __tablename__ = "search_results"

    id = Column(Integer, primary_key=True, index=True)
    job_id = Column(String(64), ForeignKey("chat_jobs.id"), index=True, nullable=False)
    rank = Column(Integer, default=0)
    title = Column(Text, default="")
    url = Column(Text, default="")
    snippet = Column(Text, default="")


class ChatJob(Base):
    """A queued chat request that a remote GPU worker picks up and completes.

    The deployed site cannot reach a GPU behind NAT, so the GPU machine
    *polls* for pending jobs instead of exposing a public tunnel.
    """
    __tablename__ = "chat_jobs"

    id = Column(String(64), primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), index=True, nullable=False)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), index=True, nullable=True)
    prompt = Column(Text, nullable=False)
    history = Column(Text, default="[]")  # JSON list of {role, content} before this prompt
    model = Column(String(255), default=None)
    status = Column(String(20), default="pending")  # pending | running | done | error
    response = Column(Text, nullable=True)
    error = Column(Text, nullable=True)
    want_zip = Column(Boolean, default=False)  # model asked to emit a project .zip
    zip_b64 = Column(Text, nullable=True)      # project archive (base64)
    zip_name = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )
    completed_at = Column(DateTime, nullable=True)
