"""Pydantic request/response schemas."""
from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field


class RegisterRequest(BaseModel):
    email: EmailStr
    username: str = Field(min_length=3, max_length=50)
    password: str = Field(min_length=6, max_length=128)


class LoginRequest(BaseModel):
    username: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: "UserOut"


class UserOut(BaseModel):
    id: int
    email: str
    username: str
    is_admin: bool
    is_active: bool

    class Config:
        from_attributes = True


class UserAdminUpdate(BaseModel):
    is_admin: Optional[bool] = None
    is_active: Optional[bool] = None
    password: Optional[str] = None


class ConversationOut(BaseModel):
    id: int
    title: str
    model: Optional[str]
    created_at: str

    class Config:
        from_attributes = True


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: Optional[int] = None
    model: Optional[str] = None
    system_prompt: Optional[str] = None


class ChatJobRequest(BaseModel):
    message: str = Field(min_length=1)
    conversation_id: Optional[int] = None
    model: Optional[str] = None
    images: Optional[List[str]] = None  # base64 data URLs, client-compressed
    want_zip: bool = False              # ask the worker to build a project .zip
    search: Optional[bool] = None       # True=force web search, False=off, None=auto


class SearchSourceOut(BaseModel):
    title: str = ""
    url: str = ""
    snippet: str = ""


class ChatJobOut(BaseModel):
    job_id: str
    conversation_id: Optional[int] = None
    status: str
    response: Optional[str] = None
    error: Optional[str] = None
    has_zip: bool = False
    sources: List[SearchSourceOut] = []


class WorkerCompleteIn(BaseModel):
    job_id: str
    response: Optional[str] = None
    error: Optional[str] = None
    zip_b64: Optional[str] = None
    zip_name: Optional[str] = None


class MessageOut(BaseModel):
    id: int
    role: str
    content: str

    class Config:
        from_attributes = True


class ModelInfo(BaseModel):
    name: str
    size_gb: Optional[float] = None


class OllamaUrlIn(BaseModel):
    url: str


class OllamaStatus(BaseModel):
    reachable: bool
    message: str = ""


class SystemStatus(BaseModel):
    app: str
    version: str
    database: str
    ollama: OllamaStatus
    models: List[ModelInfo] = []
    total_users: int = 0
    total_conversations: int = 0
