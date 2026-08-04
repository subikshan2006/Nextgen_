"""Async client for the Ollama API (talks to your own trained model).

The deployed Vercel site calls back to your machine's Ollama via a tunnel URL
set in OLLAMA_URL. Local dev uses http://localhost:11434.
"""
import json
from typing import AsyncGenerator, Dict, List, Optional

import aiohttp

from ..config import get_settings


class OllamaClient:
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or get_settings().ollama_url).rstrip("/")

    async def _get(self, path: str):
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as s:
            async with s.get(f"{self.base_url}{path}") as r:
                r.raise_for_status()
                return await r.json()

    async def list_models(self) -> List[Dict]:
        try:
            data = await self._get("/api/tags")
            models = []
            for m in data.get("models", []):
                models.append({
                    "name": m.get("name", ""),
                    "size_gb": round((m.get("size", 0) or 0) / 1e9, 2),
                })
            return models
        except Exception:
            return []

    async def check(self) -> bool:
        try:
            await self._get("/api/tags")
            return True
        except Exception:
            return False

    async def chat(
        self,
        messages: List[Dict],
        model: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        on_token=None,
    ) -> AsyncGenerator[str, None]:
        """Stream a chat completion. Yields tokens; calls on_token(tok) too."""
        model = model or get_settings().default_model
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600, sock_read=300)
        ) as s:
            async with s.post(f"{self.base_url}/api/chat", json=payload) as r:
                if r.status != 200:
                    body = await r.text()
                    raise RuntimeError(f"Ollama error {r.status}: {body[:500]}")
                async for line in r.content:
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except (json.JSONDecodeError, ValueError):
                        continue
                    msg = chunk.get("message", {})
                    token = msg.get("content") or msg.get("thinking") or ""
                    if token:
                        if on_token:
                            on_token(token)
                        yield token
                    if chunk.get("done"):
                        break
