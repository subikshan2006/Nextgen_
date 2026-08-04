# NEXTGEN AI v20 — Web App

ChatGPT-style web chat + opencode-style tool system + Claude-style reasoning.
Fully local AI using YOUR trained model (`nextgen-trained`) via Ollama.

## Architecture

- **Backend**: FastAPI (Python) — auth (JWT), conversations, SSE streaming chat,
  admin panel, model management.
- **Database**: SQLite locally, Neon Postgres on Vercel (auto-switches by
  `DATABASE_URL`).
- **Model**: Your own locally-trained model runs in Ollama on your machine.
  The deployed site reaches it through a tunnel (set `OLLAMA_URL`).

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000 — first registered user becomes admin.
Default admin (from env): `admin@nextgen.ai` / `admin12345`.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `DATABASE_URL` | `sqlite:///./nextgen.db` | Neon Postgres on Vercel |
| `JWT_SECRET` | dev-only | Sign tokens |
| `OLLAMA_URL` | `http://localhost:11434` | Your machine's Ollama (tunnel URL on Vercel) |
| `DEFAULT_MODEL` | `nextgen-trained` | Default chat model |
| `ADMIN_EMAIL` / `ADMIN_PASSWORD` | `admin@nextgen.ai` / `admin12345` | Bootstrap admin |

## Deploy to Vercel

```bash
npm i -g vercel
vercel --prod
```

Set env vars in Vercel dashboard (Project → Settings → Environment Variables):
`DATABASE_URL` (Neon), `JWT_SECRET`, `OLLAMA_URL` (tunnel), `DEFAULT_MODEL`.

## Expose your local Ollama

Cloudflare Tunnel (free):

```bash
cloudflared tunnel --url http://localhost:11434
```

Copy the `https://*.trycloudflare.com` URL into `OLLAMA_URL` on Vercel.
