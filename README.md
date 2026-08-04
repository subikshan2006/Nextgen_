# NEXTGEN AI v20 — Web App

ChatGPT-style web chat + opencode-style tool system + Claude-style reasoning.
Uses YOUR locally-trained model (`nextgen-trained`) via Ollama.

- **Backend**: FastAPI — auth (JWT), conversations, SSE streaming chat, admin panel.
- **Frontend**: login/register, streaming chat (markdown), admin dashboard.
- **Database**: SQLite locally, Neon Postgres on Vercel (auto-switch by `DATABASE_URL`).
- **Model**: runs on YOUR machine in Ollama. The deployed site reaches it through a tunnel.

## Run locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000 → first registered user becomes admin.
Bootstrap admin: `admin@nextgen.ai` / `admin12345`.

## Deploy (Vercel)

1. Create a Neon Postgres project → copy the `DATABASE_URL` (postgres://…).
2. Push this repo to GitHub.
3. In Vercel: import the repo. Set environment variables:

   | Variable | Example | Purpose |
   |---|---|---|
   | `DATABASE_URL` | `postgres://user:pass@…neon.tech/nextgen` | Neon Postgres |
   | `JWT_SECRET` | random long string | token signing |
   | `OLLAMA_URL` | tunnel URL (below) | reaches your PC's model |
   | `DEFAULT_MODEL` | `nextgen-trained` | default chat model |

4. Deploy. `vercel.json` routes everything to the FastAPI app.

> Note: Vercel Hobby serverless functions cap at 60 s. Responses are capped at
> ~1024 tokens to fit. For unlimited-length streaming, host the app on
> Railway/Render (long-running) and point the same code at the tunnel.

## Expose your local Ollama (one-time)

Install Cloudflare Tunnel, then run `start_tunnel.bat` — it prints a
`https://….trycloudflare.com` URL. Put that URL in `OLLAMA_URL`.

- The URL changes every restart — restart the tunnel after each PC reboot and
  update `OLLAMA_URL`.
- Quick tunnels don't proxy Server-Sent Events, so for reliable streaming use
  ngrok (`ngrok http 11434`, free authtoken) or a Cloudflare named tunnel
  (account + domain).
- Security: Ollama has no auth. Anyone with the URL can use your GPU. Use a
  random tunnel URL and don't share it; better: add a tunnel access rule.

## Project layout

```
app/
  main.py        FastAPI app + static hosting
  config.py      env settings
  database.py    engine/session bootstrap
  models.py      User / Conversation / Message / ApiSetting
  auth.py        bcrypt + JWT + admin dependency
  schemas.py     Pydantic models
  routers/
    auth.py      register / login / me
    chat.py      conversations + SSE streaming -> Ollama
    admin.py     users / models / status / settings
  services/
    ollama.py    async Ollama client (streaming)
static/
  index.html     chat UI (streaming, markdown)
  login.html     login / register
  admin.html     admin panel
  css/style.css
  js/common.js   API client + markdown renderer
```
