@echo off
REM Expose your local Ollama (nextgen-trained) to the deployed web app.
REM Requires ngrok authtoken:  ngrok config add-authtoken YOUR_TOKEN   (one time, free at ngrok.com)
REM Then run this file and copy the printed "Forwarding https://...." URL into Vercel's OLLAMA_URL env var.
where cloudflared >nul 2>&1
if not errorlevel 1 (
  echo [NEXTGEN] Starting Cloudflare quick tunnel to Ollama...
  cloudflared tunnel --url http://localhost:11434 --protocol http2 --no-autoupdate
  goto :eof
)
echo [NEXTGEN] cloudflared not found. Installing...
winget install --id Cloudflare.cloudflared --accept-source-agreements --accept-package-agreements --silent
echo [NEXTGEN] Run this script again to start the tunnel.
