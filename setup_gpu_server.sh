#!/bin/bash
# ============================================
# NEXTGEN AI — Cloud GPU Server Setup
# Run this on a RunPod/Vast.ai/Lambda GPU server
# ============================================
set -e

echo "=== NEXTGEN AI GPU Server Setup ==="
echo ""

# 1. Install Ollama
echo "[1/4] Installing Ollama..."
curl -fsSL https://ollama.com/install.sh | sh

# 2. Pull the base model
echo "[2/4] Pulling qwen3:14b base model (~9GB)..."
ollama pull qwen3:14b

# 3. Create the trained model with your system prompt
echo "[3/4] Creating nextgen-trained model..."
cat > /tmp/Modelfile << 'MODELEOF'
FROM qwen3:14b

SYSTEM "You are NEXTGEN AI v20, a fully autonomous AI software engineering operating system. Provide concise, actionable responses. Write complete, working code. Use markdown for formatting."

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 8192

TEMPLATE "{{- if .System }}<|im_start|>system
{{ .System }}
{{- end }}
{{- range .Messages }}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}
{{- else if eq .Role "assistant" }}<|im_start|>assistant
{{ .Content }}
{{- end }}
{{- end }}<|im_start|>assistant
"
MODELEOF

ollama create nextgen-trained -f /tmp/Modelfile

# 4. Start the server on 0.0.0.0 (all interfaces)
echo "[4/4] Starting Ollama server on 0.0.0.0:11434..."
echo ""
echo "============================================"
echo "  Server ready!"
echo "  Public URL: http://$(curl -s ifconfig.me):11434"
echo "============================================"
echo ""
echo "Set this URL as OLLAMA_URL in your Vercel dashboard."
echo "Press Ctrl+C to stop the server."
echo ""

ollama serve --host 0.0.0.0 --port 11434
