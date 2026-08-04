# ==================================================
#  NEXTGEN AI - FREE GPU RUNNER (Google Colab)
#  Runs your model on Colab's free T4 GPU.
#  Steps:
#    1. Open https://colab.research.google.com
#    2. File -> New notebook
#    3. Paste ALL of this into the single code cell
#    4. Edit VERCELL_URL / ADMIN creds below if needed
#    5. Press Play (runtime -> Run all)
#    6. Keep this tab open & connected (it stops ~12h)
#  ==================================================
import json, os, re, subprocess, time, urllib.request

VERCEL_URL    = "https://nextgen-web-eta.vercel.app"  # <- your Vercel app
ADMIN_USERNAME = "admin"                              # <- your admin username
ADMIN_PASSWORD = "admin12345"                         # <- your admin password
MODEL         = "qwen3:14b"   # nextgen-trained base; qwen3:8b = faster
TUNNEL_HOST   = "localhost"   # keep as-is

def sh(cmd, silent=True):
    if not silent: print(">", cmd[:140])
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1200)
        if r.returncode != 0 and not silent: print(r.stderr[-400:])
        return r.returncode == 0
    except Exception as e:
        print("cmd failed:", e); return False

def post(path, data, token=None):
    req = urllib.request.Request(
        VERCEL_URL + path, data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"})
    if token: req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=30) as r: return json.loads(r.read())

print("NEXTGEN AI GPU runner starting...")

import os as _os
if _os.path.exists("/dev/nvidia0"):
    print("GPU detected.")
else:
    print("WARNING: no NVIDIA GPU detected - using CPU (slow). Runtime -> Change runtime type -> T4 GPU.")

# 1) Install Ollama (free, from ollama.com)
print("[1/5] Installing Ollama...")
if not os.path.exists("/usr/local/bin/ollama"):
    sh("curl -fsSL https://ollama.com/install.sh | sh")

# 2) Install cloudflared tunnel (free)
print("[2/5] Installing tunnel...")
if not os.path.exists("/usr/local/bin/cloudflared"):
    sh("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared")
    sh("chmod +x /usr/local/bin/cloudflared")

# 3) Build nextgen-trained = MODEL + your Modelfile (no big uploads needed)
print("[3/5] Pulling %s and creating nextgen-trained..." % MODEL)
sh("ollama pull " + MODEL)
modelfile = '''FROM %s

SYSTEM "You are NEXTGEN AI v20, a fully autonomous AI software engineering operating system. Provide concise, actionable responses. Write complete, working code. Use markdown for formatting."

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
''' % MODEL
with open("/content/Modelfile", "w") as f: f.write(modelfile)
sh("ollama create nextgen-trained -f /content/Modelfile")

# 4) Start Ollama server + tunnel
print("[4/5] Starting Ollama + tunnel...")
subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(8)
subprocess.Popen(
    ["cloudflared", "tunnel", "--url", "http://%s:11434" % TUNNEL_HOST, "--no-autoupdate"],
    stdout=open("/content/tunnel.log", "w"), stderr=subprocess.STDOUT)

def get_tunnel_url():
    for _ in range(90):
        try:
            log = open("/content/tunnel.log").read()
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", log)
            if m: return m.group(0)
        except Exception: pass
        time.sleep(3)
    return None

url = get_tunnel_url()
print("[5/5] Tunnel URL:", url)
if not url:
    print("FATAL: no tunnel URL appeared. Check cloudflared install."); raise SystemExit(1)

def tunnel_works(u):
    try:
        req = urllib.request.Request(
            u + "/api/chat",
            data=json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "ping"}],
                             "stream": False, "options": {"num_predict": 2}}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as r:
            return bool(json.loads(r.read()).get("message"))
    except Exception as e:
        print("tunnel check failed:", e)
        return False

print("Validating tunnel -> Ollama...")
ok = False
for i in range(5):
    if tunnel_works(url):
        ok = True
        break
    print("retry %d/5..." % (i + 1))
    time.sleep(10)
if not ok:
    print("ERROR: tunnel is up but Ollama is not reachable through it (Cloudflare edge 403/SSE issue).")
    print("Paste your ngrok authtoken as NGROK_AUTHTOKEN in a cell and rerun, or ask the assistant.")
    raise SystemExit(1)

# 5) Tell Vercel the new tunnel URL (auto-updates, no dashboard needed)
try:
    tok = post("/api/auth/login", {"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD})["access_token"]
    post("/api/admin/ollama_url", {"url": url}, token=tok)
    print("Registered tunnel with Vercel. App live:", VERCEL_URL)
except Exception as e:
    print("Could not auto-register:", e, "(set OLLAMA_URL manually in Vercel)")

print("\nKEEP THIS TAB OPEN. It stops after ~12h; just press Play again.")
print("Monitoring tunnel...")

# 6) Keep alive: restart tunnel if it dies, re-register URL
while True:
    time.sleep(60)
    try:
        urllib.request.urlopen(url + "/api/version", timeout=10)
    except Exception:
        print("Tunnel died, restarting...")
        subprocess.Popen(
            ["cloudflared", "tunnel", "--url", "http://%s:11434" % TUNNEL_HOST, "--no-autoupdate"],
            stdout=open("/content/tunnel.log", "w"), stderr=subprocess.STDOUT)
        new_url = get_tunnel_url()
        if new_url and new_url != url:
            url = new_url
            try:
                tok = post("/api/auth/login", {"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD})["access_token"]
                post("/api/admin/ollama_url", {"url": url}, token=tok)
                print("New tunnel registered:", url)
            except Exception: pass
