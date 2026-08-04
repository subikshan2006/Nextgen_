# ==================================================
#  NEXTGEN AI - FREE GPU RUNNER (Google Colab)
#  Runs your model on Colab's free T4 GPU.
#  Usage (only 2 lines in the cell):
#    !wget -q -O /content/nextgen.py https://raw.githubusercontent.com/subikshan2006/Nextgen_/main/colab_nextgen.py
#    exec(open('/content/nextgen.py').read())
#
#  Requirements: Runtime -> Change runtime type -> T4 GPU
#  Keep this tab open & connected (session stops ~12h).
#  ==================================================
import json, os, re, subprocess, time, urllib.request

VERCEL_URL     = "https://nextgen-web-eta.vercel.app"  # <- your Vercel app
ADMIN_USERNAME = "admin"                               # <- your admin username
ADMIN_PASSWORD = "admin12345"                          # <- your admin password
MODEL          = "qwen3:14b"   # nextgen-trained base; qwen3:8b = faster
TUNNEL_HOST    = "localhost"
OLLAMA_BIN     = "/usr/local/bin/ollama"
CF_BIN         = "/usr/local/bin/cloudflared"

def sh(cmd, silent=True, timeout=1800):
    if not silent: print(">", cmd[:140])
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0 and not silent:
            print("stderr:", r.stderr[-600:])
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

# 0) Hard GPU check - do not waste time on CPU
print("Checking GPU...")
def have_gpu():
    if any(os.path.exists(p) for p in ("/dev/nvidia0", "/dev/nvidia1")):
        return True
    try:
        return subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).returncode == 0
    except Exception:
        return False

if not have_gpu():
    print("NO NVIDIA GPU DETECTED.")
    print("Menu: Runtime -> Change runtime type -> Hardware accelerator: T4 GPU -> Save")
    print("Then press the play button again.")
    raise SystemExit(1)
print("GPU OK.")

# ensure /usr/local/bin on PATH for subprocess
os.environ["PATH"] = "/usr/local/bin:" + os.environ.get("PATH", "")

# 1) Install Ollama (free, from ollama.com)
print("[1/5] Installing Ollama...")
if not os.path.exists(OLLAMA_BIN):
    print("Installing zstd (needed to unpack ollama)...")
    sh("apt-get update -qq && apt-get install -y -qq zstd", silent=False, timeout=600)
    if not (os.path.exists("/usr/bin/zstd") or os.path.exists("/bin/zstd")):
        print("WARNING: zstd binary not found, unpacking may fail")
    print("Downloading ollama (about 1.4 GB, a couple minutes)...")
    sh("curl -fsSL -o /tmp/ollama.tar.zst https://ollama.com/download/ollama-linux-amd64.tar.zst", silent=False, timeout=3600)
    print("Unpacking to /usr/local...")
    sh("zstd -d -f /tmp/ollama.tar.zst -o /tmp/ollama.tar", silent=False, timeout=600)
    sh("tar -xf /tmp/ollama.tar -C /usr/local", silent=False, timeout=600)
    sh("chmod +x " + OLLAMA_BIN)
if not os.path.exists(OLLAMA_BIN):
    print("FATAL: ollama binary missing at", OLLAMA_BIN)
    sh("ls -la /usr/local/bin 2>/dev/null | grep -i ollama; ls -la /tmp | grep ollama", silent=False)
    raise SystemExit(1)
sh(OLLAMA_BIN + " --version", silent=False)
print("Ollama installed.")

# 2) Install cloudflared tunnel (free)
print("[2/5] Installing tunnel...")
if not os.path.exists(CF_BIN):
    sh("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O " + CF_BIN, silent=False)
    sh("chmod +x " + CF_BIN)
if not os.path.exists(CF_BIN):
    print("FATAL: cloudflared missing at", CF_BIN); raise SystemExit(1)
print("cloudflared installed.")

# 3) Pull model + build nextgen-trained
print("[3/5] Pulling %s (about 9 GB, takes a few minutes)..." % MODEL)
if not sh(OLLAMA_BIN + " pull " + MODEL, silent=False):
    print("FATAL: model pull failed."); raise SystemExit(1)
modelfile = '''FROM %s

SYSTEM "You are NEXTGEN AI v20, a fully autonomous AI software engineering operating system. Provide concise, actionable responses. Write complete, working code. Use markdown for formatting."

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
''' % MODEL
with open("/content/Modelfile", "w") as f: f.write(modelfile)
if not sh(OLLAMA_BIN + " create nextgen-trained -f /content/Modelfile", silent=False):
    print("FATAL: could not create nextgen-trained."); raise SystemExit(1)
print("nextgen-trained created.")

# 4) Start Ollama server + tunnel
print("[4/5] Starting Ollama + tunnel...")
subprocess.Popen([OLLAMA_BIN, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for _ in range(20):
    try:
        urllib.request.urlopen("http://localhost:11434/api/version", timeout=3); break
    except Exception:
        time.sleep(2)
print("Ollama serving on :11434")
subprocess.Popen(
    [CF_BIN, "tunnel", "--url", "http://%s:11434" % TUNNEL_HOST, "--no-autoupdate"],
    stdout=open("/content/tunnel.log", "w"), stderr=subprocess.STDOUT)

def get_tunnel_url():
    for _ in range(90):
        try:
            log = open("/content/tunnel.log").read()
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", log)
            if m: return m.group(0)
        except Exception:
            pass
        time.sleep(3)
    return None

url = get_tunnel_url()
print("[5/5] Tunnel URL:", url)
if not url:
    print("FATAL: no tunnel URL appeared. Check cloudflared."); raise SystemExit(1)

def tunnel_works(u):
    try:
        req = urllib.request.Request(
            u + "/api/chat",
            data=json.dumps({"model": MODEL, "messages": [{"role": "user", "content": "ping"}],
                             "stream": False, "options": {"num_predict": 2}}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=180) as r:
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
            [CF_BIN, "tunnel", "--url", "http://%s:11434" % TUNNEL_HOST, "--no-autoupdate"],
            stdout=open("/content/tunnel.log", "w"), stderr=subprocess.STDOUT)
        new_url = get_tunnel_url()
        if new_url and new_url != url:
            url = new_url
            try:
                tok = post("/api/auth/login", {"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD})["access_token"]
                post("/api/admin/ollama_url", {"url": url}, token=tok)
                print("New tunnel registered:", url)
            except Exception:
                pass
