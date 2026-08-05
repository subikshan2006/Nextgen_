# ==================================================
#  NEXTGEN AI - FREE GPU WORKER (Google Colab)
#  Runs your model on Colab's free T4 GPU.
#  The worker POLLS the deployed site for chat jobs
#  (no tunnel needed - Colab only makes outbound HTTPS).
#  Usage:
#    Open https://colab.research.google.com/github/subikshan2006/Nextgen_/blob/main/colab_nextgen.ipynb
#    Then Runtime -> Run all
#  Keep this tab open & connected (session stops ~12h).
#  ==================================================
import json, os, subprocess, time, urllib.request

VERCEL_URL     = "https://nextgen-web-eta.vercel.app"  # <- your Vercel app
ADMIN_USERNAME = "admin"                               # <- your admin username
ADMIN_PASSWORD = "admin12345"                          # <- your admin password
BASE_MODEL     = "qwen3:14b"   # model to pull; qwen3:8b = faster
WORKER_MODEL   = "nextgen-trained"  # the trained model name served to the site
OLLAMA_URL     = "http://localhost:11434"
OLLAMA_BIN     = "/usr/local/bin/ollama"
BROWSER_UA     = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

def sh(cmd, silent=True, timeout=1800):
    if not silent: print(">", cmd[:140])
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0 and not silent:
            print("stderr:", r.stderr[-600:])
        return r.returncode == 0
    except Exception as e:
        print("cmd failed:", e); return False

def http(url, data=None, timeout=30, token=None):
    h = {"User-Agent": BROWSER_UA}
    if data is not None:
        h["Content-Type"] = "application/json"
    if token:
        h["Authorization"] = "Bearer " + token
    req = urllib.request.Request(url, data=(json.dumps(data).encode() if data is not None else None), headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
        return json.loads(body) if body else {}

print("NEXTGEN AI GPU worker starting...")

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
print("[1/4] Installing Ollama...")
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

# 2) Start Ollama server first (pull/create need the daemon running)
print("[2/4] Starting Ollama server...")
subprocess.Popen([OLLAMA_BIN, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
server_ok = False
for _ in range(30):
    try:
        http(OLLAMA_URL + "/api/version", timeout=3)
        server_ok = True
        break
    except Exception:
        time.sleep(2)
if not server_ok:
    print("FATAL: ollama server did not start."); raise SystemExit(1)
print("Ollama serving on :11434")

# 3) Pull the base model and create nextgen-trained
print("Pulling %s (about 9 GB, takes a few minutes)..." % BASE_MODEL)
if not sh(OLLAMA_BIN + " pull " + BASE_MODEL, silent=False):
    print("FATAL: model pull failed."); raise SystemExit(1)
modelfile = '''FROM %s

SYSTEM "You are NEXTGEN AI v20, a fully autonomous AI software engineering operating system. Provide concise, actionable responses. Write complete, working code. Use markdown for formatting."

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
''' % BASE_MODEL
with open("/content/Modelfile", "w") as f: f.write(modelfile)
if not sh(OLLAMA_BIN + " create " + WORKER_MODEL + " -f /content/Modelfile", silent=False):
    print("FATAL: could not create " + WORKER_MODEL); raise SystemExit(1)
print(WORKER_MODEL + " created.")

# 4) Worker loop: poll the site for jobs, generate replies, submit results
def login():
    r = http(VERCEL_URL + "/api/auth/login", {"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD}, timeout=30)
    return r["access_token"]

def poll_jobs(token):
    r = http(VERCEL_URL + "/api/worker/poll?limit=3", token=token, timeout=30)
    return r.get("jobs", [])

def heartbeat(token):
    http(VERCEL_URL + "/api/worker/heartbeat", {"t": 1}, token=token, timeout=15)

def complete(token, job_id, response=None, error=None):
    payload = {"job_id": job_id}
    if error:
        payload["error"] = error
    else:
        payload["response"] = response or ""
    http(VERCEL_URL + "/api/worker/complete", payload, token=token, timeout=30)

def ollama_chat(messages, model):
    data = {"model": model, "messages": messages, "stream": False,
            "options": {"num_ctx": 8192, "temperature": 0.7}}
    r = http(OLLAMA_URL + "/api/chat", data, timeout=900)
    return (r.get("message") or {}).get("content", "")

print("[4/4] Worker online. Polling for jobs every 3 seconds...")
print("KEEP THIS TAB OPEN. It stops after ~12h; just press Play again.")

token = None
beat_count = 0
while True:
    try:
        if token is None:
            token = login()
        beat_count += 1
        if beat_count % 20 == 0:
            heartbeat(token)   # every ~60s: mark the worker as online
        jobs = poll_jobs(token)
        for jb in jobs:
            jid = jb["job_id"]
            print("Job", jid[:8], "started...")
            try:
                text = ollama_chat(jb["messages"], jb.get("model") or WORKER_MODEL)
                complete(token, jid, response=text)
                print("Job", jid[:8], "done (%d chars)" % len(text))
            except Exception as e:
                print("Job", jid[:8], "failed:", e)
                try:
                    complete(token, jid, error=str(e)[:500])
                except Exception:
                    pass
    except Exception as e:
        print("poll error:", e)
        token = None  # force re-login next round
        time.sleep(5)
    time.sleep(3)
