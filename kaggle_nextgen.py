# ==================================================
#  NEXTGEN AI - FREE GPU RUNNER (Kaggle)
#  Runs your model on Kaggle's free P100/T4 GPU.
#  ~30 GPU hrs/week free; sessions last ~9h.
#  ==================================================
import json, os, re, subprocess, time, urllib.request

VERCEL_URL     = "https://nextgen-web-eta.vercel.app"  # <- your Vercel app
ADMIN_USERNAME = "admin"                               # <- your admin username
ADMIN_PASSWORD = "admin12345"                          # <- your admin password
MODEL          = "qwen3:14b"   # nextgen-trained base; qwen3:8b = faster
TUNNEL_HOST    = "localhost"
OLLAMA_BIN     = "/usr/local/bin/ollama"
WORKDIR        = "/kaggle/working"
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

def http(url, data=None, timeout=30):
    h = {"User-Agent": BROWSER_UA}
    if data is not None:
        h["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=(json.dumps(data).encode() if data is not None else None), headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read()
        return json.loads(body) if body else {}

def post(path, data, token=None):
    req = urllib.request.Request(
        VERCEL_URL + path, data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json", "User-Agent": BROWSER_UA})
    if token: req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=30) as r: return json.loads(r.read())

os.makedirs(WORKDIR, exist_ok=True)
print("NEXTGEN AI Kaggle GPU runner starting...")

# 0) Hard GPU check
print("Checking GPU...")
try:
    gpu = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True)
    print(gpu.stdout.strip() or "no GPU")
except Exception as e:
    print("nvidia-smi failed:", e)

# 1) Install Ollama
print("[1/5] Installing Ollama...")
if not os.path.exists(OLLAMA_BIN):
    print("Installing zstd (needed to unpack ollama)...")
    sh("apt-get update -qq && apt-get install -y -qq zstd openssh-client", silent=False, timeout=600)
    if not (os.path.exists("/usr/bin/zstd") or os.path.exists("/bin/zstd")):
        print("WARNING: zstd binary not found, unpacking may fail")
    print("Downloading ollama (about 1.4 GB, a couple minutes)...")
    sh("curl -fsSL -o /tmp/ollama.tar.zst https://ollama.com/download/ollama-linux-amd64.tar.zst", silent=False, timeout=3600)
    print("Unpacking to /usr/local...")
    sh("zstd -d -f /tmp/ollama.tar.zst -o /tmp/ollama.tar", silent=False, timeout=600)
    sh("tar -xf /tmp/ollama.tar -C /usr/local", silent=False, timeout=600)
    sh("chmod +x " + OLLAMA_BIN)
if not os.path.exists(OLLAMA_BIN):
    print("FATAL: ollama binary missing at", OLLAMA_BIN); raise SystemExit(1)
sh(OLLAMA_BIN + " --version", silent=False)
print("Ollama installed.")

# 2) Install cloudflared (primary tunnel)
print("[2/5] Installing tunnel...")
CF_BIN = "/usr/local/bin/cloudflared"
if not os.path.exists(CF_BIN):
    sh("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O " + CF_BIN, silent=False)
    sh("chmod +x " + CF_BIN)
if not os.path.exists(CF_BIN):
    print("FATAL: cloudflared missing at", CF_BIN); raise SystemExit(1)
print("cloudflared installed.")

# 3) Start Ollama server first (pull/create need the daemon running)
print("[3/5] Starting Ollama server...")
subprocess.Popen([OLLAMA_BIN, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
server_ok = False
for _ in range(30):
    try:
        http("http://localhost:11434/api/version", timeout=3)
        server_ok = True
        break
    except Exception:
        time.sleep(2)
if not server_ok:
    print("FATAL: ollama server did not start."); raise SystemExit(1)
print("Ollama serving on :11434")

print("Pulling %s (about 9 GB, takes a few minutes)..." % MODEL)
if not sh(OLLAMA_BIN + " pull " + MODEL, silent=False):
    print("FATAL: model pull failed."); raise SystemExit(1)
modelfile = '''FROM %s

SYSTEM "You are NEXTGEN AI v20, a fully autonomous AI software engineering operating system. Provide concise, actionable responses. Write complete, working code. Use markdown for formatting."

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
''' % MODEL
with open(WORKDIR + "/Modelfile", "w") as f: f.write(modelfile)
if not sh(OLLAMA_BIN + " create nextgen-trained -f " + WORKDIR + "/Modelfile", silent=False):
    print("FATAL: could not create nextgen-trained."); raise SystemExit(1)
print("nextgen-trained created.")

# 4) Start tunnel: cloudflared -> localhost.run -> serveo -> ngrok
print("[4/5] Starting tunnel...")
TUNNEL_LOG = WORKDIR + "/tunnel.log"

def start_cloudflared():
    subprocess.Popen([CF_BIN, "tunnel", "--url", "http://%s:11434" % TUNNEL_HOST, "--no-autoupdate"],
                     stdout=open(TUNNEL_LOG, "w"), stderr=subprocess.STDOUT)
    for _ in range(60):
        try:
            log = open(TUNNEL_LOG).read()
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", log)
            if m: return m.group(0)
        except Exception:
            pass
        time.sleep(3)
    return None

def start_ssh_tunnel(host_arg):
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
           "-o", "ServerAliveInterval=60", "-o", "ServerAliveCountMax=3",
           "-o", "ExitOnForwardFailure=yes", "-N", "-R", "80:localhost:11434", host_arg]
    subprocess.Popen(cmd, stdout=open(TUNNEL_LOG, "w"), stderr=subprocess.STDOUT)
    for _ in range(45):
        try:
            log = open(TUNNEL_LOG).read()
            m = re.search(r"https://[a-z0-9-]+\.(lhr\.life|serveo\.net)", log)
            if m: return m.group(0)
        except Exception:
            pass
        time.sleep(3)
    return None

def start_tunnel():
    print("Trying cloudflared...")
    url = start_cloudflared()
    if url: return url
    print("cloudflared failed; trying localhost.run...")
    url = start_ssh_tunnel("nokey@localhost.run")
    if url: return url
    print("localhost.run failed; trying serveo.net...")
    subprocess.run(["pkill", "-f", "localhost.run"], capture_output=True)
    time.sleep(2)
    url = start_ssh_tunnel("serveo.net")
    if url: return url
    print("all free tunnels failed; trying ngrok (set NGROK_AUTHTOKEN)...")
    return None

url = start_tunnel()
print("[5/5] Tunnel URL:", url)
if not url:
    print("FATAL: no tunnel URL appeared. All tunnel providers failed.")
    raise SystemExit(1)

def tunnel_works(u):
    try:
        r = http(u + "/api/chat",
                 {"model": MODEL, "messages": [{"role": "user", "content": "ping"}],
                  "stream": False, "options": {"num_predict": 2}},
                 timeout=180)
        return bool(r.get("message"))
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
    print("ERROR: tunnel is up but Ollama is not reachable through it.")
    raise SystemExit(1)

# 5) Tell Vercel the new tunnel URL (auto-updates, no dashboard needed)
def register(u):
    try:
        tok = post("/api/auth/login", {"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD})["access_token"]
        post("/api/admin/ollama_url", {"url": u}, token=tok)
        print("Registered tunnel with Vercel. App live:", VERCEL_URL)
    except Exception as e:
        print("Could not auto-register:", e, "(set OLLAMA_URL manually in Vercel)")

register(url)

print("\nKEEP THIS SESSION OPEN. ~9h limit; rerun to restart.")
print("Monitoring tunnel...")

# 6) Keep alive: restart tunnel if it dies, re-register URL
while True:
    time.sleep(60)
    try:
        http(url + "/api/version", timeout=10)
    except Exception:
        print("Tunnel died, restarting...")
        subprocess.run(["pkill", "-f", "cloudflared"], capture_output=True)
        subprocess.run(["pkill", "-f", "ssh"], capture_output=True)
        time.sleep(3)
        new_url = start_tunnel()
        if new_url and new_url != url:
            url = new_url
            register(url)
            print("New tunnel:", url)
