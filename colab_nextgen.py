# ==================================================
#  NEXTGEN AI - FREE GPU RUNNER (Google Colab)
#  Runs your model on Colab's free T4 GPU.
#  Usage:
#    Open https://colab.research.google.com/github/subikshan2006/Nextgen_/blob/main/colab_nextgen.ipynb
#    Then Runtime -> Run all
#  Keep this tab open & connected (session stops ~12h).
#  ==================================================
import json, os, re, subprocess, time, urllib.request

VERCEL_URL     = "https://nextgen-web-eta.vercel.app"  # <- your Vercel app
ADMIN_USERNAME = "admin"                               # <- your admin username
ADMIN_PASSWORD = "admin12345"                          # <- your admin password
MODEL          = "qwen3:14b"   # nextgen-trained base; qwen3:8b = faster
TUNNEL_HOST    = "localhost"
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

# 2) Install openssh-client for the free no-account tunnel
print("[2/5] Installing tunnel tool (openssh)...")
sh("apt-get install -y -qq openssh-client", silent=False, timeout=600)

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
with open("/content/Modelfile", "w") as f: f.write(modelfile)
if not sh(OLLAMA_BIN + " create nextgen-trained -f /content/Modelfile", silent=False):
    print("FATAL: could not create nextgen-trained."); raise SystemExit(1)
print("nextgen-trained created.")

# 4) Start a free no-account tunnel (localhost.run -> serveo -> ngrok)
print("[4/5] Starting tunnel...")
TUNNEL_LOG = "/content/tunnel.log"

def start_ssh_tunnel(host_arg):
    cmd = ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
           "-o", "ServerAliveInterval=60", "-o", "ServerAliveCountMax=3",
           "-o", "ExitOnForwardFailure=yes", "-N", "-R", "80:localhost:11434", host_arg]
    subprocess.Popen(cmd, stdout=open(TUNNEL_LOG, "w"), stderr=subprocess.STDOUT)
    for _ in range(45):
        try:
            log = open(TUNNEL_LOG).read()
            m = re.search(r"https://[a-z0-9-]+\.(lhr\.life|serveo\.net)", log)
            if m:
                return m.group(0)
        except Exception:
            pass
        time.sleep(3)
    return None

def start_ngrok():
    tok = os.environ.get("NGROK_AUTHTOKEN", "").strip()
    if not tok:
        return None
    print("Using ngrok fallback...")
    if not os.path.exists("/tmp/ngrok"):
        sh("curl -fsSL -o /tmp/ngrok.tgz https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz", silent=False, timeout=600)
        sh("tar -xzf /tmp/ngrok.tgz -C /tmp", silent=False)
    sh("/tmp/ngrok config add-authtoken " + tok, silent=False)
    subprocess.Popen(["/tmp/ngrok", "http", "11434", "--log", "stdout"],
                     stdout=open("/content/ngrok.log", "w"), stderr=subprocess.STDOUT)
    for _ in range(40):
        try:
            for line in open("/content/ngrok.log"):
                if "https://" in line:
                    m = re.search(r"https://[a-z0-9-]+\.ngrok\.(io|app)", line)
                    if m:
                        return m.group(0)
        except Exception:
            pass
        time.sleep(2)
    return None

def start_tunnel():
    print("Trying localhost.run...")
    url = start_ssh_tunnel("nokey@localhost.run")
    if url:
        return url
    print("localhost.run failed; trying serveo.net...")
    subprocess.run(["pkill", "-f", "localhost.run"], capture_output=True)
    time.sleep(2)
    url = start_ssh_tunnel("serveo.net")
    if url:
        return url
    print("serveo failed; trying ngrok (set NGROK_AUTHTOKEN if you have one)...")
    subprocess.run(["pkill", "-f", "serveo.net"], capture_output=True)
    time.sleep(2)
    return start_ngrok()

url = start_tunnel()
print("[5/5] Tunnel URL:", url)
if not url:
    print("FATAL: no tunnel URL appeared. All tunnel providers failed.")
    print("Tip: create a free ngrok account and set NGROK_AUTHTOKEN in a cell:")
    print("  import os; os.environ['NGROK_AUTHTOKEN'] = 'your_token'")
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
    try:
        print("--- tunnel log tail ---")
        print(open(TUNNEL_LOG).read()[-800:])
    except Exception:
        pass
    print("Tip: create a free ngrok account and set NGROK_AUTHTOKEN in a cell, then rerun.")
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

print("\nKEEP THIS TAB OPEN. It stops after ~12h; just press Play again.")
print("Monitoring tunnel...")

# 6) Keep alive: restart tunnel if it dies, re-register URL
while True:
    time.sleep(60)
    try:
        http(url + "/api/version", timeout=10)
    except Exception:
        print("Tunnel died, restarting...")
        subprocess.run(["pkill", "-f", "ssh"], capture_output=True)
        time.sleep(3)
        new_url = start_tunnel()
        if new_url and new_url != url:
            url = new_url
            register(url)
            print("New tunnel:", url)
