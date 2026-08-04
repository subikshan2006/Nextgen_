# ==================================================
#  NEXTGEN AI - FREE GPU RUNNER (Kaggle)
#  Same as Colab but runs on Kaggle's free GPU.
#  Steps:
#    1. Open kaggle.com -> New notebook
#    2. Settings -> Accelerator -> GPU (T4x2/P100)
#    3. Internet: ON
#    4. Paste this into a cell and Run
#  Kaggle gives ~30 GPU hrs/week; sessions last ~9h.
#  ==================================================
import json, os, re, subprocess, time, urllib.request

VERCEL_URL    = "https://nextgen-web-eta.vercel.app"
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin12345"
MODEL         = "qwen3:14b"   # qwen3:8b = faster
WORKDIR       = "/kaggle/working"
TUNNEL_HOST   = "localhost"

def sh(cmd, silent=True):
    if not silent: print(">", cmd[:140])
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=1800)
        if r.returncode != 0 and not silent: print(r.stderr[-400:])
        return r.returncode == 0
    except Exception as e:
        print("cmd failed:", e); return False

def post(path, data, token=None):
    req = urllib.request.Request(VERCEL_URL + path, data=json.dumps(data).encode(),
                                 headers={"Content-Type": "application/json"})
    if token: req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=30) as r: return json.loads(r.read())

os.makedirs(WORKDIR, exist_ok=True)
print("NEXTGEN AI Kaggle GPU runner starting...")

print("[1/5] Installing Ollama...")
if not os.path.exists("/usr/local/bin/ollama"):
    sh("curl -fsSL https://ollama.com/install.sh | sh")

print("[2/5] Installing tunnel...")
if not os.path.exists("/usr/local/bin/cloudflared"):
    sh("wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared")
    sh("chmod +x /usr/local/bin/cloudflared")

print("[3/5] Pulling %s..." % MODEL)
sh("ollama pull " + MODEL)
modelfile = '''FROM %s

SYSTEM "You are NEXTGEN AI v20, a fully autonomous AI software engineering operating system. Provide concise, actionable responses. Write complete, working code. Use markdown for formatting."

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 8192
''' % MODEL
with open(WORKDIR + "/Modelfile", "w") as f: f.write(modelfile)
sh("ollama create nextgen-trained -f " + WORKDIR + "/Modelfile")

print("[4/5] Starting Ollama + tunnel...")
subprocess.Popen(["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
time.sleep(8)
subprocess.Popen(["cloudflared", "tunnel", "--url", "http://%s:11434" % TUNNEL_HOST, "--no-autoupdate"],
                 stdout=open(WORKDIR + "/tunnel.log", "w"), stderr=subprocess.STDOUT)

def get_tunnel_url():
    for _ in range(90):
        try:
            log = open(WORKDIR + "/tunnel.log").read()
            m = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", log)
            if m: return m.group(0)
        except Exception: pass
        time.sleep(3)
    return None

url = get_tunnel_url()
print("[5/5] Tunnel URL:", url)

try:
    tok = post("/api/auth/login", {"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD})["access_token"]
    post("/api/admin/ollama_url", {"url": url}, token=tok)
    print("Registered tunnel with Vercel. App live:", VERCEL_URL)
except Exception as e:
    print("Could not auto-register:", e)

print("\nKEEP THIS SESSION OPEN. ~9h limit; rerun to restart.")
while True:
    time.sleep(60)
    try:
        urllib.request.urlopen(url + "/api/version", timeout=10)
    except Exception:
        print("Tunnel died, restarting...")
        subprocess.Popen(["cloudflared", "tunnel", "--url", "http://%s:11434" % TUNNEL_HOST, "--no-autoupdate"],
                         stdout=open(WORKDIR + "/tunnel.log", "w"), stderr=subprocess.STDOUT)
        new_url = get_tunnel_url()
        if new_url and new_url != url:
            url = new_url
            try:
                tok = post("/api/auth/login", {"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD})["access_token"]
                post("/api/admin/ollama_url", {"url": url}, token=tok)
                print("New tunnel registered:", url)
            except Exception: pass
