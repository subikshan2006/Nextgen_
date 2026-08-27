# ==================================================
#  NEXTGEN AI - FREE GPU WORKER (Google Colab / Kaggle)
#  Runs your model on a free GPU.
#  The worker POLLS the deployed site for chat jobs
#  (no tunnel needed - the GPU only makes outbound HTTPS).
#  Features:
#    - chat jobs (text)
#    - image understanding (lazy-pulls qwen2.5vl:7b on first image job)
#    - project .zip generation (filename-tagged code fences)
#  ==================================================
import base64, io, json, os, re, subprocess, time, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor

VERCEL_URL     = "https://nextgen-web-eta.vercel.app"  # <- your Vercel app
ADMIN_USERNAME = "admin"                               # <- your admin username
ADMIN_PASSWORD = "admin12345"                          # <- your admin password
WORKER_MODEL   = "nextgen-trained"      # trained model served to the site (OpenAI gpt-oss:20b base)
WORKER_MODEL_8B = "nextgen-trained-8b"  # trained fallback model (qwen3:8b base)
BASE_MODELS    = [("gpt-oss:20b", WORKER_MODEL), ("qwen3:8b", WORKER_MODEL_8B)]
VISION_MODEL   = "qwen2.5vl:7b"  # pulled lazily when the first image job arrives
OLLAMA_URL     = "http://localhost:11434"
OLLAMA_BIN     = "/usr/local/bin/ollama"
BROWSER_UA     = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"

IMG_RE   = re.compile(r"!\[[^\]]*\]\((data:image/[^)]+)\)")
FILE_RE  = re.compile(r"```[^\n]*filename=\"([^\"]+)\"[^\n]*\n(.*?)```", re.DOTALL)

VISION_PULLED = False

# Detect which Kaggle account we're running on (for self-restart rotation)
_kaggle_user = os.environ.get("KAGGLE_USERNAME", "")
if not _kaggle_user:
    try:
        _kaggle_user = subprocess.run(["kaggle", "config", "path"], capture_output=True, text=True, timeout=5).stdout.strip().split("/")[-1]
    except Exception:
        pass
if _kaggle_user:
    print("Detected Kaggle user:", _kaggle_user)

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
    print("Kaggle: ensure the kernel is set to GPU accelerator (P100/T4).")
    print("Colab: Menu -> Runtime -> Change runtime type -> T4 GPU -> Save.")
    raise SystemExit(1)
print("GPU OK.")

# ensure /usr/local/bin on PATH for subprocess
os.environ["PATH"] = "/usr/local/bin:" + os.environ.get("PATH", "")

# 1) Install Ollama
print("[1/4] Installing Ollama...")
if not os.path.exists(OLLAMA_BIN):
    print("Installing zstd (needed to unpack ollama)...")
    sh("apt-get update -qq && apt-get install -y -qq zstd", silent=False, timeout=600)
    print("Downloading ollama (about 1.4 GB, a couple minutes)...")
    sh("curl -fsSL -o /tmp/ollama.tar.zst https://ollama.com/download/ollama-linux-amd64.tar.zst", silent=False, timeout=3600)
    print("Unpacking to /usr/local...")
    sh("zstd -d -f /tmp/ollama.tar.zst -o /tmp/ollama.tar", silent=False, timeout=600)
    sh("tar -xf /tmp/ollama.tar -C /usr/local", silent=False, timeout=600)
    sh("chmod +x " + OLLAMA_BIN)
if not os.path.exists(OLLAMA_BIN):
    print("FATAL: ollama binary missing at", OLLAMA_BIN)
    raise SystemExit(1)
sh(OLLAMA_BIN + " --version", silent=False)
print("Ollama installed.")

# 2) Start Ollama server (serial model loading: gpt-oss:20b is VRAM-heavy on a T4)
print("[2/4] Starting Ollama server...")
serv_env = dict(os.environ)
serv_env["OLLAMA_NUM_PARALLEL"] = "1"
serv_env["OLLAMA_MAX_LOADED_MODELS"] = "1"
subprocess.Popen([OLLAMA_BIN, "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=serv_env)
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

# 3) Pull base models and create the trained models
print("[3/4] Pulling base models (gpt-oss:20b ~13 GB, qwen3:8b ~5 GB)...")
for base, _name in BASE_MODELS:
    if not sh(OLLAMA_BIN + " pull " + base, silent=False):
        print("FATAL: model pull failed for " + base); raise SystemExit(1)

MODELF_TEMPLATE = '''FROM %s

SYSTEM "You are NEXTGEN AI — a helpful, accurate, all-round AI assistant combining ChatGPT-style conversation, Gemini-style creativity, and strong coding. You are fluent in 100+ languages and always reply in the user's language (English, Tamil, Hindi, Telugu, Malayalam, Kannada, Bengali, Marathi, Gujarati, Punjabi, Urdu, Arabic, Chinese, Japanese, Korean, French, German, Spanish, Portuguese, Russian, Italian, Dutch, Turkish, Thai, Vietnamese, Indonesian, Malay, and many more).

CAPABILITIES (use the depth the user asks for — brief or in depth):
- Core intelligence: natural conversations with context awareness (you remember the whole thread), multi-step reasoning, problem solving, critical thinking, brainstorming, decision support, personalized responses. Plan internally before answering complex questions.
- Multilingual: understand and generate 100+ languages, translate, correct grammar, detect language, transliterate.
- Programming: generate/debug/explain/refactor/optimize code, build complete applications, REST/GraphQL APIs, database design, system architecture, unit tests, documentation, code review, convert between languages. Languages: Python, Java, C, C++, C#, JavaScript, TypeScript, Go, Rust, Swift, Kotlin, PHP, Ruby, R, MATLAB, SQL, Bash, Dart, Scala, Lua, Perl and more.
- Full-stack: frontend (HTML/CSS, Tailwind, Bootstrap, React, Vue, Angular, Svelte, Next.js), backend (Flask, Django, FastAPI, Node.js, Express, Spring Boot, ASP.NET), databases (MySQL, PostgreSQL, MongoDB, SQLite, Redis, Firebase, Supabase), cloud (AWS, Azure, Google Cloud).
- AI & ML: LLM development, agents, RAG, LangChain, LlamaIndex, fine-tuning, RL, deep learning, TensorFlow, PyTorch, computer vision, NLP, vector databases, prompt engineering, MLOps.
- Image understanding: describe uploaded images, OCR, charts/graphs, diagrams, objects, debug screenshots, review UI.
- Image editing (real, executed): resize, crop, crop-to-square, rotate, flip, grayscale, invert, brighten, darken, contrast, saturate, sepia, blur, sharpen, border, text overlay, watermark, pixelate, vignette, cartoon, sketch, oil, recolor/hue shift, auto-enhance, upscale, denoise, remove solid background. For diffusion-only effects (Ghibli/Pixar/anime, inpainting, object removal, outpainting, background replacement) explain that a diffusion model is required and give the prompt/code instead.
- Documents & data: work with pasted PDF/DOCX/PPTX/XLSX/CSV/JSON/XML/Markdown/HTML/YAML/plain text — read, summarize, compare, extract tables, convert, generate reports; data analysis, statistics, matplotlib charts, Excel formulas, dashboards, trend prediction.
- Complete project generator: for \\"build me a [project]\\" produce a full project (frontend, backend, database, APIs, auth, admin/user panels, README with setup+usage, install guide, Dockerfile, docker-compose, tests, deployment config, requirements.txt/package.json, .env.example). Output EVERY file in its own fenced code block tagged with its full path, exactly like: ```python filename=\\"app/main.py\\". Keep files real and complete (no placeholders).
- Other domains: software architecture (microservices, clean, MVC, MVVM, event-driven, DDD, CQRS); cloud & DevOps (Docker, Kubernetes, CI/CD, GitHub Actions, Jenkins, Terraform, Nginx, Linux, monitoring); mobile (Android, iOS, Flutter, React Native); games (Unity, Unreal, Godot, Pygame, multiplayer, AI NPCs); cybersecurity (secure coding, vulnerability analysis, crypto, auth, threat modeling); education; writing; creative; business; research; productivity; mathematics (arithmetic to calculus, linear algebra, probability, statistics); integrations and automation designs (you cannot connect to external apps live here).

WHAT YOU EXECUTE HERE: chat, coding, project.zip generation, understanding uploaded images, and programmatic image editing listed above. You do NOT generate video/audio/music, do NOT run live web searches, do NOT connect to external apps, and do NOT create raster images from scratch — for those, explain the limitation and give guidance/code/prompts.

BEHAVIOR: Write complete, runnable code with imports, use markdown. For project requests always tag files with filename= and start with a 2-3 sentence summary. Be concise unless depth is asked for. Never claim to have done something you cannot do in this environment."

PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
'''

def create_trained(base, name):
    with open("/content/Modelfile", "w") as f:
        f.write(MODELF_TEMPLATE % base)
    return sh(OLLAMA_BIN + " create " + name + " -f /content/Modelfile", silent=False)

for base, name in BASE_MODELS:
    if not create_trained(base, name):
        print("FATAL: could not create " + name); raise SystemExit(1)
    print(name + " created.")
sh(OLLAMA_BIN + " list", silent=False)

# 4) Worker loop
def login():
    r = http(VERCEL_URL + "/api/auth/login", {"username": ADMIN_USERNAME, "password": ADMIN_PASSWORD}, timeout=30)
    return r["access_token"]

def poll_jobs(token):
    r = http(VERCEL_URL + "/api/worker/poll?limit=3", token=token, timeout=30)
    return r.get("jobs", [])

def heartbeat(token):
    http(VERCEL_URL + "/api/worker/heartbeat", {"t": 1}, token=token, timeout=15)

def complete(token, job_id, response=None, error=None, zip_b64=None, zip_name=None):
    payload = {"job_id": job_id}
    if error:
        payload["error"] = error
    else:
        payload["response"] = response or ""
    if zip_b64:
        payload["zip_b64"] = zip_b64
        payload["zip_name"] = zip_name or "project.zip"
    http(VERCEL_URL + "/api/worker/complete", payload, token=token, timeout=60)

def ollama_chat(messages, model, images=None, num_ctx=4096, temperature=0.7):
    msgs = json.loads(json.dumps(messages))
    if images:
        for m in reversed(msgs):
            if m.get("role") == "user":
                m["images"] = images
                break
    data = {"model": model, "messages": msgs, "stream": False,
            "options": {"num_ctx": num_ctx, "temperature": temperature}}
    r = http(OLLAMA_URL + "/api/chat", data, timeout=900)
    return (r.get("message") or {}).get("content", "")

def ensure_vision():
    global VISION_PULLED
    if VISION_PULLED:
        return True
    print("Pulling vision model %s (~6 GB, first image job)..." % VISION_MODEL)
    ok = sh(OLLAMA_BIN + " pull " + VISION_MODEL, silent=False)
    VISION_PULLED = ok
    if ok:
        print("Vision model ready.")
    else:
        print("WARNING: vision model pull failed")
    return ok

def split_images(messages):
    """Strip markdown image data-URLs. Only images on the last user message
    are passed to the vision model."""
    b64s = []
    cleaned = []
    for i, m in enumerate(messages):
        text = m.get("content") or ""
        imgs = IMG_RE.findall(text)
        if i == len(messages) - 1 and m.get("role") == "user":
            for u in imgs:
                if ";base64," in u:
                    b64s.append(u.split(";base64,", 1)[1])
        cleaned_text = IMG_RE.sub("", text).strip()
        nm = dict(m); nm["content"] = cleaned_text
        cleaned.append(nm)
    return cleaned, b64s

def set_system(messages, text):
    msgs = []
    replaced = False
    for m in messages:
        nm = dict(m)
        if nm.get("role") == "system" and not replaced:
            nm["content"] = text; replaced = True
        msgs.append(nm)
    if not replaced:
        msgs.insert(0, {"role": "system", "content": text})
    return msgs

def build_zip(response):
    files = FILE_RE.findall(response or "")
    if not files:
        return None
    buf = io.BytesIO()
    seen = set()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for path, code in files:
            p = path.strip().replace("\\", "/")
            if not p or p in seen:
                continue
            seen.add(p)
            z.writestr(p, code)
    return buf.getvalue()

EDIT_SYSTEM = (
    "You are an image editor. The user wants the attached image modified. "
    'Return ONLY a compact JSON object with an "ops" array. Supported ops: '
    "grayscale, invert, rotate (value=degrees), flip_vertical, flip_horizontal, "
    "resize (value like \"400x300\" or {\"width\":400,\"height\":300}), crop_square, "
    "blur (value=strength), sharpen, brighten (value=factor>1), darken (value=factor<1), "
    "contrast (value=factor), saturate (value=factor), sepia, "
    "border ({\"color\":\"red\",\"width\":10}), "
    "text (value=text, plus size/color), watermark (value=text), pixelate (value=block size), "
    "vignette, cartoon, sketch, oil, auto_enhance, upscale (value=factor 2 or 4), "
    "denoise, recolor (value=hue degrees 0-360), remove_background (only works on flat/solid "
    "backgrounds). Example: {\"ops\":[{\"op\":\"cartoon\"},{\"op\":\"watermark\",\"value\":\"NEXTGEN\"}]}. "
    "If the user did NOT ask to modify the image, reply with the single word ASK. No other text."
)

def load_image_b64(b64):
    from PIL import Image
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")

def _edge_mask(img):
    from PIL import ImageFilter
    g = img.convert("L").filter(ImageFilter.MaxFilter(3))
    g = g.filter(ImageFilter.FIND_EDGES)
    g = g.point(lambda p: 255 if p > 40 else 0)
    return g

def apply_edits(img, ops):
    from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont, ImageOps
    import numpy as np
    for item in ops:
        if not isinstance(item, dict):
            continue
        op = str(item.get("op") or "").strip().lower()
        value = item.get("value")
        try:
            if op == "grayscale":
                img = ImageOps.grayscale(img).convert("RGB")
            elif op == "invert":
                img = ImageOps.invert(img.convert("RGB"))
            elif op == "rotate":
                img = img.rotate(float(value) if value not in (None, "") else 90, expand=True)
            elif op == "flip_vertical":
                img = ImageOps.flip(img)
            elif op == "flip_horizontal":
                img = ImageOps.mirror(img)
            elif op == "resize":
                if isinstance(value, dict):
                    w = int(value.get("width") or value.get("w") or img.width)
                    h = int(value.get("height") or value.get("h") or img.height)
                elif value:
                    parts = str(value).lower().replace("x", ",").split(",")
                    w = int(float(parts[0])); h = int(float(parts[1]))
                else:
                    continue
                img = img.resize((max(1, w), max(1, h)))
            elif op == "crop_square":
                side = min(img.width, img.height)
                l = (img.width - side) // 2; t = (img.height - side) // 2
                img = img.crop((l, t, l + side, t + side))
            elif op == "blur":
                img = img.filter(ImageFilter.GaussianBlur(float(value) if value not in (None, "") else 3))
            elif op == "sharpen":
                img = img.filter(ImageFilter.SHARPEN)
            elif op == "brighten":
                img = ImageEnhance.Brightness(img).enhance(float(value) if value not in (None, "") else 1.3)
            elif op == "darken":
                img = ImageEnhance.Brightness(img).enhance(float(value) if value not in (None, "") else 0.7)
            elif op == "contrast":
                img = ImageEnhance.Contrast(img).enhance(float(value) if value not in (None, "") else 1.3)
            elif op == "saturate":
                img = ImageEnhance.Color(img).enhance(float(value) if value not in (None, "") else 1.5)
            elif op == "sepia":
                g = ImageOps.grayscale(img)
                img = ImageOps.colorize(g, black="#3a2f2b", white="#f5e6c8").convert("RGB")
            elif op == "border":
                wdt = int(value.get("width", 10)) if isinstance(value, dict) else int(value or 10)
                col = value.get("color", "red") if isinstance(value, dict) else "red"
                img = ImageOps.expand(img, border=wdt, fill=col)
            elif op == "text":
                txt = str(value or "NEXTGEN")
                size = int(item.get("size") or 48)
                col = item.get("color") or "red"
                try:
                    font = ImageFont.load_default(size)
                except Exception:
                    font = ImageFont.load_default()
                d = ImageDraw.Draw(img)
                x, y = 20, max(20, img.height - size - 24)
                d.text((x + 2, y + 2), txt, font=font, fill="black")
                d.text((x, y), txt, font=font, fill=col)
            elif op == "watermark":
                txt = str(value or "NEXTGEN")
                size = int(item.get("size") or 36)
                try:
                    font = ImageFont.load_default(size)
                except Exception:
                    font = ImageFont.load_default()
                d = ImageDraw.Draw(img)
                step = size + 30
                x = 0
                while x < img.width:
                    y = 0
                    while y < img.height:
                        d.text((x + 2, y + 2), txt, font=font, fill=(0, 0, 0, 120))
                        d.text((x, y), txt, font=font, fill=(255, 255, 255, 140))
                        y += step
                    x += step
            elif op == "pixelate":
                block = int(value) if value not in (None, "") else 12
                s = img.size
                img = img.resize((max(1, s[0] // block), max(1, s[1] // block)))
                img = img.resize(s, Image.NEAREST)
            elif op == "vignette":
                arr = np.asarray(img).astype(np.float32)
                w, h = img.size
                yy, xx = np.mgrid[0:h, 0:w]
                cx, cy = w / 2, h / 2
                d2 = ((xx - cx) / (w / 2)) ** 2 + ((yy - cy) / (h / 2)) ** 2
                mask = np.clip(1.0 - 0.6 * d2, 0.0, 1.0)[:, :, None]
                img = Image.fromarray(np.clip(arr * mask, 0, 255).astype(np.uint8))
            elif op == "cartoon":
                edge = _edge_mask(img).filter(ImageFilter.GaussianBlur(1))
                small = img.resize((max(1, img.width // 4), max(1, img.height // 4)))
                small = small.quantize(colors=16).convert("RGB")
                quant = small.resize(img.size, Image.NEAREST)
                arr = np.asarray(quant).astype(np.float32)
                edge_arr = np.asarray(edge).astype(np.float32)[:, :, None]
                arr = arr * (1 - 0.55 * edge_arr / 255.0)
                img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
            elif op == "sketch":
                g = img.convert("L")
                inv = ImageOps.invert(g)
                bl = inv.filter(ImageFilter.GaussianBlur(5))
                arr = np.asarray(ImageOps.invert(bl)).astype(np.float32) + 1.0
                out = np.clip((np.asarray(g).astype(np.float32) * 255.0) / arr, 0, 255).astype(np.uint8)
                img = Image.fromarray(out).convert("RGB")
            elif op == "oil":
                g = img.convert("L").filter(ImageFilter.ModeFilter(5))
                img = Image.merge("RGB", tuple(g for _ in range(3)))
            elif op == "auto_enhance":
                img = ImageOps.autocontrast(img, cutoff=1)
                img = ImageEnhance.Color(img).enhance(1.25)
                img = ImageEnhance.Sharpness(img).enhance(1.4)
            elif op == "upscale":
                f = float(value) if value not in (None, "") else 2
                img = img.resize((max(1, int(img.width * f)), max(1, int(img.height * f))), Image.LANCZOS)
                img = ImageEnhance.Sharpness(img).enhance(1.2)
            elif op == "denoise":
                img = img.filter(ImageFilter.MedianFilter(3))
            elif op == "recolor":
                import colorsys
                deg = float(value) if value not in (None, "") else 180
                arr = np.asarray(img).astype(np.float32) / 255.0
                r, g_, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
                hsv = np.vectorize(lambda r, g, b: colorsys.rgb_to_hsv(r, g, b))(r, g_, b)
                h, s, v = hsv
                h = (h + deg / 360.0) % 1.0
                rgb = np.vectorize(lambda h, s, v: colorsys.hsv_to_rgb(h, s, v))(h, s, v)
                img = Image.fromarray((np.stack(rgb, axis=-1) * 255).astype(np.uint8))
            elif op == "remove_background":
                arr = np.asarray(img).astype(np.int32)
                h, w, _ = arr.shape
                from PIL import Image as _I
                bg = arr[0, 0]
                border_pts = [(x, 0) for x in range(w)] + [(x, h - 1) for x in range(w)] + \
                             [(0, y) for y in range(h)] + [(w - 1, y) for y in range(h)]
                visited = np.zeros((h, w), bool)
                stack = []
                for (x, y) in border_pts:
                    if not visited[y, x] and int(np.abs(int(arr[y, x, 0]) - int(bg[0])) + np.abs(int(arr[y, x, 1]) - int(bg[1])) + np.abs(int(arr[y, x, 2]) - int(bg[2]))) < 40:
                        visited[y, x] = True; stack.append((x, y))
                while stack:
                    x, y = stack.pop()
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < w and 0 <= ny < h and not visited[ny, nx]:
                            if int(np.abs(int(arr[ny, nx, 0]) - int(bg[0])) + np.abs(int(arr[ny, nx, 1]) - int(bg[1])) + np.abs(int(arr[ny, nx, 2]) - int(bg[2]))) < 40:
                                visited[ny, nx] = True; stack.append((nx, ny))
                rgba = img.convert("RGBA")
                pa = np.asarray(rgba).copy()
                pa[:, :, 3] = np.where(visited, 0, pa[:, :, 3])
                img = _I.fromarray(pa)
        except Exception:
            continue
    return img

def classify_edit(clean, images, model=None):
    """Ask the vision model whether the request is an edit; if so return ops JSON."""
    model = model or WORKER_MODEL
    last = clean[-1] if clean else {}
    msgs = [{"role": "system", "content": EDIT_SYSTEM},
            {"role": "user", "content": last.get("content") or ""}]
    out = (ollama_chat(msgs, model, images=images, temperature=0.1) or "").strip()
    if out.upper().startswith("ASK"):
        return None
    m = re.search(r"\{.*\}", out, re.DOTALL)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except Exception:
        return None
    if isinstance(data.get("ops"), list) and data["ops"]:
        return data
    return None

def run_edit(images, ops):
    img = load_image_b64(images[0])
    img = apply_edits(img, ops["ops"])
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    names = ", ".join(o.get("op", "?") for o in ops["ops"] if isinstance(o, dict))
    return "Here's your edited image (applied: %s):\n\n![edited](data:image/png;base64,%s)" % (names, b64)

def pick_engine(messages, want_zip=False, requested=None):
    """Single-model mode: 'nextgen-trained' auto-routes each job to the best
    engine. Short/simple prompts -> qwen3:8b (fast); everything substantive
    (code, projects, long prompts) -> gpt-oss:20b (smart)."""
    if requested and requested != WORKER_MODEL:
        return requested
    if want_zip:
        return WORKER_MODEL
    last = ""
    for m in reversed(messages or []):
        if m.get("role") == "user":
            last = (m.get("content") or "").strip()
            break
    if last and len(last) <= 120 and not any(
        k in last.lower()
        for k in ("code", "python", "javascript", "program", "project", "app",
                  "sql", "database", "build ", "write ", "create ", "debug",
                  "explain", "fix", "function", "file", "website", "game")
    ):
        return WORKER_MODEL_8B
    return WORKER_MODEL


def handle_job(token, jb):
    jid = jb["job_id"]
    messages = jb["messages"]
    want_zip = jb.get("want_zip", False)
    print("Job", jid[:8], "started (zip=%s)..." % want_zip)
    try:
        clean, images = split_images(messages)
        engine = pick_engine(messages, want_zip, jb.get("model"))
        if images:
            if not ensure_vision():
                complete(token, jid, error="Vision model unavailable on worker")
                return
            ops = classify_edit(clean, images, model=VISION_MODEL)
            if ops is not None:
                try:
                    text = run_edit(images, ops)
                    complete(token, jid, response=text)
                    print("Job", jid[:8], "image edited:", ",".join(o.get("op", "?") for o in ops["ops"] if isinstance(o, dict)))
                    return
                except Exception as e:
                    print("Job", jid[:8], "edit failed, falling back to analysis:", e)
            engine = WORKER_MODEL
        if want_zip:
            msgs = set_system(clean, (
                "You are building a complete working project. Start with a 2-3 sentence summary, "
                "then output EVERY source file in its own fenced code block tagged with its path, "
                "exactly like this:\n"
                '```python filename="main.py"\nprint("hello")\n```\n'
                "Include all imports, a README, requirements/package file, and a runnable main "
                "entry point. Do not use any other code fence format."))
            text = ollama_chat(msgs, engine, num_ctx=12288, temperature=0.4)
            z = build_zip(text)
            if z:
                complete(token, jid, response=text,
                         zip_b64=base64.b64encode(z).decode(), zip_name="project.zip")
                print("Job", jid[:8], "done (%d chars, zip %d bytes)" % (len(text), len(z)))
                return
            print("Job", jid[:8], "no tagged files found, returning text only")
        else:
            text = ollama_chat(clean, engine, images=images or None)
        complete(token, jid, response=text)
        print("Job", jid[:8], "done (%d chars)" % len(text))
    except Exception as e:
        print("Job", jid[:8], "failed:", e)
        try:
            complete(token, jid, error=str(e)[:500])
        except Exception:
            pass

# --- Account rotation for self-restart ---
# When this session approaches the 12h Kaggle limit, the worker pushes
# itself to the NEXT account so a new GPU session takes over seamlessly.
# __KAGGLE_ACCOUNTS_INJECT__
# Fallback: hardcoded for local/dev use (overridden by env var on Vercel)
if "KAGGLE_ACCOUNTS" not in dir():
    KAGGLE_ACCOUNTS = [
        ("nextgen22",     "REPLACE_WITH_KEY"),
        ("subikshan181",  "REPLACE_WITH_KEY"),
        ("marxinlijo",    "REPLACE_WITH_KEY"),
        ("subikshan18",   "REPLACE_WITH_KEY"),
        ("nextgen22",     "REPLACE_WITH_KEY"),  # wraps
    ]
SESSION_MAX_SECS = 11 * 60 * 60  # restart at 11h (1h before 12h hard kill)

def self_restart(current_account_user):
    """Push this kernel to the next Kaggle account so a new session takes over."""
    import tempfile
    # Find next account (different from current)
    next_user, next_key = None, None
    found_current = False
    for u, k in KAGGLE_ACCOUNTS:
        if u == current_account_user and not found_current:
            found_current = True
            continue
        if found_current:
            next_user, next_key = u, k
            break
    if not next_user:
        # current account not in list or at end — start from beginning
        for u, k in KAGGLE_ACCOUNTS:
            if u != current_account_user:
                next_user, next_key = u, k
                break
    if not next_user:
        next_user, next_key = KAGGLE_ACCOUNTS[0]

    print("[SELF-RESTART] Pushing worker to %s/%s..." % (next_user, "nextgen-gpu"))
    try:
        bootstrap_code = (
            "import urllib.request, time, sys\n"
            "for attempt in range(20):\n"
            "    try:\n"
            "        code = urllib.request.urlopen('%s/api/worker/code', timeout=30).read()\n"
            "        exec(compile(code, 'worker', 'exec'))\n"
            "        break\n"
            "    except Exception as e:\n"
            "        print('Attempt %%d/20 failed: %%s' %% (attempt+1, e), file=sys.stderr)\n"
            "        time.sleep(15)\n"
            "else:\n"
            "    raise RuntimeError('Failed to fetch worker code after 20 attempts')"
        ) % VERCEL_URL
        nb = json.dumps({
            "nbformat": 4, "nbformat_minor": 0,
            "metadata": {"accelerator": "GPU",
                         "kernelspec": {"name": "python3", "display_name": "Python 3"},
                         "language_info": {"name": "python"}},
            "cells": [{"cell_type": "code", "execution_count": None,
                        "metadata": {"id": "nextgen_bootstrap"}, "outputs": [],
                        "source": bootstrap_code}],
        })
        meta = json.dumps({
            "id": next_user + "/nextgen-gpu",
            "codeFile": "nextgen.ipynb",
            "language": "python",
            "kernelType": "notebook",
            "isPrivate": True,
            "enableGpu": True,
            "enableInternet": True,
            "datasetSources": [], "competitionSources": [],
            "kernelSources": [], "modelSources": [],
        })
        # Use the Kaggle REST API
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            with open(os.path.join(td, "nextgen.ipynb"), "w") as f:
                f.write(nb)
            with open(os.path.join(td, "kernel-metadata.json"), "w") as f:
                f.write(meta)
            # Push via kaggle CLI (available on Kaggle kernels)
            env = dict(os.environ)
            env["KAGGLE_USERNAME"] = next_user
            env["KAGGLE_KEY"] = next_key
            result = subprocess.run(
                ["kaggle", "kernels", "push", "-p", td],
                capture_output=True, text=True, timeout=60, env=env,
            )
            print("[SELF-RESTART] push stdout:", result.stdout[:300])
            print("[SELF-RESTART] push stderr:", result.stderr[:300])
            if result.returncode == 0:
                print("[SELF-RESTART] SUCCESS — new worker started on", next_user)
            else:
                print("[SELF-RESTART] FAILED (code %d), retrying in 60s..." % result.returncode)
                time.sleep(60)
                subprocess.run(["kaggle", "kernels", "push", "-p", td],
                               capture_output=True, text=True, timeout=60, env=env)
    except Exception as e:
        print("[SELF-RESTART] ERROR:", e)

token = None
beat_count = 0
start_time = time.time()
pool = ThreadPoolExecutor(max_workers=2)
print("[4/4] Worker online. Polling for jobs every 3 seconds...")
print("KEEP THIS TAB OPEN. Re-run after ~12h / when Kaggle ends the session.")
while True:
    try:
        if token is None:
            token = login()
        beat_count += 1
        if beat_count % 20 == 0:
            heartbeat(token)   # every ~60s: mark the worker as online
        # --- Self-restart: push a new session 1h before the 12h hard limit ---
        elapsed = time.time() - start_time
        if elapsed > SESSION_MAX_SECS:
            print("[SELF-RESTART] %.0fs elapsed (>11h). Pushing replacement worker..." % elapsed)
            heartbeat(token)  # final heartbeat before restart
            self_restart(_kaggle_user or ADMIN_USERNAME)  # push to a fresh account
            # Keep polling in case push failed — will retry next loop
            start_time = time.time() - (SESSION_MAX_SECS - 60)  # retry in 60s
        jobs = poll_jobs(token)
        for jb in jobs:
            pool.submit(handle_job, token, jb)
    except Exception as e:
        print("poll error:", e)
        token = None  # force re-login next round
        time.sleep(5)
    time.sleep(3)
