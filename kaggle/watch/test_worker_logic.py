"""Offline test of the worker command pipeline against a fake Vercel.

Confirms: /poll commands are consumed (not stranded in running), both
command_id and id are handled, smart_chat routes/verifies, and a 32K OOM
degrades to a smaller context instead of failing the job.
"""
import sys, types, json, io, contextlib

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"C:\Users\admin\OneDrive\Documents\Default Project\nextgen-web"

calls = {"memory": [], "cmd_done": [], "chat": [], "improve": []}


def load_worker(chat_impl):
    src = open(ROOT + r"\kaggle_nextgen.py", encoding="utf-8").read()
    # Everything above the worker-loop section is the Kaggle boot driver
    # (env detect, apt-get Ollama, create trained model) and is not runnable
    # here. Keep the worker helpers that follow it.
    start = src.find("# 4) Worker loop")
    assert start != -1, "worker-loop marker not found"
    src = src[start:]
    # Drop the trailing infinite polling driver; we drive the functions directly.
    for marker in ('print("[4/4]', "token = None\nbeat_count"):
        if marker in src:
            src = src.split(marker)[0]
    g = {"__name__": "worker_test"}
    # Config constants + imports live in the boot block we skipped.
    for mod in ("base64", "io", "json", "os", "re", "subprocess",
                "time", "urllib.request", "zipfile"):
        __import__(mod)
        g[mod] = sys.modules[mod]
    g.update({
        "VERCEL_URL": "https://example.test",
        "OLLAMA_URL": "http://ollama.test",
        "ADMIN_USERNAME": "admin",
        "ADMIN_PASSWORD": "x",
        "BASE_MODEL": "qwen2.5vl:7b",
        "WORKER_MODEL": "nextgen-trained",
        "OLLAMA_BIN": "/usr/local/bin/ollama",
        "IMG_RE": __import__("re").compile(r"!\[[^\]]*\]\((data:image/[^)]+)\)"),
    })
    def fake_http(url, data=None, token=None, timeout=30):
        calls["chat"].append({"url": url, "data": data})
        return chat_impl(url, data)
    g["http"] = fake_http
    g["time"] = types.SimpleNamespace(sleep=lambda *a: None)
    g["sh"] = lambda *a, **k: ""
    exec(compile(src, "kaggle_nextgen", "exec"), g)
    return g


# ---------- test 1: command pipeline ----------
def chat_for_commands(url, data):
    if "/api/worker/poll" in url:
        # the server claims 1 pending command and returns it
        return {"jobs": [], "commands": [
            {"command_id": 77, "kind": "self_update", "payload": "{}"}]}
    if "/api/worker/commands/complete" in url:
        calls["cmd_done"].append(data); return {"ok": True}
    if "/api/worker/commands" in url:
        return {"commands": []}
    if "/api/worker/memory" in url:
        calls["memory"].append(data); return {"ok": True}
    return {}

g = load_worker(chat_for_commands)
jobs, commands = g["poll_jobs"]("t")
print("poll returned jobs=%d commands=%d" % (len(jobs), len(commands)))
for c in commands:
    g["execute_command"]("t", c)
print("memory posts   :", calls["memory"])
print("command done   :", calls["cmd_done"])
t1 = (len(calls["cmd_done"]) == 1
      and calls["cmd_done"][0]["command_id"] == 77
      and calls["cmd_done"][0]["status"] == "done")
print("TEST 1 (poll consumes commands, command_id honoured):", "PASS" if t1 else "FAIL")

# ---------- test 2: legacy "id" key + server-side kinds ----------
calls["cmd_done"].clear()
g["execute_command"]("t", {"id": 5, "kind": "emotion",
                           "payload": json.dumps({"mood": "focused", "intensity": 7})})
t2 = (len(calls["cmd_done"]) == 1
      and calls["cmd_done"][0]["command_id"] == 5
      and calls["cmd_done"][0]["status"] == "done")
print("TEST 2 (legacy id key):", "PASS" if t2 else "FAIL", calls["cmd_done"])

# ---------- test 3: smart_chat routing ----------
def chat_router(url, data):
    msgs = json.dumps(data.get("messages", []))
    if "strict reviewer" in msgs:
        return {"message": {"content": "OK"}}
    if "work through this carefully" in msgs:
        return {"message": {"content": "plan: derive then verify"}}
    return {"message": {"content": "FINAL ANSWER"}}

g2 = load_worker(chat_router)
g2["OLLAMA_URL"] = "http://x"
model = "nextgen-trained"
r_hard = g2["smart_chat"]([{"role": "user", "content": "derive the derivative of x^2 step by step"}], model)
r_easy = g2["smart_chat"]([{"role": "user", "content": "hey how are you"}], model)
r_img = g2["smart_chat"]([{"role": "user", "content": "what is this"}], model, images=["b64"])
print("hard ->", r_hard)
print("easy ->", r_easy)
print("image->", r_img)
t3 = (r_hard == "FINAL ANSWER" and r_easy == "FINAL ANSWER" and r_img == "FINAL ANSWER")
print("TEST 3 (smart_chat routes correctly):", "PASS" if t3 else "FAIL")

# ---------- test 4: 32K OOM degrades ----------
seq = []
def chat_oom(url, data):
    ctx = (data.get("options") or {}).get("num_ctx")
    seq.append(ctx)
    if "work through this carefully" in json.dumps(data.get("messages", [])):
        if ctx and ctx > 8192:
            raise RuntimeError("CUDA out of memory")
        return {"message": {"content": "small-context plan"}}
    if "strict reviewer" in json.dumps(data.get("messages", [])):
        return {"message": {"content": "OK"}}
    if ctx and ctx > 8192:
        raise RuntimeError("CUDA out of memory")
    return {"message": {"content": "ANSWER AT SMALL CTX"}}

g3 = load_worker(chat_oom)
g3["OLLAMA_URL"] = "http://x"
out = g3["smart_chat"]([{"role": "user", "content": "implement a complex database algorithm"}], "m")
print("contexts tried:", seq)
print("final:", out)
t4 = out == "ANSWER AT SMALL CTX" and any(c and c > 8192 for c in seq) and 8192 in seq
print("TEST 4 (OOM degrades 32K->8K instead of failing):", "PASS" if t4 else "FAIL")

print()
print("ALL:", "PASS" if all([t1, t2, t3, t4]) else "FAIL")
