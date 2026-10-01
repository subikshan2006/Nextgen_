import base64, json, re, sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"C:\Users\admin\OneDrive\Documents\Default Project\nextgen-web"

src = open(ROOT + r"\kaggle_nextgen.py", encoding="utf-8").read()
w = open(ROOT + r"\kaggle\watch\kaggle_embedded\worker_embedded.py", encoding="utf-8").read()
nb = json.load(open(ROOT + r"\kaggle\watch\kaggle_embedded\nextgen.ipynb", encoding="utf-8"))
cell = "".join(nb["cells"][0]["source"])

pat = re.compile(r"B64 = '''([A-Za-z0-9+/=]+)'''")
ok = True
for name, text in (("worker_embedded.py", w), ("nextgen.ipynb", cell)):
    m = pat.search(text)
    if not m:
        print(name, ": NO B64 FOUND")
        ok = False
        continue
    dec = base64.b64decode(m.group(1)).decode("utf-8")
    same = dec == src
    ok = ok and same
    print("%-22s -> matches kaggle_nextgen.py: %s" % (name, same))

decoded = base64.b64decode(pat.search(w).group(1)).decode("utf-8")
print()
for probe in (
    'r = http(VERCEL_URL + "/api/worker/poll?limit=3", token=token, timeout=30)',
    'return r.get("jobs", []), r.get("commands", [])',
    'def execute_command(token, c):',
    'def _chat_raw(',
    'def smart_chat(',
    'c["id"] = c.get("id", c.get("command_id"))',
):
    print("  present  %-62s %s" % (probe[:62], probe in decoded))

for stale in ('def execute_command(token, c, ollama)', 'ollama.save_patch(lesson)'):
    print("  STALE    %-62s %s" % (stale[:62], stale in decoded))
    if stale in decoded:
        ok = False

print()
print("RESULT:", "PASS" if ok else "FAIL")
