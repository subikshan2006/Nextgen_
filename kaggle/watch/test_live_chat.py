"""Live end-to-end chat test against the real GPU worker.

Submits an analytical question that routes through smart_chat's reasoning +
verification path, and checks the answer is a real grounded response rather
than an error or a stub.
"""
import json
import sys
import time
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = "https://nextgen-web-eta.vercel.app"

QUESTION = ("If a train leaves at 14:35 travelling 80 km/h and another leaves the "
            "same station at 15:10 travelling 120 km/h on the same track, when "
            "does the second catch the first? Show the working.")


def call(path, data=None, token=None, timeout=60):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(BASE + path, data=body)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode()
    return json.loads(raw) if raw.strip() else {}


token = call("/api/auth/login", {"username": "admin", "password": "admin12345"})["access_token"]

job = call("/api/chat/job", {"message": QUESTION}, token=token)
job_id = job.get("job_id") or job.get("id")
print("submitted job", job_id)

answer = None
status = None
start = time.time()
for i in range(80):
    time.sleep(5)
    d = call("/api/chat/job/" + job_id, token=token)
    status = d.get("status")
    if i % 4 == 0:
        print("  t+%3ds status=%s" % (int(time.time() - start), status))
    if status in ("done", "error", "failed"):
        answer = d.get("response") or d.get("error")
        break

print()
print("status:", status, " elapsed=%ds" % int(time.time() - start))
print("-" * 60)
print((answer or "(no answer)")[:1400])
print("-" * 60)

ok = False
if status == "done" and answer:
    low = answer.lower()
    # 14:35 + 0.5833h*80 = 16:22 ; gap 35min at 80 => 46.67km ; closing 40km/h
    # time to catch = 46.6667/40 = 1.1667h = 70min after 15:10 => 16:20
    has_math = any(t in low for t in ("16:20", "16:2", "70", "46.6", "1.16"))
    has_working = any(t in low for t in ("km", "hour", "speed", "difference", "catch"))
    ok = has_math and has_working
    print("numeric result present:", has_math, "| working shown:", has_working)

print("RESULT:", "PASS" if ok else "FAIL")
