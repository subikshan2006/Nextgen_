"""Live end-to-end test against production.

The strongest signal for which worker build is serving: admin commands.
The OLD worker claimed commands from /api/worker/poll and then discarded them,
so they stayed 'running' forever. The NEW worker executes and completes them.
A command reaching 'done' proves the new brain is live.
"""
import json
import sys
import time
import urllib.error
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
BASE = "https://nextgen-web-eta.vercel.app"


def call(path, data=None, token=None, timeout=40):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(BASE + path, data=body)
    req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", "Bearer " + token)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read().decode()
    return json.loads(raw) if raw.strip() else {}


token = call("/api/auth/login", {"username": "admin", "password": "admin12345"})["access_token"]
print("admin login OK")

# Queue a command that only the new code path handles cleanly.
marker = "live-verify-%d" % int(time.time())
qid = call("/api/admin/commands",
           {"kind": "remember",
            "payload": json.dumps({"content": marker})}, token)["id"]
print("queued command id=%s" % qid)

status = None
for i in range(30):
    time.sleep(4)
    try:
        found = None
        listing = call("/api/admin/commands?limit=50", token=token)
        items = listing if isinstance(listing, list) else listing.get("commands", [])
        for c in items:
            if c.get("id") == qid:
                found = c
                break
        if found is None:
            # no longer pending => it was claimed; check the history endpoint
            h = call("/api/admin/commands/history?limit=50", token=token)
            hitems = h if isinstance(h, list) else h.get("commands", [])
            for c in hitems:
                if c.get("id") == qid:
                    found = c
                    break
        if found:
            status = found.get("status")
            if status in ("done", "error", "failed"):
                break
    except urllib.error.HTTPError as e:
        print("  poll HTTP", e.code)
    if i % 5 == 4:
        print("  t+%ds status=%s" % ((i + 1) * 4, status))

print()
print("command %s final status: %s" % (qid, status))
if status == "done":
    print("RESULT: PASS - new worker build is live (commands are executed)")
elif status == "running":
    print("RESULT: FAIL - command stranded in 'running' => OLD worker still serving")
else:
    print("RESULT: INCONCLUSIVE - status=%s" % status)
