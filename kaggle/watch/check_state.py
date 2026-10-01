import json
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
B = "https://nextgen-web-eta.vercel.app"


def call(p, d=None, t=None):
    r = urllib.request.Request(B + p, data=json.dumps(d).encode() if d is not None else None)
    r.add_header("Content-Type", "application/json")
    if t:
        r.add_header("Authorization", "Bearer " + t)
    x = urllib.request.urlopen(r, timeout=40).read().decode()
    return json.loads(x) if x.strip() else {}


tk = call("/api/auth/login", {"username": "admin", "password": "admin12345"})["access_token"]
h = call("/api/admin/commands", t=tk)
items = h if isinstance(h, list) else h.get("commands", [])
print("total history:", len(items))
for c in items:
    print("  %-6s %-10s %-12s %s" % (
        str(c.get("id"))[:6], c.get("status"), (c.get("kind") or "")[:12],
        (c.get("result") or "")[:40]))

m = call("/api/alive", t=tk)
print()
print("alive:", json.dumps(m)[:600])
