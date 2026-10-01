"""Verify the admin Command Center actually changes real system state.

Every command here used to report success while doing nothing:
  - "emotion set to X" wrote a memory row, never the ApiSetting the prompt
    and /api/alive actually read, so the emotion never changed.
  - "remember" from the queue was scoped to the admin's own user_id, so normal
    users never saw the operator's instructions.
This exercises the real endpoints against a local database.
"""
import os
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ROOT = r"C:\Users\admin\OneDrive\Documents\Default Project\nextgen-web"
sys.path.insert(0, ROOT)

tmpdb = os.path.join(tempfile.gettempdir(), "nextgen_cmdtest.db")
if os.path.exists(tmpdb):
    os.remove(tmpdb)
os.environ["DATABASE_URL"] = "sqlite:///" + tmpdb.replace("\\", "/")
os.environ["ADMIN_EMAIL"] = "admin@test.local"
os.environ["ADMIN_PASSWORD"] = "admin12345"
os.environ.setdefault("JWT_SECRET", "test-secret-key-for-local-verification-only")

import app.database as database  # noqa: E402

database.init_db()
SessionLocal = database.SessionLocal
from app.models import ApiSetting, Memory, SelfImprovement, User  # noqa: E402
from app.routers.admin_command import _apply_host_side, issue_command  # noqa: E402
from app.routers.worker import build_lifelong_context  # noqa: E402
from app.schemas import WorkerCommandIn  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from app.auth import get_current_admin  # noqa: E402

app = FastAPI()
app.include_router(
    __import__("app.routers.admin_command", fromlist=["router"]).router,
    prefix="", dependencies=[],
)
app.dependency_overrides[get_current_admin] = lambda: None

from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(app)

db = SessionLocal()
admin = db.query(User).filter(User.is_admin == True).first()  # noqa: E712
other = User(username="regularuser", email="u@test.local",
             password_hash="x", is_admin=False, is_active=True)
db.add(other)
db.commit()
other_id = other.id
db.close()

fails = []


def check(name, cond, detail=""):
    print(("  PASS  " if cond else "  FAIL  ") + name + (" :: " + detail if detail else ""))
    if not cond:
        fails.append(name)


print("1) emotion command must change the emotion the system reads")
r = client.post("/api/admin/commands",
                json={"kind": "emotion",
                      "payload": '{"mood":"focused","intensity":8}'})
body = r.json()
check("status 200", r.status_code == 200, str(r.status_code))
check("command is done", body.get("status") == "done", str(body.get("status")))
db = SessionLocal()
mood = db.query(ApiSetting).filter(ApiSetting.key == "emotion_mood").first()
check("emotion_mood row written", mood is not None and mood.value == "focused",
      mood.value if mood else "missing")
db.close()

print()
print("2) emotion must be injected into the system prompt")
db = SessionLocal()
ctx = build_lifelong_context(db, other_id)
db.close()
check("prompt contains emotion", "focused" in ctx and "EMOTIONAL STATE" in ctx,
      ctx[:120].replace("\n", " "))

print()
print("3) 'remember' must be global so it reaches every user")
r = client.post("/api/admin/commands",
                json={"kind": "remember",
                      "payload": '{"content":"Always answer in metric units"}'})
db = SessionLocal()
global_mem = db.query(Memory).filter(
    Memory.user_id.is_(None), Memory.content == "Always answer in metric units").first()
check("command done", r.json().get("status") == "done")
check("stored as global memory (user_id NULL)", global_mem is not None)
db.close()

print()
print("4) that global instruction must appear for a NON-admin user")
db = SessionLocal()
ctx_other = build_lifelong_context(db, other_id)
db.close()
check("regular user sees operator instruction",
      "Always answer in metric units" in ctx_other,
      ctx_other[:200].replace("\n", " "))

print()
print("5) 'improve' must write a real lesson")
r = client.post("/api/admin/commands",
                json={"kind": "improve",
                      "payload": '{"content":"Prefer runnable examples"}'})
db = SessionLocal()
lesson = db.query(SelfImprovement).filter(
    SelfImprovement.content == "Prefer runnable examples").first()
check("command done", r.json().get("status") == "done")
check("lesson row exists", lesson is not None)
db.close()

print()
print("6) 'grant' must write the tool setting the prompt reads")
r = client.post("/api/admin/commands",
                json={"kind": "grant",
                      "payload": '{"tool":"web_search","enabled":true}'})
db = SessionLocal()
tool = db.query(ApiSetting).filter(ApiSetting.key == "tool_web_search").first()
check("command done", r.json().get("status") == "done")
check("tool_web_search == 1", tool is not None and str(tool.value) == "1",
      tool.value if tool else "missing")
ctx2 = None
db.close()
db = SessionLocal()
ctx2 = build_lifelong_context(db, other_id)
db.close()
check("tool grant injected", "web_search" in ctx2 and "TOOL ACCESS" in ctx2)

print()
print("7) self_update must stay queued for the GPU worker")
r = client.post("/api/admin/commands", json={"kind": "self_update", "payload": "{}"})
check("self_update is pending", r.json().get("status") == "pending",
      str(r.json().get("status")))

print()
print("8) duplicate remember must not create a second row")
client.post("/api/admin/commands",
            json={"kind": "remember", "payload": '{"content":"Always answer in metric units"}'})
db = SessionLocal()
n = db.query(Memory).filter(
    Memory.user_id.is_(None), Memory.content == "Always answer in metric units").count()
db.close()
check("still exactly 1 row", n == 1, "count=%d" % n)

print()
print("ALL PASS" if not fails else "FAILURES: " + ", ".join(fails))
sys.exit(1 if fails else 0)
