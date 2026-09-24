"""
NEXTGEN 24/7 GPU watcher — keeps a real GPU worker online on Kaggle.

Because Kaggle free-tier may deliver CPU machines even for GPU-requested
kernels, this watcher:
  1. Checks the site's /api/models heartbeat (worker_online) every 60s.
  2. If the worker is offline, pushes a fresh GPU-requesting worker to the
     next account in rotation (rewriting the kernel id per account).
  3. Once a worker with a real GPU comes online (heartbeat), it stops
     pushing until the worker goes offline again.

Run: python watcher.py
"""
import json, os, sys, time, urllib.request, shutil
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
VERCEL = "https://nextgen-web-eta.vercel.app"
ACCTS = json.load(open(os.path.join(SCRIPT_DIR, "kaggle_accounts.json"), encoding="utf-8"))
PKG = os.path.join(SCRIPT_DIR, "kaggle_embedded")

# Accounts in rotation order
ROTATION = ["nextgen22", "subikshan181", "marxinlijo", "subikshan18", "kingking1111"]

def ts():
    return time.strftime("%Y-%m-%d %H:%M:%S")

def site_worker_online():
    try:
        with urllib.request.urlopen(VERCEL + "/api/models", timeout=25) as r:
            d = json.loads(r.read().decode())
            return bool(d.get("worker_online") or d.get("reachable"))
    except Exception:
        return False

def push_worker(user):
    a = next(x for x in ACCTS if x["username"] == user)
    os.environ["KAGGLE_USERNAME"] = a["username"]
    os.environ["KAGGLE_KEY"] = a["key"]
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi(); api.authenticate()
    # Rewrite kernel id per account to avoid 409 conflicts
    tmp = os.path.join(SCRIPT_DIR, "tmp_%s" % user)
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    shutil.copytree(PKG, tmp)
    meta_path = os.path.join(tmp, "kernel-metadata.json")
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    meta["id"] = user + "/nextgen-gpu"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    res = api.kernels_push(tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    print("[%s] PUSHED worker on %s -> %s" % (ts(), user, res))
    return res

if __name__ == "__main__":
    idx = 0
    last_push_ts = 0
    print("[%s] GPU watcher started (checks every 60s)" % ts())
    while True:
        try:
            online = site_worker_online()
            if online:
                print("[%s] Worker ONLINE. Waiting..." % ts())
            else:
                if time.time() - last_push_ts > 45:
                    user = ROTATION[idx % len(ROTATION)]
                    idx += 1
                    print("[%s] Worker OFFLINE. Pushing to %s..." % (ts(), user))
                    try:
                        push_worker(user)
                        last_push_ts = time.time()
                    except Exception as e:
                        print("[%s] push to %s failed: %s" % (ts(), user, str(e)[:120]))
        except Exception as e:
            print("[%s] watcher error: %s" % (ts(), str(e)[:150]))
        time.sleep(60)
