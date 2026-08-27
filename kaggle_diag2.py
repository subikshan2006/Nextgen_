import os, socket, subprocess

for k in ["HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "NO_PROXY"]:
    print(k, "=", os.environ.get(k))

try:
    print("resolv.conf:")
    print(open("/etc/resolv.conf").read())
except Exception as e:
    print("no resolv.conf:", e)

for host in ["kaggle.com", "www.kaggle.com", "storage.googleapis.com"]:
    try:
        print(host, "resolves ->", socket.gethostbyname(host))
    except Exception as e:
        print(host, "DNS FAIL:", e)

try:
    s = socket.create_connection(("8.8.8.8", 53), timeout=5)
    s.close()
    print("8.8.8.8:53 reachable")
except Exception as e:
    print("8.8.8.8:53 fail:", e)

try:
    r = subprocess.run(["python", "-m", "pip", "config", "list"], capture_output=True, text=True, timeout=30)
    print("pip config:", r.stdout.strip(), r.stderr.strip())
except Exception as e:
    print("pip config ERR", e)

try:
    r = subprocess.run(["python", "-m", "pip", "download", "--no-deps", "-d", "/tmp/piptest", "zstandard", "-q"],
                       capture_output=True, text=True, timeout=90)
    print("pip download rc=", r.returncode)
    print("pip stdout:", r.stdout[-500:])
    print("pip stderr:", r.stderr[-500:])
except Exception as e:
    print("pip download ERR:", e)

import datetime; print('recheck run at', datetime.datetime.utcnow().isoformat())
