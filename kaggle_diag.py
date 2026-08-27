import subprocess, socket

def t(url):
    try:
        r = subprocess.run(["curl", "-sI", "-o", "/dev/null", "-w", "%{http_code}", url],
                           capture_output=True, text=True, timeout=30)
        print(url, "->", r.stdout.strip(), "|", r.stderr[-200:].strip())
    except Exception as e:
        print(url, "ERR", e)

for u in ["https://github.com", "https://api.github.com",
          "https://ollama.com", "https://registry.ollama.ai",
          "https://pypi.org", "https://huggingface.co",
          "https://archive.ubuntu.com", "https://raw.githubusercontent.com"]:
    t(u)

for host in ["github.com", "ollama.com", "registry.ollama.ai", "huggingface.co", "pypi.org", "archive.ubuntu.com"]:
    try:
        print(host, "resolves ->", socket.gethostbyname(host))
    except Exception as e:
        print(host, "DNS FAIL:", e)
