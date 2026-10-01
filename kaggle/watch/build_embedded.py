"""
Rebuild the embedded Kaggle worker package from the canonical source.

The Kaggle kernel cannot read files from this machine, so the whole worker is
base64-encoded into a single notebook cell. That means the notebook silently
goes stale whenever kaggle_nextgen.py changes. This script regenerates both
artifacts from the real source so the watcher always pushes current code.

Run: python kaggle/watch/build_embedded.py
"""
import base64
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
SRC = os.path.join(ROOT, "kaggle_nextgen.py")
PKG = os.path.join(SCRIPT_DIR, "kaggle_embedded")

WORKER_TEMPLATE = (
    "import base64, urllib.request\n"
    "B64 = '''{b64}'''\n"
    "exec(compile(base64.b64decode(B64).decode('utf-8'), 'worker', 'exec'))\n"
)


def main():
    with open(SRC, encoding="utf-8") as f:
        source = f.read()
    b64 = base64.b64encode(source.encode("utf-8")).decode("ascii")
    if "'''" in b64:
        raise SystemExit("base64 payload cannot contain the triple quote")

    os.makedirs(PKG, exist_ok=True)

    worker_path = os.path.join(PKG, "worker_embedded.py")
    with open(worker_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(WORKER_TEMPLATE.format(b64=b64))

    cell = WORKER_TEMPLATE.format(b64=b64).splitlines(True)
    cell[0] = "import base64, urllib.request\n"
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": cell,
            }
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    nb_path = os.path.join(PKG, "nextgen.ipynb")
    with open(nb_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")

    # The kernel entry point must be runnable on Kaggle.
    import py_compile
    py_compile.compile(worker_path, doraise=True)

    print("rebuilt from %s" % SRC)
    print("  source    %d bytes" % len(source.encode("utf-8")))
    print("  worker    %s (%d bytes)" % (worker_path, os.path.getsize(worker_path)))
    print("  notebook  %s (%d bytes)" % (nb_path, os.path.getsize(nb_path)))


if __name__ == "__main__":
    main()
