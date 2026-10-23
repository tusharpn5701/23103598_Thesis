"""Inline web/data.json and the source files into template.html -> index.html."""
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(ROOT, ".."))
ORIG = sys.argv[1] if len(sys.argv) > 1 else REPO   # the original thesis scripts sit at the repo root

V2_FILES = ["soccer_env.py", "dqn.py", "ddpg.py", "replay.py", "agents.py", "train.py", "evaluate.py", "export_web.py"]
ORIG_FILES = ["DQN_attacker_phase1.py", "DQN_attacker_phase2.py", "DQN_attacker_phase3.py", "DQN_goalkeeper.py",
              "DDPG_attacker.py", "DDPG_goalkeeper.py", "DQN_Simulation.py", "DDPG_simulation.py"]


def read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


sources = [
    {"group": "v2 (revised)", "files": [{"name": n, "text": read(os.path.join(REPO, "v2", n))} for n in V2_FILES]},
    {"group": "Original thesis code", "files": [{"name": n, "text": read(os.path.join(ORIG, n)), "note": "as submitted"}
                                              for n in ORIG_FILES if os.path.exists(os.path.join(ORIG, n))]},
]
data = read(os.path.join(REPO, "v2", "web", "data.json"))
src = json.dumps(sources).replace("</", "<\\/")
html = read(os.path.join(ROOT, "template.html")).replace("__DATA__", data.replace("</", "<\\/")).replace("__SOURCES__", src)
out = os.path.join(ROOT, "index.html")
with open(out, "w", encoding="utf-8") as f:
    f.write(html)
print(f"wrote {out} ({os.path.getsize(out) / 1024:.0f} KB)")

# Standalone copy with a full document skeleton, for GitHub Pages (serve the docs/ folder)
standalone = ('<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
              '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
              '<style>:root{color-scheme:light;padding-top:env(safe-area-inset-top,0px);padding-bottom:env(safe-area-inset-bottom,0px)}'
              'body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>\n</head>\n<body>\n'
              + html + '\n</body>\n</html>\n')
docs = os.path.join(REPO, "docs")
os.makedirs(docs, exist_ok=True)
with open(os.path.join(docs, "index.html"), "w", encoding="utf-8") as f:
    f.write(standalone)
print(f"wrote {os.path.join(docs, 'index.html')}")
