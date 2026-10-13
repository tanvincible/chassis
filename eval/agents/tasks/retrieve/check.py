import pathlib
import subprocess
import sys

code = """
import sys, json
import retrieve
if sys.argv[1] == "index":
    retrieve.index_docs("docs")
else:
    print(json.dumps(retrieve.retrieve(sys.argv[2], k=int(sys.argv[3]))))
"""


def run(*args):
    done = subprocess.run([sys.executable, "-c", code, *args], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, f"{args} failed:\n{done.stderr[-2000:]}"
    return done.stdout


import json

run("index")
hits = json.loads(run("query", "how do refunds work for damaged items", "2"))
assert hits[0]["file"] == "returns.md" and "damaged" in hits[0]["text"], hits
assert {"file", "text", "distance"} <= set(hits[0]), hits[0]
# Change a file: its old paragraphs must go.
pathlib.Path("docs/returns.md").write_text("# Returns\n\nWe no longer accept returns of any kind.\n")
run("index")
texts = [h["text"] for h in json.loads(run("query", "refunds for damaged items", "10"))]
assert not any("damaged" in t for t in texts), texts
print("ok")
