import json
import subprocess
import sys
import time

import numpy as np

from embed import embed

products = [json.loads(line) for line in open("products.jsonl")]
texts = [f"{p['name']} {p['description']}" for p in products]
vectors = embed(texts)
vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)


def run(*args):
    done = subprocess.run([sys.executable, "similar.py", *args], capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, f"similar.py {' '.join(args)} failed:\n{done.stderr[-2000:]}"
    return done.stdout


run("index")
row = {p["id"]: i for i, p in enumerate(products)}
hits = total = 0
for target in [0, 17, 42, 99, 150]:
    start = time.time()
    got = [int(x) for x in run("like", str(products[target]["id"]), "-k", "5").split()]
    assert time.time() - start < 10, "like took over 10 s"
    assert len(got) == 5, f"asked for 5, got {got}"
    assert products[target]["id"] not in got, f"product {products[target]['id']} is in its own results"
    distances = 1 - vectors @ vectors[target]
    distances[target] = np.inf
    # Many products are exactly as far as the 5th nearest: any of them is a right answer.
    fifth = np.sort(distances)[4]
    hits, total = hits + sum(distances[row[g]] <= fifth + 1e-5 for g in got), total + 5
assert hits / total >= 0.8, f"recall {hits}/{total}"
print("ok")
