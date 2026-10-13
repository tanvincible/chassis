import subprocess
import sys

import numpy as np

from embed import embed

lines = open("tickets.txt").read().splitlines()
vectors = embed(lines)
vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
near = (1 - vectors @ vectors.T) < 0.35
parent = list(range(len(lines)))


def root(i):
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


for i, j in zip(*np.nonzero(np.triu(near, 1))):
    parent[root(i)] = root(j)
groups = {}
for i in range(len(lines)):
    groups.setdefault(root(i), []).append(i + 1)
want = {tuple(g) for g in groups.values() if len(g) > 1}

done = subprocess.run([sys.executable, "dedupe.py"], capture_output=True, text=True, timeout=300)
assert done.returncode == 0, done.stderr[-2000:]
got = {tuple(sorted(int(x) for x in line.split())) for line in done.stdout.splitlines() if line.strip()}
found = len(want & got) / max(len(want), 1)
assert found >= 0.9, f"{len(want & got)} of {len(want)} groups exactly right; {len(got - want)} wrong"
print("ok")
