"""Turns the output of examples/ann.rs and reference_bench.py into Markdown tables, one per dataset:
build time, file size, and recall@10 / queries per second at each ef.

    python bench/ann/summary.py results/*.tsv
"""

import sys
from collections import defaultdict
from pathlib import Path

runs = defaultdict(lambda: defaultdict(dict))  # dataset -> run -> key -> fields
for path in map(Path, sys.argv[1:]):
    for line in path.read_text().splitlines():
        engine, dataset, key, *fields = line.split("\t")
        runs[dataset][path.stem][key] = fields

for dataset, by_run in runs.items():
    efs = sorted({int(k) for fields in by_run.values() for k in fields if k.isdigit()})
    print(f"### {dataset}\n")
    print("| Run | Build | File | " + " | ".join(f"ef {ef}" for ef in efs) + " |")
    print("|---|---|---|" + "---|" * len(efs))
    for run, fields in sorted(by_run.items()):
        build, size = fields.get("build", ["", ""])
        cells = [" / ".join(fields.get(str(ef), ["", ""])) for ef in efs]
        print(f"| {run} | {build} | {size} | " + " | ".join(cells) + " |")
    print("\nCells are recall@10 / queries per second, one thread.\n")
