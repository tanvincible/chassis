"""The decision in byte-plan.md, from bytebench.yml's artifacts (lab only).

    python bench/lab/bytebench.py <artifact dir>...

Each dir holds lscpu.txt and bytebench.tsv. A job whose H2 is more than 5% from H anywhere is
too noisy to decide on, and one where the planted HH check fails is left out; the rest are taken
by CPU model, the median over jobs of each speed.
"""

import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path

DIMS = (384, 768, 1536)


def job(d: Path):
    model = re.search(r"^Model name:\s*(.+)$", (d / "lscpu.txt").read_text(), re.M).group(1).strip()
    ns = {}
    for line in (d / "bytebench.tsv").read_text().splitlines():
        f = line.split("\t")
        if f[0] == "bytebench":
            ns[(f[1], int(f[2]), f[3])] = float(f[5])
    return model, ns


def main(dirs: list[Path]) -> None:
    by_model, noisy, planted = defaultdict(list), [], []
    for d in dirs:
        model, ns = job(d)
        cells = [(r, n) for r in ("cache", "memory") for n in DIMS]
        off = max(abs(ns[(r, n, "H")] / ns[(r, n, "H2")] - 1) for r, n in cells)
        doubled = min((ns[("cache", n, "HH")] - ns[("cache", n, "Z")]) / (ns[("cache", n, "H")] - ns[("cache", n, "Z")]) for n in DIMS)
        planted.append(f"* {d.name}, {model}: H2 at most {off:.1%} from H; HH less Z at least {doubled:.2f} times H less Z in cache")
        if doubled < 1.5:
            planted[-1] += " **(planted check fails: left out)**"
            continue
        if off > 0.05:
            noisy.append(f"{d.name} ({model})")
            continue
        by_model[model].append(ns)

    print("## Planted outcomes\n")
    print("\n".join(planted))
    print(f"\nToo noisy to decide on: {', '.join(noisy) or 'none'}\n")

    print("## Speed against half precision (H's time over the kernel's), median by CPU\n")
    print("| CPU | jobs | regime | kernel | 384 | 768 | 1536 |")
    print("|---|---|---|---|---|---|---|")
    speed = {}
    for model, jobs in sorted(by_model.items()):
        for regime in ("cache", "memory"):
            for k in ("D", "E", "E4"):
                row = []
                for n in DIMS:
                    s = statistics.median(j[(regime, n, "H")] / j[(regime, n, k)] for j in jobs)
                    speed[(model, regime, k, n)] = s
                    row.append(f"{s:.2f}")
                print(f"| {model} | {len(jobs)} | {regime} | {k} | {' | '.join(row)} |")

    print("\n## Decision (pre-registered)\n")
    models = sorted(by_model)
    if not models:
        print("Nothing to decide on: every job was too noisy.")
        return

    def worst(k, regime):
        return min(speed[(m, regime, k, n)] for m in models for n in DIMS)

    verdicts = {}
    for k in ("D", "E"):
        cache, memory = worst(k, "cache"), worst(k, "memory")
        if cache >= 0.9 and memory >= 1.0:
            verdicts[k] = (1, cache, memory)
        elif memory >= 1.0 and cache >= 0.75:
            verdicts[k] = (2, cache, memory)
        else:
            verdicts[k] = (3, cache, memory)
        print(f"* {k}: worst {cache:.2f} in cache, {memory:.2f} out of cache: outcome {verdicts[k][0]}")
    best = min(verdicts.values())[0]
    passing = [k for k in ("D", "E") if verdicts[k][0] == best]
    if len(passing) == 2:
        close = all(speed[(m, r, "D", n)] >= 0.95 * speed[(m, r, "E", n)] for m in models for r in ("cache", "memory") for n in DIMS)
        passing = ["D"] if close else ["E"]
    text = {
        1: "the kernel is not what decides it: the file format is next, with",
        2: "the format is next, with its cost in cache stated, with",
        3: "no format yet: a kernel with the query in eight bits too is tried first",
    }[best]
    print(f"\n**{text}{' ' + passing[0] if best < 3 else ''}.** CPUs decided on: {', '.join(models)}.")


if __name__ == "__main__":
    main([Path(a) for a in sys.argv[1:]])
