"""avx512 RUN 1 (avx512-plan.md): checks per job, then each CPU's median ratios.

    python bench/lab/avx512_report.py <dir of downloaded artifacts>
"""

import glob
import os
import re
import statistics
import sys
from collections import defaultdict

VECTOR_BYTES = 99_000 * 1536 * 4
WARM, WARM512, AGAIN = "chassis-py-batch-warm", "chassis-py-batch-warm-512", "chassis-py-batch-warm-again"


def at(points: dict, target: float):
    """Queries a second at `target` recall, linear between the two `ef` around it."""
    pts = [points[ef] for ef in sorted(points)]
    for (r1, q1), (r2, q2) in zip(pts, pts[1:]):
        if r1 <= target <= r2:
            return q1 + (q2 - q1) * (target - r1) / (r2 - r1) if r2 > r1 else q1
    return None


def short(cpu):
    for pat, label in [("7763", "Zen 3"), ("9V74", "Zen 4"), ("9V45", "Zen 5")]:
        if pat in cpu:
            return label
    m = re.search(r"(\d{4}[A-Z]*(-C)?)", cpu)
    return f"Xeon {m.group(1)}" if m else cpu[:24]


def job(d):
    lscpu = open(f"{d}/lscpu.txt").read()
    cpu = re.search(r"Model name:\s*(.*)", lscpu).group(1).strip()
    avx512 = " avx512f " in f" {re.search(r'Flags:(.*)', lscpu).group(1)} "
    dist, policy = defaultdict(dict), ""
    for line in open(f"{d}/distbench.tsv") if os.path.exists(f"{d}/distbench.tsv") else []:
        f = line.rstrip("\n").split("\t")
        if f[1] == "policy":
            policy = " ".join(f[2:])
        elif len(f) == 5:
            dist[f[1]][f[3]] = float(f[4])
            dist[f[1]]["share"] = float(f[2])
    search, pages, levels, kernels = defaultdict(dict), {}, {}, {}
    for line in open(f"{d}/inc.tsv") if os.path.exists(f"{d}/inc.tsv") else []:
        f = line.rstrip("\n").split("\t")
        if f[0] == "inc" and f[3] == "search":
            search[f[1]][int(f[4])] = (float(f[5]), float(f[6]))
        elif f[0] == "inc-pages":
            pages[f[1]] = {k: int(v) for k, v in re.findall(r"(\w+) (\d+) kB", f[2])}
        elif f[0] == "inc-lib" and "faiss" in f[2]:
            levels[f[1]] = f[2].rsplit("level ", 1)[-1]
        elif f[0] == "inc-kernel":
            kernels[f[1]] = f[2]
    failed = [l.split()[1] for l in open(f"{d}/inc.log") if l.startswith("inc-failed")] if os.path.exists(f"{d}/inc.log") else []
    return os.path.basename(d), cpu, avx512, policy, dist, search, pages, levels, kernels, failed


def drift(search, a, b):
    xs = [search[b][ef][1] / search[a][ef][1] for ef in search.get(a, {}) if ef in search.get(b, {})]
    return max((abs(x - 1) for x in xs), default=None)


PART1 = [
    ("cache avx512/avx2", "cache", "avx512", "avx2"),
    ("file-huge avx512/avx2", "file-huge", "avx512", "avx2"),
    ("anon avx512/avx2", "anon", "avx512", "avx2"),
    ("cache faiss-avx512/avx512", "cache", "faiss-avx512", "avx512"),
    ("cache faiss-avx2/avx2", "cache", "faiss-avx2", "avx2"),
    ("anon faiss-avx512/avx512", "anon", "faiss-avx512", "avx512"),
]
PART2 = [
    ("warm/faiss", WARM, "faiss"),
    ("warm-512/faiss", WARM512, "faiss"),
    ("warm-512/warm", WARM512, WARM),
    ("faiss/faiss-avx2", "faiss", "faiss-avx2"),
    ("plain/faiss", "chassis-py-batch", "faiss"),
    ("warm/plain", WARM, "chassis-py-batch"),
]


def main():
    medians = defaultdict(list)  # (part, ratio, cpu) -> values
    for d in sorted(glob.glob(f"{sys.argv[1]}/*")):
        name, cpu, avx512, policy, dist, search, pages, levels, kernels, failed = job(d)
        label = short(cpu)
        print(f"\n== {name}  {cpu}  ({label}, avx512f {avx512})  {policy}")
        if failed:
            print(f"   FAILED: {failed}")

        # Part 1.
        ok1 = bool(dist)
        for memory, arms in dist.items():
            noise = abs(arms["avx2-again"] / arms["avx2"] - 1)
            huge_ok = memory == "cache" or arms["share"] >= 0.9
            ok1 &= noise <= 0.05 and huge_ok
            cells = "  ".join(f"{a} {v:.1f}" for a, v in arms.items() if a != "share")
            print(f"   {memory:9} huge {arms['share']:.3f}  {cells}  again {noise:.1%}{'' if noise <= 0.05 and huge_ok else '  CHECK FAILS'}")
        for ratio, memory, a, b in PART1:
            if ok1 and a in dist.get(memory, {}) and b in dist.get(memory, {}):
                medians[(1, ratio, label)].append(dist[memory][a] / dist[memory][b])

        # Part 2.
        noise_f, noise_c = drift(search, "faiss", "faiss-again"), drift(search, WARM, AGAIN)
        want_level = (lambda l: l.startswith("AVX512")) if avx512 else (lambda l: l == "AVX2")
        level_ok = want_level(levels.get("faiss", "")) and levels.get("faiss-avx2") == "AVX2"
        kernel_ok = kernels.get(WARM512) == ("avx512" if avx512 else "avx2, no avx512f")
        huge = {e: pages.get(e, {}).get("FilePmdMapped", 0) * 1024 / VECTOR_BYTES for e in (WARM, WARM512, AGAIN)}
        recall = max((abs(search[WARM512][ef][0] - search[WARM][ef][0]) for ef in search.get(WARM, {}) if ef in search.get(WARM512, {})), default=None)
        ok2 = (noise_f is not None and noise_f <= 0.05 and noise_c is not None and noise_c <= 0.05
               and level_ok and kernel_ok and min(huge.values()) >= 0.9 and recall is not None and recall <= 0.002)
        print(f"   faiss levels {levels}; kernel {kernels}; huge {', '.join(f'{v:.1%}' for v in huge.values())}")
        print(f"   drift faiss {noise_f if noise_f is None else f'{noise_f:.1%}'}, chassis {noise_c if noise_c is None else f'{noise_c:.1%}'}; "
              f"recall 512 vs avx2 {recall}{'' if ok2 else '  PART 2 LEFT OUT'}")
        for target in (0.95, 0.99):
            q = {e: at(v, target) for e, v in search.items()}
            line = [f"{r} {q[a] / q[b]:.3f}" for r, a, b in PART2 if q.get(a) and q.get(b)]
            print(f"   @{target}: " + ", ".join(line))
            if ok2 and target == 0.95:
                for r, a, b in PART2:
                    if q.get(a) and q.get(b):
                        medians[(2, r, label)].append(q[a] / q[b])

    print("\n\n== Each CPU's median (jobs)")
    for part, ratios in ((1, PART1), (2, PART2)):
        print(f"\n-- part {part}" + (" (time a distance)" if part == 1 else " (queries a second at 0.95 recall)"))
        cpus = sorted({c for p, _, c in medians if p == part})
        for r, *_ in ratios:
            cells = [f"{c} {statistics.median(medians[(part, r, c)]):.3f} ({len(medians[(part, r, c)])})" for c in cpus if medians.get((part, r, c))]
            if cells:
                print(f"   {r:26} " + "  ".join(cells))


if __name__ == "__main__":
    main()
