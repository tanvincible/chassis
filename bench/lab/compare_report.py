"""Incumbents RUN 14 (compare-plan.md): checks per job, then each CPU's median ratios.

    python bench/lab/compare_report.py <dir of downloaded artifacts>
"""

import glob
import os
import re
import statistics
import sys
from collections import defaultdict

DIMS = {"sift-128": 128, "gist-960": 960, "dbpedia-openai3-1536": 1536}


def at(points: dict, target: float):
    """Queries a second at `target` recall, linear between the two `ef` around it."""
    pts = [points[ef] for ef in sorted(points)]
    for (r1, q1), (r2, q2) in zip(pts, pts[1:]):
        if r1 <= target <= r2:
            return q1 + (q2 - q1) * (target - r1) / (r2 - r1) if r2 > r1 else q1
    return None


def job(d):
    cpu = next(l.split(":", 1)[1].strip() for l in open(f"{d}/lscpu.txt") if l.startswith("Model name"))
    name = os.path.basename(d).replace("incumbents-", "")
    dataset = next(k for k in DIMS if name.startswith(k))
    search, pages, libs, warmed, n = defaultdict(dict), {}, {}, {}, 0
    for line in open(f"{d}/inc.tsv"):
        f = line.rstrip("\n").split("\t")
        if f[0] == "inc" and f[3] == "search":
            search[f[1]][int(f[4])] = (float(f[5]), float(f[6]))
            n = int(f[2])
        elif f[0] == "inc-pages":
            pages[f[1]] = {k: int(v) for k, v in re.findall(r"(\w+) (\d+) kB", f[2])}
        elif f[0] == "inc-lib":
            libs[f[1]] = f[2].split(" /")[0]
        elif f[0] == "inc-warmed":
            warmed[f[1]] = f"{f[2]}, {f[3]}"
    failed = [l.split()[1] for l in open(f"{d}/inc.log") if l.startswith("inc-failed")] if os.path.exists(f"{d}/inc.log") else []
    return name, dataset, cpu, n, search, pages, libs, warmed, failed


def short(cpu):
    for pat, label in [("7763", "Zen 3"), ("9V74", "Zen 4"), ("9V45", "Zen 5"), ("Neoverse", "N2")]:
        if pat in cpu:
            return label
    return re.sub(r".*(Xeon\S*|Platinum|Gold)\s*", "Xeon ", cpu).replace("CPU @", "@")[:24]


ARMS = {
    "full": "chassis-py-batch",
    "half": "chassis-py-half-batch",
    "warm full": "chassis-py-batch-warm",
    "warm half": "chassis-py-half-batch-warm",
}
RATIOS = [
    ("full/hnswlib", "chassis-py-batch", "hnswlib"),
    ("half/hnswlib", "chassis-py-half-batch", "hnswlib"),
    ("full/faiss", "chassis-py-batch", "faiss"),
    ("half/faiss", "chassis-py-half-batch", "faiss"),
    ("warm full/faiss", "chassis-py-batch-warm", "faiss"),
    ("warm half/faiss", "chassis-py-half-batch-warm", "faiss"),
    ("warm full/full", "chassis-py-batch-warm", "chassis-py-batch"),
    ("warm half/half", "chassis-py-half-batch-warm", "chassis-py-half-batch"),
    ("pano/faiss", "faiss-pano", "faiss"),
    ("pano/best", "faiss-pano", "best"),
    ("full/pano", "chassis-py-batch", "faiss-pano"),
    ("half/pano", "chassis-py-half-batch", "faiss-pano"),
]


def main():
    medians = defaultdict(lambda: defaultdict(list))  # (dataset, target) -> (ratio, cpu) -> values
    for d in sorted(glob.glob(f"{sys.argv[1]}/*")):
        name, dataset, cpu, n, search, pages, libs, warmed, failed = job(d)
        dims = DIMS[dataset]
        print(f"\n== {name}  ({cpu})")
        print("   libs: " + "; ".join(sorted(set(libs.values()))))
        if failed:
            print(f"   FAILED: {failed}")
        drift = [search["faiss-again"][ef][1] / search["faiss"][ef][1] for ef in search.get("faiss", {}) if ef in search.get("faiss-again", {})]
        noisy = not drift or max(abs(x - 1) for x in drift) > 0.05
        print(f"   faiss-again/faiss by ef: {' '.join(f'{x:.3f}' for x in drift)}{'  NOISY: left out' if noisy else ''}")
        huge_ok = True
        for label, engine in ARMS.items():
            pmd = pages.get(engine, {}).get("FilePmdMapped", -1) * 1024
            vectors = n * dims * (2 if "half" in label else 4)
            asks = "warm" in label and dims * (2 if "half" in label else 4) >= 2048
            share = pmd / vectors if vectors else 0
            ok = share >= 0.9 if asks else pmd == 0
            huge_ok &= ok or "half" in label and not asks and pmd == 0
            print(f"   {label:9} FilePmdMapped {pmd / 1e6:7.1f} MB = {share:5.1%} of vectors; warm asks: {asks}; {'ok' if ok else 'CHECK FAILS'}"
                  + (f"; warmed {warmed.get(engine.replace('chassis-py-batch-warm', 'chassis-full.chassis').replace('chassis-py-half-batch-warm', 'chassis-half.chassis'), '')}" if "warm" in label else ""))
        for target in (0.95, 0.99):
            q = {e: at(v, target) for e, v in search.items()}
            chassis = [q[e] for e in ARMS.values() if q.get(e)]
            q["best"] = max(chassis) if chassis else None
            line = []
            for ratio, a, b in RATIOS:
                if q.get(a) and q.get(b):
                    r = q[a] / q[b]
                    line.append(f"{ratio} {r:.3f}")
                    usable = not noisy and (huge_ok or "warm" not in ratio)
                    if usable:
                        medians[(dataset, target)][(ratio, short(cpu))].append(r)
            print(f"   @{target}: " + ", ".join(f"{e} {v:.0f}" for e, v in q.items() if v) )
            print(f"          " + ", ".join(line))
        for plain, warm in (("chassis-py-batch", "chassis-py-batch-warm"), ("chassis-py-half-batch", "chassis-py-half-batch-warm")):
            control = dims * (2 if "half" in plain else 4) < 2048
            if control and q.get(plain):
                pass
    print("\n\n== Each CPU's median (jobs)")
    for (dataset, target), table in sorted(medians.items()):
        print(f"\n-- {dataset} at {target} recall")
        cpus = sorted({c for _, c in table})
        for ratio, _, _ in RATIOS:
            cells = [f"{c} {statistics.median(table[(ratio, c)]):.3f} ({len(table[(ratio, c)])})" for c in cpus if table.get((ratio, c))]
            if cells:
                print(f"   {ratio:16} " + "  ".join(cells))


if __name__ == "__main__":
    main()
