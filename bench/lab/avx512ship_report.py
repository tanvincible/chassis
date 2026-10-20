"""avx512ship RUN 1 (avx512ship-plan.md): checks per job, then each CPU's median ratios.

    python bench/lab/avx512ship_report.py <dir of downloaded artifacts>
"""

import glob
import os
import re
import statistics
import sys
from collections import defaultdict

ARMS = ("full", "half", "warm")


def short(cpu):
    for pat, label in [("7763", "Zen 3"), ("9V74", "Zen 4"), ("9V45", "Zen 5")]:
        if pat in cpu:
            return label
    m = re.search(r"(\d{4}[A-Z]*(-C)?)", cpu)
    return f"Xeon {m.group(1)}" if m else cpu[:24]


def read(d, name):
    path = f"{d}/{name}"
    return open(path).read().splitlines() if os.path.exists(path) else []


def job(d):
    lscpu = "\n".join(read(d, "lscpu.txt"))
    cpu = short(re.search(r"Model name:\s*(.*)", lscpu).group(1).strip())
    avx512 = " avx512f " in f" {re.search(r'Flags:(.*)', lscpu).group(1)} "
    qps = defaultdict(list)  # (config, arm, bin, ef) -> [qps per round]
    digests = defaultdict(set)  # (config, arm, ef) -> {digest}
    mains = defaultdict(set)  # (config, arm, ef) -> main's digests alone
    builds = defaultdict(float)  # (config, bin) -> seconds over both builds
    for line in read(d, "lab.tsv"):
        f = line.split("\t")
        if len(f) < 6 or f[0] != "chassis":
            continue
        if f[3] in ("search", "digest"):
            b, arm, config = f[1].split("/")
            key = (config, arm, int(f[4]))
            if f[3] == "search":
                qps[key[:2] + (b, key[2])].append(float(f[6]))
            else:
                digests[key].add(f[5])
                if b == "main":
                    mains[key].add(f[5])
        elif f[3] == "build_batch_s":
            _, b, config = f[1].split("/")
            builds[(config, b)] += float(f[6])
    pmd = defaultdict(list)  # (config, arm) -> FilePmdMapped kB per search
    for line in read(d, "pages.txt"):
        m = re.match(r"pages chassis (\w+)/(\w+)/([\d:]+): .*FilePmd (\d+) kB", line)
        if m:
            pmd[(m.group(3), m.group(2))].append(int(m.group(4)))
    cmps = [l.split("\t") for l in read(d, "cmp.tsv")]
    unit = "\n".join(read(d, "unit.txt"))
    thp = "".join(read(d, "thp.txt"))
    return cpu, avx512, qps, digests, mains, builds, pmd, cmps, unit, thp


def ratio(qps, config, arm):
    efs = sorted({k[3] for k in qps if k[:2] == (config, arm)})
    per_ef = []
    for ef in efs:
        m, b = qps.get((config, arm, "main", ef)), qps.get((config, arm, "branch", ef))
        if m and b:
            per_ef.append(statistics.median(b) / statistics.median(m))
    return statistics.median(per_ef) if per_ef else None


def main(root):
    rows = defaultdict(lambda: defaultdict(list))  # (cpu, config) -> measure -> [ratio]
    configs = set()
    problems = []
    for d in sorted(glob.glob(f"{root}/*/")):
        name = os.path.basename(d.rstrip("/"))
        cpu, avx512, qps, digests, mains, builds, pmd, cmps, unit, thp = job(d)
        tag = f"{name} ({cpu}, avx512f {'yes' if avx512 else 'no'}, THP {thp})"
        if "test result: ok" not in unit:
            problems.append(f"{tag}: distance tests did not pass: {unit[-200:]!r}")
        for key, ds in sorted(digests.items()):
            if len(ds) != 1:
                what = "main differs from itself" if len(mains[key]) > 1 else "branch differs from main"
                problems.append(f"{tag}: {key[0]} {key[1]} ef {key[2]}: {what}")
        for c in cmps:
            print(f"{tag}: build {c[1]} {c[2]} vs {c[3]}: {c[4]}")
        for config in sorted({k[0] for k in qps}, key=lambda c: tuple(map(int, c.split(":")))):
            configs.add(config)
            d_, n = map(int, config.split(":"))
            r = {arm: ratio(qps, config, arm) for arm in ARMS}
            if r["half"] is None or abs(r["half"] - 1) > 0.03:
                print(f"{tag}: {config} left out, half A/A {r['half']}")
                continue
            vec_kb = n * d_ * 4 / 1024
            if r["warm"] is not None:
                least = min(pmd[(config, "warm")], default=0)
                if least < 0.9 * vec_kb:
                    print(f"{tag}: {config} warm left out, FilePmd {least} kB of {vec_kb:.0f}")
                    r["warm"] = None
            small = max(pmd[(config, "full")], default=0)
            if small > 0.1 * vec_kb:
                print(f"{tag}: {config} full arm had {small} kB on huge pages")
            bm, bb = builds.get((config, "main")), builds.get((config, "branch"))
            r["build"] = bm / bb if bm and bb else None  # main's time over the branch's
            print(f"{tag}: {config} " + " ".join(f"{k} {v:.3f}" for k, v in r.items() if v is not None))
            for k, v in r.items():
                if v is not None:
                    rows[(cpu, config)][k].append(v)
    print("\nPer CPU: branch over main, the median of clean jobs (range); build is main's time over the branch's")
    order = lambda c: tuple(map(int, c.split(":")))
    for cpu in sorted({k[0] for k in rows}):
        for config in sorted(configs, key=order):
            m = rows.get((cpu, config))
            if not m:
                continue
            cells = []
            for k in ("full", "warm", "half", "build"):
                xs = m.get(k)
                if xs:
                    cells.append(f"{k} {statistics.median(xs):.3f} ({min(xs):.3f}-{max(xs):.3f}, {len(xs)})")
            print(f"  {cpu:16} {config:12} " + "  ".join(cells))
    print("\nProblems:" if problems else "\nProblems: none")
    for p in problems:
        print("  " + p)


if __name__ == "__main__":
    main(sys.argv[1])
