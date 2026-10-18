"""distbench RUNs 1 and 2 and incumbents RUNs 10 to 12 (`dist-plan.md`), from their artifacts in the current directory.

    gh run download <distbench run>; gh run download <incumbents run>; python dist_report.py
"""
import collections, glob, re, statistics

TARGET = 0.95


def cpu(d):
    model = next(l.split(":", 1)[1].strip() for l in open(f"{d}/lscpu.txt") if l.startswith("Model name"))
    return {"AMD EPYC 7763 64-Core Processor": "Zen 3", "AMD EPYC 9V74 80-Core Processor": "Zen 4",
            "AMD EPYC 9V45 96-Core Processor": "Zen 5"}.get(model, model)


def at(points, target=TARGET):
    """Linear interpolation at recall `target` over (ef, recall, value) points."""
    pts = sorted(points)
    for (_, r0, v0), (_, r1, v1) in zip(pts, pts[1:]):
        if r0 <= target <= r1:
            return v0 + (target - r0) / (r1 - r0) * (v1 - v0)
    return float("nan")


def part1():
    by_cpu = collections.defaultdict(list)
    for d in sorted(glob.glob("distbench-*")):
        ns, share, policy, collapse = {}, {}, "?", []
        for line in open(f"{d}/distbench.tsv"):
            f = line.rstrip("\n").split("\t")
            if f[1] == "policy":
                policy = f[2]
                continue
            if f[1] == "collapse":
                collapse.append(f"{f[2]} {f[3]} {f[4]}")
                continue
            share[f[1]] = float(f[2])
            ns[(f[1], f[3])] = float(f[4])
        noisy = [m for m in share if m != "cache" and abs(ns[(m, "now-again")] / ns[(m, "now")] - 1) > 0.05]
        pages_ok = share["anon"] >= 0.9 and share.get("anon-4k", 0) <= 0.05
        print(f"{d}: {cpu(d)}, {policy}, shares " + " ".join(f"{m} {s:.3f}" for m, s in share.items())
              + (f", NOISY in {noisy}" if noisy else "") + ("" if pages_ok else ", PAGES FAIL")
              + (f"\n   collapse: {'; '.join(collapse)}" if collapse else ""))
        if not noisy and pages_ok:
            by_cpu[cpu(d)].append((ns, share))
    arms = ["now", "pf0", "pf8", "pf16", "pf32", "pf96", "faiss", "now-again"]
    memories = list(dict.fromkeys(m for j in by_cpu.values() for x in j for m in x[1]))
    for name, jobs in by_cpu.items():
        print(f"\n### {name}, {len(jobs)} jobs: ns a distance, the median over jobs\n")
        print("| memory | huge | " + " | ".join(arms) + " |")
        print("| --- | --- |" + " --- |" * len(arms))
        for m in memories:
            med = lambda a: statistics.median(j[0][(m, a)] for j in jobs)
            share = statistics.median(j[1][m] for j in jobs)
            print(f"| {m} | {share:.2f} | " + " | ".join(f"{med(a):.1f}" for a in arms) + " |")
        med = lambda m, a: statistics.median(j[0][(m, a)] for j in jobs)
        if "cache" in memories:
            print(f"\n1. now anon-4k / anon: {med('anon-4k', 'now') / med('anon', 'now'):.3f}")
            print(f"2. now file / anon-4k: {med('file', 'now') / med('anon-4k', 'now'):.3f}")
            print(f"3. anon pf0 / faiss: {med('anon', 'pf0') / med('anon', 'faiss'):.3f}")
            print(f"4. cache pf0 / faiss: {med('cache', 'pf0') / med('cache', 'faiss'):.3f}")
        for m in memories:
            print(f"   now {m} / file-huge: {med(m, 'now') / med('file-huge', 'now'):.3f}")


def kb(text, key):
    m = re.search(rf"{key}:? (\d+) kB", text)
    return int(m.group(1)) if m else None


def chassis(d):
    """lab.tsv's searches: queries a second by (tag, ef), recall by ef, and whether every tag found the same."""
    qps, recall = collections.defaultdict(list), collections.defaultdict(set)
    for line in open(f"{d}/lab.tsv"):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 7 and f[3] == "search":
            qps[(f[1], int(f[4]))].append(float(f[6]))
            recall[int(f[4])].add(f[5])
    return qps, {e: float(next(iter(r))) for e, r in recall.items()}, all(len(r) == 1 for r in recall.values())


def part2():
    by_cpu = collections.defaultdict(list)
    for d in sorted(glob.glob("incumbents-*/inc.tsv")):
        d = d.split("/")[0]
        qps, rec, same = chassis(d)
        pages = collections.defaultdict(list)
        for line in open(f"{d}/pages.txt"):
            m = re.match(r"pages chassis (\S+): (.*)", line)
            if m:
                pages[m.group(1)].append(kb(m.group(2), "FilePmd"))
        fq, fpages, level = collections.defaultdict(dict), {}, "?"
        for line in open(f"{d}/inc.tsv"):
            f = line.rstrip("\n").split("\t")
            if f[0] == "inc-lib":
                level = f[2].split("level ")[-1]
            elif f[0] == "inc-pages":
                fpages[f[1]] = kb(f[2], "AnonHuge")
            elif f[0] == "inc" and len(f) >= 7 and f[3] == "search":
                fq[f[1]][int(f[4])] = (float(f[5]), float(f[6]))
        out = {t: at([(e, rec[e], statistics.median(qps[(t, e)])) for e in rec]) for t, _ in qps}
        for e in fq:
            out[e] = at([(ef, r, v) for ef, (r, v) in fq[e].items()])
        drift = max((abs(fq["faiss-again"][ef][1] / fq["faiss"][ef][1] - 1) for ef in fq.get("faiss", {})), default=1)
        bad_pages = [t for t, v in pages.items() if (min(v) < 500_000 if t.endswith("-huge") else max(v) > 10_000)]
        bad_pages += [e for e, v in fpages.items() if (v > 10_000 if e.endswith("-4k") else v < 500_000)]
        print(f"{d}: {cpu(d)}, FAISS level {level}, recall same {same}, faiss-again drift {drift:.3f}, "
              f"FilePmd {dict(pages)}, FAISS AnonHuge {fpages}" + (f", PAGES FAIL {bad_pages}" if bad_pages else ""))
        print("   " + "  ".join(f"{t} {v:.0f}" for t, v in out.items()))
        if drift <= 0.05 and same:
            by_cpu[cpu(d)].append((out, bad_pages))
    for name, jobs in by_cpu.items():
        print(f"\n### {name}, {len(jobs)} jobs: queries a second at 0.95 recall, the median over jobs\n")
        keys = [k for k in jobs[0][0]]
        med = {k: statistics.median(j[0][k] for j in jobs) for k in keys}
        print("| " + " | ".join(keys) + " |\n|" + " --- |" * len(keys))
        print("| " + " | ".join(f"{med[k]:.0f}" for k in keys) + " |")
        own = "pf32" if name == "Zen 4" else "pf8"
        print(f"\n5. {own}-huge / {own}: {med[own + '-huge'] / med[own]:.3f}")
        print(f"6. faiss / faiss-4k: {med['faiss'] / med['faiss-4k']:.3f}")
        print(f"7. {own}-huge / faiss: {med[own + '-huge'] / med['faiss']:.3f}")
        for h in ("", "-huge"):
            print(f"8. pfN{h} / {own}{h}: " + ", ".join(f"pf{n} {med[f'pf{n}{h}'] / med[own + h]:.3f}" for n in (0, 8, 32)))
        print(f"   {own} / faiss: {med[own] / med['faiss']:.3f}; faiss-4k / {own}: {med['faiss-4k'] / med[own]:.3f}")


def part3():
    """RUNs 11 to 13: Chassis alone, each tag against `off`; RUN 13 has an index per subdirectory."""
    by = collections.defaultdict(list)
    for d in sorted(p.rsplit("/", 1)[0] for p in glob.glob("incumbents-*/pages.txt") + glob.glob("incumbents-*/*/pages.txt")):
        if glob.glob(f"{d}/inc.tsv"):
            continue
        qps, rec, same = chassis(d)
        tags = list(dict.fromkeys(t for t, _ in qps))
        out = {t: at([(e, rec[e], statistics.median(qps[(t, e)])) for e in rec]) for t in tags}
        pmd = collections.defaultdict(list)
        for line in open(f"{d}/pages.txt"):
            m = re.match(r"pages chassis (\S+): (.*)", line)
            if m:
                pmd[m.group(1)].append(kb(m.group(2), "FilePmd") // 1024)
        spread = max(max(qps[k]) / min(qps[k]) - 1 for k in qps)
        print(f"{d}: {cpu(d)}, recall same {same}, rounds apart {spread:.2f}, FilePmd MB {dict(pmd)}")
        print("   " + "  ".join(f"{t} {v:.0f}" for t, v in out.items())
              + "   " + "  ".join(f"{t}/off {out[t] / out['off']:.3f}" for t in tags if t != "off"))
        if same:
            data = d.split("/")[1] if "/" in d else d.split("-ubuntu")[0].removeprefix("incumbents-")
            by[(data, cpu(d))].append(out)
    for (data, name), jobs in sorted(by.items()):
        tags = [t for t in jobs[0] if t != "off"]
        print(f"{data}, {name}, {len(jobs)} jobs, the median over jobs of each tag / off: "
              + "  ".join(f"{t} {statistics.median(j[t] / j['off'] for j in jobs):.3f}" for t in tags))


if glob.glob("distbench-*"):
    part1()
if glob.glob("incumbents-*/inc.tsv"):
    part2()
if glob.glob("incumbents-*/pages.txt") + glob.glob("incumbents-*/*/pages.txt"):
    part3()
