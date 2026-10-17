"""RUN 9: distances and queries a second at 0.95 recall, Chassis and FAISS, per job."""
import collections, glob, re, statistics, sys

TARGET = 0.95

def at(points, target=TARGET):
    """Linear interpolation of value at recall `target` over (ef, recall, value) points."""
    pts = sorted(points)
    for (e0, r0, v0), (e1, r1, v1) in zip(pts, pts[1:]):
        if r0 <= target <= r1:
            return v0 + (target - r0) / (r1 - r0) * (v1 - v0)
    return float("nan")

rows = []
for d in sorted(glob.glob("incumbents-*")):
    cpu = next(l.split(":", 1)[1].strip() for l in open(f"{d}/lscpu.txt") if l.startswith("Model name"))
    flags = next((l for l in open(f"{d}/lscpu.txt") if l.startswith("Flags")), "")
    name = re.sub(r"incumbents-|-ubuntu", "", d)
    qps = collections.defaultdict(list)          # (tag, ef) -> [qps]
    recall, count = {}, {}
    for line in open(f"{d}/lab.tsv"):
        f = line.rstrip("\n").split("\t")
        if len(f) < 8 or f[3] != "search":
            continue
        tag, ef, r, q = f[1], int(f[4]), float(f[5]), float(f[6])
        if tag.startswith("count-"):
            count[(tag[6:], ef)] = (r, float(f[7]), float(f[8]))
        else:
            qps[(tag, ef)].append(q)
            recall[(tag, ef)] = r
    fq, fd, fh, level = {}, {}, {}, "?"
    for line in open(f"{d}/inc.tsv"):
        f = line.rstrip("\n").split("\t")
        if f[0] == "inc-lib":
            level = f[2].split("level ")[-1]
        if f[0] != "inc" or len(f) < 7:
            continue
        what, ef, r, v = f[3], f[4], float(f[5]), float(f[6])
        if what in ("search", "distances", "hops"):
            {"search": fq, "distances": fd, "hops": fh}[what][int(ef)] = (r, v)
    out = {"job": name, "cpu": cpu, "avx512": "avx512f" in flags, "level": level}
    # Harness check: the counted search finds the uncounted one's recall.
    out["recall_same"] = all(abs(count[(t, e)][0] - recall[(t, e)]) < 1e-9 for (t, e) in count)
    for tag in ("full", "half"):
        efs = sorted(e for (t, e) in recall if t == tag)
        out[f"{tag}_qps"] = at([(e, recall[(tag, e)], statistics.median(qps[(tag, e)])) for e in efs])
        out[f"{tag}_dist"] = at([(e, count[(tag, e)][0], count[(tag, e)][1]) for e in efs])
        out[f"{tag}_hops"] = at([(e, count[(tag, e)][0], count[(tag, e)][2]) for e in efs])
    out["faiss_qps"] = at([(e, r, v) for e, (r, v) in fq.items()])
    out["faiss_dist"] = at([(e, fd[e][0], fd[e][1]) for e in fd])
    out["faiss_hops"] = at([(e, fh[e][0], fh[e][1]) for e in fh])
    for k in ("full", "half", "faiss"):
        out[f"{k}_ns"] = 1e9 / (out[f"{k}_qps"] * out[f"{k}_dist"])
    rows.append(out)

print("| Job | CPU | FAISS level | recall same | distances, full / FAISS | hops, full / FAISS | ns a distance: full, FAISS, ratio | half ns / FAISS ns | QPS FAISS / full |")
print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
for o in rows:
    print(f"| {o['job']} | {o['cpu'].replace('AMD EPYC ', 'EPYC ')} | {o['level']} | {o['recall_same']} | "
          f"{o['full_dist']:.0f} / {o['faiss_dist']:.0f} = {o['full_dist']/o['faiss_dist']:.3f} | "
          f"{o['full_hops']:.0f} / {o['faiss_hops']:.0f} | "
          f"{o['full_ns']:.0f}, {o['faiss_ns']:.0f}, {o['full_ns']/o['faiss_ns']:.3f} | {o['half_ns']/o['faiss_ns']:.3f} | "
          f"{o['faiss_qps']/o['full_qps']:.3f} |")
