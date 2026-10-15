"""Draws recall against queries a second for each engine, from a run of bench/lab/incumbents.py.

    python3 scripts/recall_chart.py bench/results/engines-2026-10-10.csv > docs/src/architecture/recall-qps.svg

One panel per dataset, on one machine each, from the run in which every engine is called with
all its queries at once; plain SVG, so it needs nothing installed.
"""

import csv
import math
import sys
from collections import defaultdict

RUN = "2026-10-10b"
PANELS = [("sift-128", "SIFT, 1M vectors of 128 dimensions"), ("dbpedia-1536", "dbpedia, 99k of 1,536")]
CPU = "Intel(R) Xeon(R) 6973P-C"
ENGINES = [
    ("chassis-py-batch", "Chassis", "#d9480f"),
    ("chassis-py-half-batch", "Chassis, half precision", "#f08c00"),
    ("faiss", "FAISS (HNSW)", "#1971c2"),
    ("hnswlib", "hnswlib", "#2f9e44"),
    ("usearch", "usearch", "#7048e8"),
]
W, H, LEFT, TOP, BOTTOM, GAP = 380, 300, 52, 30, 44, 20
QPS_MIN, QPS_MAX, RECALL_MIN = 200, 20_000, 0.90


def points(rows, dataset):
    found = defaultdict(list)
    for r in rows:
        if r["run"] == RUN and r["dataset"] == dataset and r["cpu"] == CPU and r["what"] == "search":
            found[r["engine"]].append((float(r["value"]), float(r["recall"])))
    return {engine: sorted(p) for engine, p in found.items()}


def main() -> None:
    rows = list(csv.DictReader(open(sys.argv[1])))
    width = 2 * W + GAP
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {H + 70}" '
        'font-family="system-ui, sans-serif" font-size="12" fill="#868e96">'
    ]
    for p, (dataset, title) in enumerate(PANELS):
        x0 = p * (W + GAP)
        pw, ph = W - LEFT - 10, H - TOP - BOTTOM
        x = lambda q: x0 + LEFT + pw * (math.log10(q) - math.log10(QPS_MIN)) / math.log10(QPS_MAX / QPS_MIN)
        y = lambda r: TOP + ph * (1 - (r - RECALL_MIN) / (1 - RECALL_MIN))
        out.append(f'<text x="{x0 + LEFT}" y="18" font-weight="600">{title}</text>')
        for r in (0.90, 0.92, 0.94, 0.96, 0.98, 1.00):
            out.append(f'<line x1="{x0 + LEFT}" x2="{x0 + LEFT + pw}" y1="{y(r):.1f}" y2="{y(r):.1f}" stroke="#868e96" stroke-opacity="0.25"/>')
            out.append(f'<text x="{x0 + LEFT - 6}" y="{y(r) + 4:.1f}" text-anchor="end">{r:.2f}</text>')
        for q in (300, 1000, 3000, 10000):
            out.append(f'<line x1="{x(q):.1f}" x2="{x(q):.1f}" y1="{TOP}" y2="{TOP + ph}" stroke="#868e96" stroke-opacity="0.25"/>')
            out.append(f'<text x="{x(q):.1f}" y="{TOP + ph + 16}" text-anchor="middle">{q:,}</text>')
        out.append(f'<text x="{x0 + LEFT + pw / 2}" y="{TOP + ph + 34}" text-anchor="middle">queries a second, one thread (log scale)</text>')
        out.append(f'<text transform="translate({x0 + 12},{TOP + ph / 2}) rotate(-90)" text-anchor="middle">recall@10</text>')
        found = points(rows, dataset)
        for engine, _, colour in ENGINES:
            pts = [(q, r) for q, r in found.get(engine, []) if r >= RECALL_MIN and QPS_MIN <= q <= QPS_MAX]
            if pts:
                path = " ".join(f"{x(q):.1f},{y(r):.1f}" for q, r in pts)
                out.append(f'<polyline points="{path}" fill="none" stroke="{colour}" stroke-width="2"/>')
                out.extend(f'<circle cx="{x(q):.1f}" cy="{y(r):.1f}" r="3" fill="{colour}"/>' for q, r in pts)
    lx = 10
    for _, label, colour in ENGINES:
        out.append(f'<rect x="{lx}" y="{H + 10}" width="12" height="3" fill="{colour}"/>')
        out.append(f'<text x="{lx + 17}" y="{H + 15}">{label}</text>')
        lx += 17 + 7 * len(label) + 24  # About 7 pixels a character at this size
    out.append(f'<text x="10" y="{H + 40}">Intel Xeon 6973P (a GitHub runner), 2026-10-10. Each engine through its Python package,</text>')
    out.append(f'<text x="10" y="{H + 56}">with every query in one call; M 16 and ef_construction 200 where an engine has them.</text>')
    out.append("</svg>")
    print("\n".join(out))


if __name__ == "__main__":
    main()
