"""The pull-request benchmark (.github/workflows/bench.yml): main against the pull request.

    python bench/compare.py run --base B --head H --data DIR --dataset NAME --n N
                                --precision full|half --label LABEL --out result.json
    python bench/compare.py report RESULTS_DIR > comment.md

`run` builds and searches with both builds of chassis-core/examples/bench.rs on one machine,
taking turns, so that the machine's speed cancels out. `report` turns every job's result into the
pull request's comment, and exits 1 if anything got markedly slower or lost recall.
"""

import argparse
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path

EFS = [32, 64, 128]
ROUNDS = 5
# Within this of 1.0, a ratio is noise between two runs on one machine.
NOISE = 0.03
# Below this, the check fails.
FAIL = 0.90
# A recall this much lower at the same ef fails it too.
RECALL_DROP = 0.005


def lines(command):
    out = subprocess.run(command, check=True, capture_output=True, text=True).stdout
    return [json.loads(line) for line in out.splitlines() if line.startswith("{")]


def cpu() -> str:
    try:
        out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if line.startswith("Model name:"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def run(a) -> None:
    work = Path(a.out).parent
    truth, index = work / "truth.u32", work / "index.chassis"
    data = [a.data, a.dataset]
    lines([a.head, "truth", *data, str(a.n), str(truth)])

    # Each builds twice, in turns; the faster of its two counts. Searches use main's index.
    builds = {"base": [], "head": []}
    for side in ["base", "head", "base", "head"]:
        binary = a.base if side == "base" else a.head
        [result] = lines([binary, "build", *data, str(a.n), str(index), a.precision])
        builds[side].append(result["seconds"])

    # Each round runs both, the first going second the next time; a round's ratio compares two
    # runs a minute apart, and the median over rounds is reported.
    searched = {"base": {ef: [] for ef in EFS}, "head": {ef: [] for ef in EFS}}
    recall = {"base": {}, "head": {}}
    order = [("base", a.base), ("head", a.head)]
    for _ in range(ROUNDS):
        for side, binary in order:
            for result in lines([binary, "search", *data, str(index), str(truth), *map(str, EFS)]):
                searched[side][result["ef"]].append(result["qps"])
                recall[side][result["ef"]] = result["recall"]
        order.reverse()

    Path(a.out).write_text(json.dumps({
        "label": a.label,
        "arch": platform.machine(),
        "cpu": cpu(),
        "build": {side: min(times) for side, times in builds.items()},
        "search": [
            {
                "ef": ef,
                "base": statistics.median(searched["base"][ef]),
                "head": statistics.median(searched["head"][ef]),
                "ratio": statistics.median(
                    h / b for b, h in zip(searched["base"][ef], searched["head"][ef])
                ),
                "base_recall": recall["base"][ef],
                "head_recall": recall["head"][ef],
            }
            for ef in EFS
        ],
    }, indent=1))


def arch_name(arch: str) -> str:
    return "ARM" if arch in ("aarch64", "arm64") else "x86"


def verdict(ratio: float) -> str:
    if ratio < 1 - NOISE:
        return f"**{(1 - ratio) * 100:.0f}% slower**"
    if ratio > 1 + NOISE:
        return f"{(ratio - 1) * 100:.0f}% faster"
    return "no change"


def report(directory: str) -> int:
    results = [json.loads(p.read_text()) for p in sorted(Path(directory).rglob("*.json"))]
    if not results:
        print("No benchmark results: every benchmark job failed. See the workflow's log.")
        return 1
    labels = list(dict.fromkeys(r["label"] for r in results))
    archs = sorted({arch_name(r["arch"]) for r in results}, key=lambda a: a != "x86")
    cell, failures, worse = {}, [], []
    for r in results:
        arch = arch_name(r["arch"])
        search = statistics.median(s["ratio"] for s in r["search"])
        build = r["build"]["base"] / r["build"]["head"]
        drop = max(s["base_recall"] - s["head_recall"] for s in r["search"])
        cell[(r["label"], arch)] = (search, build, r)
        for what, ratio in ((f"search, {r['label']}", search), (f"batch build, {r['label']}", build)):
            if ratio < FAIL:
                failures.append(f"{what} on {arch} is {(1 - ratio) * 100:.0f}% slower")
            elif ratio < 1 - NOISE:
                worse.append(f"{what} on {arch} is {(1 - ratio) * 100:.0f}% slower")
        if drop > RECALL_DROP:
            failures.append(f"recall, {r['label']} on {arch}, is {drop:.3f} lower at the same ef")

    out = ["<!-- chassis-bench -->", "### Benchmark: this pull request against main", ""]
    if failures:
        out += ["**Slower:** " + "; ".join(failures) + ".", ""]
    elif worse:
        out += ["Slightly slower, possibly noise: " + "; ".join(worse) + ".", ""]
    else:
        out += ["No slowdown beyond noise (±3%).", ""]

    keys = [(label, arch) for label in labels for arch in archs if (label, arch) in cell]
    names = ", ".join(f'"{label} {arch}"' for label, arch in keys)
    bars = ", ".join(f"{cell[k][0]:.2f}" for k in keys)
    low = min(0.8, min(cell[k][0] for k in keys) - 0.05)
    high = max(1.2, max(cell[k][0] for k in keys) + 0.05)
    out += [
        "```mermaid",
        "xychart-beta",
        '  title "Search speed, this pull request over main"',
        f"  x-axis [{names}]",
        f'  y-axis "times main" {low:.2f} --> {high:.2f}',
        f"  bar [{bars}]",
        f"  line [{', '.join('1' for _ in keys)}]",
        "```",
        "",
        "| | " + " | ".join(f"Search, {a}" for a in archs) + " | " + " | ".join(f"Build, {a}" for a in archs) + " |",
        "| --- |" + " --- |" * (2 * len(archs)),
    ]
    for label in labels:
        row = [verdict(cell[(label, a)][0]) if (label, a) in cell else "–" for a in archs]
        row += [verdict(cell[(label, a)][1]) if (label, a) in cell else "–" for a in archs]
        out.append(f"| {label} | " + " | ".join(row) + " |")
    out += [
        "",
        "Queries a second at the same `ef`, and batch-build time, for each build on the same runner, "
        f"taking turns ({ROUNDS} rounds of search). Recall is compared at the same `ef`.",
        "",
        "<details><summary>Per ef</summary>",
        "",
        "| Case | CPU | ef | main q/s | PR q/s | Ratio | Recall main → PR |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label, arch in keys:
        r = cell[(label, arch)][2]
        for s in r["search"]:
            out.append(
                f"| {label}, {arch} | {r['cpu']} | {s['ef']} | {s['base']:.0f} | {s['head']:.0f} | "
                f"{s['ratio']:.2f} | {s['base_recall']:.3f} → {s['head_recall']:.3f} |"
            )
    out += ["", "</details>"]
    print("\n".join(out))
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run")
    for flag in ("base", "head", "data", "dataset", "precision", "label", "out"):
        r.add_argument(f"--{flag}", required=True)
    r.add_argument("--n", type=int, required=True)
    p = sub.add_parser("report")
    p.add_argument("directory")
    a = parser.parse_args()
    if a.command == "run":
        run(a)
    else:
        sys.exit(report(a.directory))


if __name__ == "__main__":
    main()
