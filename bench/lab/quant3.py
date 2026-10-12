"""The query in eight bits too: what rounding the query costs LVQ-8's recall (lab only).

    python bench/lab/quant3.py run <dataset> <data_dir> <out_dir> <n> [dims] [--no-graph]
    python bench/lab/quant3.py report <out_dir>...
    python bench/lab/quant3.py selftest

Data comes from `quant.py prepare`. The vectors are rounded as LVQ-8 (quant2.py's i8-vec-c), and
each query the same way, around the same center, in 65,536, 256 or 128 levels, or not at all.
Stage 1 ranks every vector by the rounded query, stage 2 searches Chassis built on the rounded
vectors with it; recall is always against the query as given. The plan, predictions and decision
rule are in quant-plan.md, "The query in eight bits", committed before any result.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import quant
import quant2

QUERIES = {"f32": None, "q16": 65536, "q8": 256, "q7": 128}
CENTERS = ["first1k", "cluster1k"]


def query(q: np.ndarray, center: np.ndarray, levels: int | None) -> np.ndarray:
    """The vectors' rounding: less the center, in `levels` steps over its own range, center back."""
    if levels is None:
        return q
    y = q - center
    return quant.affine(y, y.min(axis=1, keepdims=True), y.max(axis=1, keepdims=True), levels)[0] + center


def stage1(scorer, qs, test, gt_ids, gt_tenth, train):
    """Every vector ranked by the rounded queries; recall against the queries as given."""
    found = np.concatenate([np.argpartition(scorer.scores(q), quant.K, axis=1)[:, : quant.K] for q in np.array_split(qs, 50)])
    return quant.recall(found, gt_ids, gt_tenth, train, test)


def run(name: str, data_dir: Path, out: Path, n: int, dims: int | None, graph: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    train, test = quant.read(data_dir / f"{name}.train.f32")[:n], quant.read(data_dir / f"{name}.test.f32")
    if dims:
        train, test = quant.prepare.unit(train[:, :dims]), quant.prepare.unit(test[:, :dims])
    tag = f"{name}-{dims or train.shape[1]}-{n}"
    lines, per_query = [], {}

    def emit(**row):
        row = {"config": tag, **row}
        lines.append(row)
        print("\t".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row.values()), flush=True)

    start = time.time()
    gt_ids, gt_tenth = quant.truth(train, test)
    emit(stage="truth", seconds=time.time() - start)
    calib = quant.calibrations(train)
    centers = {c: data.mean(axis=0) for c, data in calib.items()}

    def ranked(scheme, c, scorer, qs):
        by_id, by_dist = stage1(scorer, qs, test, gt_ids, gt_tenth, train)
        per_query[f"s1/{scheme}/{c}/1"] = by_id
        emit(stage=1, scheme=scheme, calib=c, r=1, recall=float(by_id.mean()), recall_dist=float(by_dist.mean()), clipped=0.0)

    for c in calib:
        start = time.time()
        scorer = quant.Rounded(quant2.rounded("i8-vec-c", train, calib[c])[0])
        for q, levels in QUERIES.items():
            ranked(f"i8-vec-c:{q}", c, scorer, query(test, centers[c], levels))
        del scorer
        print(f"# stage 1 {c}: {time.time() - start:.0f} s", flush=True)
    ranked("half", None, quant.Rounded(train.astype(np.float16).astype(np.float32)), test)
    # Planted loss: every vector under another's id, ranked by the eight-bit queries.
    shuffled = quant.Rounded(train[np.random.default_rng(4).permutation(len(train))])
    ranked("shuffled", None, shuffled, query(test, centers["first1k"], 256))
    del shuffled

    if graph:
        from chassis import IndexOptions, VectorIndex

        d = train.shape[1]
        for scheme, c in [("half", None)] + [("i8-vec-c", c) for c in CENTERS]:
            precision = "half" if c is None else "full"
            data = train if c is None else quant2.rounded(scheme, train, calib[c])[0]
            # Each build searched with every query, so the query's own cost is paired.
            queries = {scheme: test} if c is None else {f"{scheme}:{q}": query(test, centers[c], v) for q, v in QUERIES.items()}
            for b in quant.BUILDS:
                path = out / f"{scheme}.chassis"
                path.unlink(missing_ok=True)
                start = time.time()
                with VectorIndex(path, d, IndexOptions(precision=precision)) as index:
                    index.add_batch(data)
                    index.flush()
                emit(stage="build", scheme=scheme, calib=c, build=b, seconds=time.time() - start)
                for ef in quant.EFS:
                    with VectorIndex(path, d, IndexOptions(precision=precision, ef_search=ef)) as index:
                        for arm, qs in queries.items():
                            ids, _ = index.search_batch(qs, k=quant.K)
                            by_id, by_dist = quant.recall(ids.astype(np.int64), gt_ids, gt_tenth, train, test)
                            per_query[f"s2/{arm}/{c}/{b}/{ef}/1"] = by_id
                            emit(stage=2, scheme=arm, calib=c, build=b, ef=ef, r=1,
                                 recall=float(by_id.mean()), recall_dist=float(by_dist.mean()))
                path.unlink()
            del data
    (out / f"{tag}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in lines))
    np.savez_compressed(out / f"{tag}.npz", **per_query)


# The pre-registered analysis: quant.py's loss and bar, with these arms and this decision.

quant.ARMS = [("half", None)] + [(f"i8-vec-c:{q}", c) for c in CENTERS for q in QUERIES]
quant.GRAPH_RERANK = [1]


def worst(results: dict, arm: str) -> float:
    return max(quant.loss(pq, arm, c, ef, 1)[0] for pq in results.values() for c in CENTERS for ef in (64, 128))


def decide(results: dict) -> str:
    def ok(q):
        return all(quant.passes(results, f"i8-vec-c:{q}", c, [1]) for c in CENTERS)

    close = [f"{q} ({worst(results, f'i8-vec-c:{q}'):.4f})" for q in QUERIES
             if ok(q) and worst(results, f"i8-vec-c:{q}") > quant.LIMIT - 0.002]
    note = f" Close calls: {', '.join(close)}." if close else ""
    if not ok("f32"):
        return "the follow-up does not reproduce: LVQ-8 with the query in f32 fails here, and nothing else is decided." + note
    if ok("q8"):
        return f"the integer kernel is next, with the query in eight bits (q7 {'passes' if ok('q7') else 'fails'}).{note}"
    if ok("q16"):
        return "the query stays wider than eight bits: the integer kernel is next, with the query in sixteen." + note
    return "eight bits waits: the query stays in f32, and its kernel's cost in cache stands." + note


def added(pq: dict, arm: str, c: str, ef: int):
    """What rounding the query costs on the same builds: the f32 query's recall less the arm's."""
    def mean(a):
        return np.mean([pq[f"s2/{a}/{c}/{b}/{ef}/1"] for b in quant.BUILDS], axis=0)

    diff = mean("i8-vec-c:f32") - mean(arm)
    boot = np.random.default_rng(3).integers(len(diff), size=(2000, len(diff)))
    lo, hi = np.percentile(diff[boot].mean(axis=1), [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def report(dirs: list[Path]) -> None:
    rows = [json.loads(line) for d in dirs for f in sorted(d.rglob("*.jsonl")) for line in f.read_text().splitlines()]
    print("## Planted: the query in sixteen bits, stage 1, recall by distance (want within 0.002 of f32)\n")
    s1 = {(r["config"], r["scheme"], r["calib"]): r["recall_dist"] for r in rows if r.get("stage") == 1}
    for config in sorted({r["config"] for r in rows}):
        gaps = [abs(s1[(config, "i8-vec-c:q16", c)] - s1[(config, "i8-vec-c:f32", c)])
                for c in ("all", *CENTERS) if (config, "i8-vec-c:q16", c) in s1]
        verdict = "not run" if not gaps else "holds" if max(gaps) <= 0.002 else "FAILS: the harness is wrong, and no result counts"
        print(f"* {config}: at most {max(gaps, default=0):.4f}: {verdict}")
    print()
    quant.RERANK = [1]  # here, not at import: quant.selftest uses its own
    quant.report(dirs)
    per_config = {f.stem: dict(np.load(f)) for d in dirs for f in sorted(d.rglob("*.npz"))}
    graphed = {c: pq for c, pq in per_config.items() if any(k.startswith("s2/") for k in pq)}
    if not graphed:
        return
    print("\n## What rounding the query costs, on the same builds: f32 query's recall less the arm's [95%]\n")
    print("| config | center | query | at ef 64 | at ef 128 |\n|---|---|---|---|---|")
    for config, pq in sorted(graphed.items()):
        for c in CENTERS:
            for q in ("q16", "q8", "q7"):
                cells = " | ".join(f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]" for m, lo, hi in (added(pq, f"i8-vec-c:{q}", c, ef) for ef in (64, 128)))
                print(f"| {config} | {c} | {q} | {cells} |")


quant.decide = decide


def selftest() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2000, 64)).astype(np.float32)
    x[:, 0] -= 20  # an offset every vector shares: only centering keeps it out of each query's range
    center = x.mean(axis=0)
    stored = quant2.rounded("i8-vec-c", x, x)[0]
    # A stored vector is its own eight-bit query, so a graph built on rounded vectors is the one an
    # integer kernel would build.
    assert np.abs(query(stored, center, 256) - stored).max() < 1e-5
    assert query(x, center, None) is x
    for levels in (128, 256, 65536):
        step = np.ptp(x - center, axis=1, keepdims=True) / (levels - 1)
        assert np.all(np.abs(query(x, center, levels) - x) <= step / 2 + 1e-5), levels
    # Recall is against the query as given: a coarse query loses neighbors, sixteen bits none.
    train = quant.prepare.unit(rng.standard_normal((3000, 32)).astype(np.float32))
    test = quant.prepare.unit(rng.standard_normal((50, 32)).astype(np.float32))
    gt_ids, gt_tenth = quant.truth(train, test)
    exact, c = quant.Rounded(train), train.mean(axis=0)
    assert stage1(exact, test, test, gt_ids, gt_tenth, train)[0].mean() == 1.0
    assert stage1(exact, query(test, c, 65536), test, gt_ids, gt_tenth, train)[0].mean() == 1.0
    assert stage1(exact, query(test, c, 4), test, gt_ids, gt_tenth, train)[0].mean() < 0.9
    quant2.selftest()


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "run":
        rest = [a for a in args[4:] if a != "--no-graph"]
        run(args[0], Path(args[1]), Path(args[2]), int(args[3]), int(rest[0]) if rest else None, "--no-graph" not in args)
    elif cmd == "report":
        report([Path(a) for a in args])
    elif cmd == "selftest":
        selftest()
