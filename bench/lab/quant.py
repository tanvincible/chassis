"""Vectors in fewer bits: what eight, four and one bit a component would do to recall (lab only).

    python bench/lab/quant.py prepare <dataset> <data_dir>
    python bench/lab/quant.py run <dataset> <data_dir> <out_dir> <n> [dims] [--no-graph]
    python bench/lab/quant.py report <out_dir>...
    python bench/lab/quant.py selftest

Each scheme rounds the vectors and searches the rounded ones, as LAB_ROUND16 did for half
precision: nothing here is a file format or a kernel. Stage 1 ranks every vector by each scheme's
distance (no graph), stage 2 builds Chassis on the rounded vectors and searches them with the
queries as given. Re-ranking takes the first r*10 results and orders them by their
half-precision distances. The plan, predictions and the decision rule are in quant-plan.md and
in decide(), committed before any result.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ann"))
import prepare  # noqa: E402

K = 10
QUERIES = 1000
RERANK = [1, 2, 4, 10, 20, 50]
GRAPH_RERANK = [1, 2, 4]
EFS = [16, 32, 64, 128, 256]
BUILDS = ["a", "b"]
# Stage 2: (scheme, calibration). full and half are Chassis's own precisions.
ARMS = [
    ("full", None),
    ("half", None),
    ("i8-unit", None),
    ("i8-vec", None),
    ("i8-global", "first1k"),
    ("i8-q99", "first1k"),
    ("i8-dim", "first1k"),
    ("i8-dim", "cluster1k"),
]
CALIBRATED = ["i8-global", "i8-q99", "i8-dim", "i4-dim", "b1-rabitq"]
UNCALIBRATED = ["i8-unit", "i8-vec", "b1-sign"]
TEXT = {"openai3", "ada", "minilm", "bge"}

OPENAI3 = (
    "https://huggingface.co/datasets/Qdrant/"
    "dbpedia-entities-openai3-text-embedding-3-large-1536-100K/resolve/main/data"
)
ADA = (
    "https://huggingface.co/datasets/KShivendu/dbpedia-entities-openai-1M/"
    "resolve/refs%2Fconvert%2Fparquet/default/train"
)
MODELS = {"minilm": "sentence-transformers/all-MiniLM-L6-v2", "bge": "BAAI/bge-small-en-v1.5"}


def read(path: Path) -> np.ndarray:
    shape = np.fromfile(path, dtype="<u4", count=2)
    return np.fromfile(path, dtype="<f4", offset=8).reshape(shape)


def split(data: np.ndarray, texts=None):
    """The same held-out queries for every model of a corpus: a seeded permutation."""
    order = np.random.default_rng(0).permutation(len(data))
    data = data[order]
    texts = None if texts is None else [texts[i] for i in order]
    return data[QUERIES:], data[:QUERIES], texts


def prepare_dataset(name: str, data_dir: Path) -> None:
    import pyarrow.parquet as pq

    data_dir.mkdir(parents=True, exist_ok=True)
    if name in ("openai3", "minilm", "bge"):
        column = "text-embedding-3-large-1536-embedding"
        tables = [
            pq.read_table(
                prepare.fetch(f"{OPENAI3}/train-0000{i}-of-00003.parquet", data_dir / f"o3-{i}.parquet"),
                columns=["title", "text", column],
            )
            for i in range(3)
        ]
        vectors = np.concatenate([np.array(t[column].to_pylist(), dtype=np.float32) for t in tables])
        texts = [f"{a}. {b}" for t in tables for a, b in zip(t["title"].to_pylist(), t["text"].to_pylist())]
        train, test, texts = split(vectors, texts)
        if name != "openai3":
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(MODELS[name], device="cpu")
            model.max_seq_length = 256
            start = time.time()
            every = model.encode(texts, batch_size=64, normalize_embeddings=True, convert_to_numpy=True)
            print(f"{name}: embedded {len(texts)} texts in {time.time() - start:.0f} s", flush=True)
            train, test = every[QUERIES:], every[:QUERIES]
    elif name == "ada":
        vectors = []
        for i in range(8):
            path = prepare.fetch(f"{ADA}/{i:04d}.parquet", data_dir / f"ada-{i}.parquet")
            vectors.append(np.array(pq.read_table(path, columns=["openai"])["openai"].to_pylist(), dtype=np.float32))
            path.unlink()
        train, test, _ = split(np.concatenate(vectors))
    elif name == "sift":
        import h5py

        path = prepare.fetch("http://ann-benchmarks.com/sift-128-euclidean.hdf5", data_dir / "sift.hdf5")
        with h5py.File(path) as f:
            train, test, _ = split(f["train"][:201000].astype(np.float32))
    else:
        raise SystemExit(f"no dataset {name}")
    if name != "sift":
        train, test = prepare.unit(train), prepare.unit(test)
    prepare.write(data_dir / f"{name}.train.f32", train, "<f4")
    prepare.write(data_dir / f"{name}.test.f32", test, "<f4")
    print(f"{name}: {train.shape[0]} x {train.shape[1]}, {test.shape[0]} queries", flush=True)


# Schemes. Each returns the rounded vectors and the share of components clipped to its range.


def affine(x, lo, hi, levels):
    lo, hi = np.asarray(lo, np.float32), np.asarray(hi, np.float32)
    step = np.where(hi > lo, (hi - lo) / (levels - 1), 1).astype(np.float32)
    codes = np.clip(np.rint((x - lo) / step), 0, levels - 1)
    clipped = float(np.mean((x < lo) | (x > hi)))
    return (lo + codes * step).astype(np.float32), clipped


def rounded(scheme: str, x: np.ndarray, calib: np.ndarray):
    if scheme == "i8-unit":
        return affine(x, -1, 1, 255)
    if scheme == "i8-global":
        top = np.abs(calib).max()
        return affine(x, -top, top, 255)
    if scheme == "i8-q99":
        lo, hi = np.quantile(calib, [0.005, 0.995])
        return affine(x, lo, hi, 256)
    if scheme == "i8-dim":
        return affine(x, calib.min(axis=0), calib.max(axis=0), 256)
    if scheme == "i4-dim":
        return affine(x, calib.min(axis=0), calib.max(axis=0), 16)
    if scheme == "i8-vec":
        return affine(x, x.min(axis=1, keepdims=True), x.max(axis=1, keepdims=True), 256)
    if scheme == "b1-sign":
        return np.where(x >= 0, 1, -1).astype(np.float32) * np.abs(x).mean(axis=1, keepdims=True), 0.0
    raise ValueError(scheme)


class RaBitQ:
    """One bit a component after a random rotation about the centroid, with RaBitQ's estimator
    (Gao and Long, SIGMOD 2024). Queries stay in full precision, its most favourable case."""

    def __init__(self, x: np.ndarray, calib: np.ndarray, seed: int = 2):
        d = x.shape[1]
        self.c = calib.mean(axis=0).astype(np.float32)
        q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((d, d)))
        self.rot = (q * np.sign(np.diag(r))).astype(np.float32)
        y = (x - self.c) @ self.rot
        self.norm = np.linalg.norm(y, axis=1).astype(np.float32)
        self.bits = np.where(y >= 0, 1, -1).astype(np.float32)
        self.dot = (np.abs(y).sum(axis=1) / np.where(self.norm > 0, self.norm, 1)).astype(np.float32)

    def scores(self, q: np.ndarray) -> np.ndarray:
        inner = ((q - self.c) @ self.rot) @ self.bits.T
        return self.norm**2 - 2 * self.norm * inner / self.dot


class Rounded:
    def __init__(self, x: np.ndarray):
        self.x, self.sq = x, np.einsum("ij,ij->i", x, x)

    def scores(self, q: np.ndarray) -> np.ndarray:
        return self.sq - 2 * (q @ self.x.T)


def calibrations(train: np.ndarray) -> dict:
    """all: every vector. first1k: the first 1,000 added, a random sample here. cluster1k: the
    1,000 nearest one random vector, as when the first notes added are all on one topic."""
    seed = train[np.random.default_rng(1).integers(len(train))]
    nearest = np.argpartition(((train - seed) ** 2).sum(axis=1), 1000)[:1000]
    return {"all": train, "first1k": train[:1000], "cluster1k": train[nearest]}


def exact(store: np.ndarray, test: np.ndarray, cand: np.ndarray, dtype=np.float32) -> np.ndarray:
    """Squared distances from each query to its candidates, 50 queries at a time."""
    out = np.empty(cand.shape, dtype)
    for i in range(0, len(cand), 50):
        x = store[cand[i : i + 50]].astype(dtype)
        out[i : i + 50] = ((x - test[i : i + 50, None, :].astype(dtype)) ** 2).sum(axis=2)
    return out


def truth(train: np.ndarray, test: np.ndarray):
    """Exact nearest 10, the nearest 50 by float32 re-ranked in float64, and the 10th distance
    for a recall that counts a tie at the 10th as found."""
    sq = np.einsum("ij,ij->i", train, train)
    top = np.concatenate(
        [np.argpartition(sq - 2 * (q @ train.T), 50, axis=1)[:, :50] for q in np.array_split(test, 50)]
    )
    d = exact(train, test, top, np.float64)
    order = np.argsort(d, axis=1)[:, :K]
    return np.take_along_axis(top, order, axis=1), np.take_along_axis(d, order, axis=1)[:, -1]


def recall(found, gt_ids, gt_tenth, train, test):
    """Per-query recall@10 by id, and by distance."""
    assert found.min() >= 0 and found.max() < len(train), "a padded or foreign id"
    by_id = np.array([len(set(f) & set(g)) / K for f, g in zip(found, gt_ids)])
    d = exact(train, test, found, np.float64)
    by_dist = (d <= gt_tenth[:, None] * (1 + 1e-6) + 1e-12).sum(axis=1) / K
    return by_id, by_dist


def rerank(cand: np.ndarray, half: np.ndarray, test: np.ndarray, r: int) -> np.ndarray:
    c = cand[:, : r * K]
    if r == 1:
        return c
    order = np.argsort(exact(half, test, c), axis=1)[:, :K]
    return np.take_along_axis(c, order, axis=1)


def stage1(scorer, half, test, gt_ids, gt_tenth, train):
    depth = max(RERANK) * K
    cand = []
    for q in np.array_split(test, 50):
        s = scorer.scores(q)
        top = np.argpartition(s, depth, axis=1)[:, :depth]
        cand.append(np.take_along_axis(top, np.argsort(np.take_along_axis(s, top, axis=1), axis=1), axis=1))
    cand = np.concatenate(cand)
    return {r: recall(rerank(cand, half, test, r), gt_ids, gt_tenth, train, test) for r in RERANK}


def run(name: str, data_dir: Path, out: Path, n: int, dims: int | None, graph: bool) -> None:
    out.mkdir(parents=True, exist_ok=True)
    train, test = read(data_dir / f"{name}.train.f32")[:n], read(data_dir / f"{name}.test.f32")
    if dims:
        train, test = prepare.unit(train[:, :dims]), prepare.unit(test[:, :dims])
    tag = f"{name}-{dims or train.shape[1]}-{n}"
    half = train.astype(np.float16)
    lines, per_query = [], {}

    def emit(**row):
        row = {"config": tag, **row}
        lines.append(row)
        print("\t".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row.values()), flush=True)

    start = time.time()
    gt_ids, gt_tenth = truth(train, test)
    emit(stage="truth", seconds=time.time() - start)
    calib = calibrations(train)
    unit = np.abs(train).max() <= 1
    for scheme in CALIBRATED + UNCALIBRATED + ["half"]:
        if scheme == "i8-unit" and not unit:
            continue
        for c in calib if scheme in CALIBRATED else [None]:
            start = time.time()
            clipped = 0.0
            if scheme == "b1-rabitq":
                scorer = RaBitQ(train, calib[c])
            elif scheme == "half":
                scorer = Rounded(half.astype(np.float32))
            else:
                x, clipped = rounded(scheme, train, calib[c] if c else train)
                scorer = Rounded(x)
            for r, (by_id, by_dist) in stage1(scorer, half, test, gt_ids, gt_tenth, train).items():
                per_query[f"s1/{scheme}/{c}/{r}"] = by_id
                emit(stage=1, scheme=scheme, calib=c, r=r, recall=float(by_id.mean()),
                     recall_dist=float(by_dist.mean()), clipped=clipped)
            del scorer
            print(f"# {scheme} {c}: {time.time() - start:.0f} s", flush=True)

    # Planted loss: every vector under another's id. Recall at r=1 has to be about 0.
    shuffled = Rounded(train[np.random.default_rng(4).permutation(len(train))])
    by_id, by_dist = stage1(shuffled, half, test, gt_ids, gt_tenth, train)[1]
    emit(stage=1, scheme="shuffled", calib=None, r=1, recall=float(by_id.mean()), recall_dist=float(by_dist.mean()), clipped=0.0)
    del shuffled

    if graph:
        from chassis import IndexOptions, VectorIndex

        d = train.shape[1]
        for scheme, c in ARMS:
            if scheme == "i8-unit" and not unit:
                continue
            precision = "half" if scheme == "half" else "full"
            data = train if scheme in ("full", "half") else rounded(scheme, train, calib[c] if c else train)[0]
            for b in BUILDS:
                path = out / f"{scheme}.chassis"
                path.unlink(missing_ok=True)
                start = time.time()
                with VectorIndex(path, d, IndexOptions(precision=precision)) as index:
                    index.add_batch(data)
                    index.flush()
                emit(stage="build", scheme=scheme, calib=c, build=b, seconds=time.time() - start)
                for ef in EFS:
                    with VectorIndex(path, d, IndexOptions(precision=precision, ef_search=ef)) as index:
                        for r in GRAPH_RERANK:
                            ids, _ = index.search_batch(test, k=r * K)
                            found = rerank(ids.astype(np.int64), half, test, r)
                            by_id, by_dist = recall(found, gt_ids, gt_tenth, train, test)
                            per_query[f"s2/{scheme}/{c}/{b}/{ef}/{r}"] = by_id
                            emit(stage=2, scheme=scheme, calib=c, build=b, ef=ef, r=r,
                                 recall=float(by_id.mean()), recall_dist=float(by_dist.mean()))
                path.unlink()
            del data
    (out / f"{tag}.jsonl").write_text("".join(json.dumps(row) + "\n" for row in lines))
    np.savez_compressed(out / f"{tag}.npz", **per_query)


# The pre-registered analysis. Loss is half precision's recall@10 minus the arm's, both averaged
# over the two builds, with a 95% bootstrap interval over queries, paired.

LIMIT = 0.01
CI_LIMIT = 0.02


def loss(pq: dict, scheme: str, c, ef: int, r: int):
    arm = np.mean([pq[f"s2/{scheme}/{c}/{b}/{ef}/{r}"] for b in BUILDS], axis=0)
    base = np.mean([pq[f"s2/half/None/{b}/{ef}/1"] for b in BUILDS], axis=0)
    diff = base - arm
    boot = np.random.default_rng(3).integers(len(diff), size=(2000, len(diff)))
    lo, hi = np.percentile(diff[boot].mean(axis=1), [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi)


def passes(results: dict, scheme: str, c, rs) -> bool:
    """Within LIMIT of half precision, and its interval within CI_LIMIT, at ef 64 and 128, in
    every text configuration, for some r in rs."""
    def ok(pq, ef):
        return any(m <= LIMIT and hi <= CI_LIMIT for m, _, hi in (loss(pq, scheme, c, ef, r) for r in rs))

    return all(ok(pq, ef) for pq in results.values() for ef in (64, 128))


def decide(results: dict) -> str:
    alone = [(s, c) for s, c in ARMS[2:] if passes(results, s, c, [1])]
    if any(c is None for _, c in alone):
        return f"int8 alone, with no calibration: {alone}"
    if ("i8-dim", "first1k") in alone and ("i8-dim", "cluster1k") in alone:
        return f"int8 alone, calibrated on the first vectors added: {alone}"
    if alone:
        return f"int8 alone only with a representative calibration: {alone}"
    reranked = [(s, c) for s, c in ARMS[2:] if passes(results, s, c, [2, 4])]
    if reranked:
        return f"int8 only with re-ranking from half-precision copies: smaller in memory, not on disk: {reranked}"
    return "not worth building: no int8 arm within 0.01 of half precision, even re-ranked"


def report(dirs: list[Path]) -> None:
    rows = [json.loads(line) for d in dirs for f in sorted(d.rglob("*.jsonl")) for line in f.read_text().splitlines()]
    per_config = {}
    for d in dirs:
        for f in sorted(d.rglob("*.npz")):
            per_config[f.stem] = dict(np.load(f))
    configs = sorted({r["config"] for r in rows})

    print("## Stage 1: every vector ranked by each scheme, recall@10 by re-ranking factor\n")
    for config in configs:
        print(f"### {config}\n\n| scheme | calibration | clipped | " + " | ".join(f"r={r}" for r in RERANK) + " |")
        print("|---|---|---|" + "---|" * len(RERANK))
        s1 = [r for r in rows if r["config"] == config and r.get("stage") == 1]
        for scheme, c in dict.fromkeys((r["scheme"], r["calib"]) for r in s1 if r["scheme"] != "shuffled"):
            got = {r["r"]: r for r in s1 if r["scheme"] == scheme and r["calib"] == c}
            clipped = got[1]["clipped"]
            cells = " | ".join(f"{got[r]['recall']:.4f}" for r in RERANK)
            print(f"| {scheme} | {c or '-'} | {clipped:.2%} | {cells} |")
        print()

    print("## Planted outcomes\n")
    checks = [("sift", "i8-dim", "all", lambda v: v >= 0.995, "at least 0.995")]
    checks += [(c, "shuffled", None, lambda v: v <= 0.01, "at most 0.01") for c in configs]
    for prefix, scheme, calib, ok, want in checks:
        got = [r["recall"] for r in rows
               if r["config"].startswith(prefix) and r.get("scheme") == scheme and r.get("calib") == calib and r.get("r") == 1]
        verdict = "not run" if not got else "holds" if ok(got[0]) else "FAILS: the harness is wrong, and no result counts"
        print(f"* {prefix} {scheme}{'/' + calib if calib else ''}, stage 1, r=1: {f'{got[0]:.4f}' if got else '-'} (want {want}): {verdict}")
    print()

    graphed = {c: pq for c, pq in per_config.items() if any(k.startswith("s2/") for k in pq)}
    if not graphed:
        return
    print("## Stage 2: Chassis on the rounded vectors, recall@10 (mean of builds a, b)\n")
    for config, pq in sorted(graphed.items()):
        print(f"### {config}\n\n| arm | r | " + " | ".join(f"ef {ef}" for ef in EFS) + " | loss at 64 [95%] | loss at 128 [95%] |")
        print("|---|---|" + "---|" * (len(EFS) + 2))
        for scheme, c in ARMS:
            for r in GRAPH_RERANK:
                if f"s2/{scheme}/{c}/a/64/{r}" not in pq:
                    continue
                cells = []
                for ef in EFS:
                    a, b = (pq[f"s2/{scheme}/{c}/{x}/{ef}/{r}"].mean() for x in BUILDS)
                    cells.append(f"{(a + b) / 2:.4f} ±{abs(a - b) / 2:.4f}")
                losses = [loss(pq, scheme, c, ef, r) for ef in (64, 128)]
                fmt = " | ".join(f"{m:+.4f} [{lo:+.4f}, {hi:+.4f}]" for m, lo, hi in losses)
                print(f"| {scheme}{'/' + c if c else ''} | {r} | " + " | ".join(cells) + f" | {fmt} |")
        print()
    text = {c: pq for c, pq in graphed.items() if c.split("-")[0] in TEXT}
    print(f"## Decision (pre-registered, over {len(text)} text configurations)\n\n{decide(text)}")


def selftest() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2000, 64)).astype(np.float32)
    for scheme in ["i8-global", "i8-dim", "i8-vec", "i4-dim"]:
        y, clipped = rounded(scheme, x, x)
        levels = {"i4-dim": 16, "i8-global": 255}.get(scheme, 256)
        span = (x.max(axis=0) - x.min(axis=0)) if scheme.endswith("dim") else np.ptp(x, axis=1, keepdims=True)
        if scheme == "i8-global":
            span = 2 * np.abs(x).max()
        assert clipped == 0, scheme
        assert np.all(np.abs(y - x) <= span / (levels - 1) / 2 + 1e-5), scheme
        assert len(np.unique(y[0] if scheme == "i8-vec" else y[:, 0])) <= levels, scheme
    ints = rng.integers(0, 256, (2000, 32)).astype(np.float32)
    ints[0], ints[1] = 0, 255
    assert np.array_equal(rounded("i8-dim", ints, ints)[0], ints), "integers 0..255 survive i8-dim"
    # Ten samples bound a component of an eleventh with probability 9/11: about 2/11 clipped.
    assert 0.12 < rounded("i8-dim", x, x[:10])[1] < 0.25, "a narrow calibration clips"
    # RaBitQ's estimate of a unit inner product is unbiased; check over many pairs.
    o = prepare.unit(rng.standard_normal((4000, 128)).astype(np.float32))
    q = prepare.unit(rng.standard_normal((1, 128)).astype(np.float32))
    zero = np.zeros((1, 128), np.float32)
    est = RaBitQ(o, zero)
    guess = -(est.scores(q) - est.norm**2) / (2 * est.norm)
    true = (o @ q.T).ravel()
    err = guess.ravel() - true
    assert abs(err.mean()) < 3 * err.std() / np.sqrt(len(err)) + 1e-3, err.mean()
    assert err.std() < 0.1, err.std()
    # The recall machinery: exact vectors find everything, noisy ones do not.
    train, test = prepare.unit(rng.standard_normal((3000, 32)).astype(np.float32)), prepare.unit(
        rng.standard_normal((50, 32)).astype(np.float32)
    )
    gt_ids, gt_tenth = truth(train, test)
    half = train.astype(np.float16).astype(np.float32)
    assert stage1(Rounded(train), half, test, gt_ids, gt_tenth, train)[1][0].mean() == 1.0
    noisy = stage1(Rounded(train + 0.05 * rng.standard_normal(train.shape).astype(np.float32)), half, test, gt_ids, gt_tenth, train)
    assert noisy[1][0].mean() < 0.8 and noisy[50][0].mean() > 0.99, "noise loses neighbors, re-ranking finds them"
    print("selftest ok")


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "prepare":
        prepare_dataset(args[0], Path(args[1]))
    elif cmd == "run":
        rest = [a for a in args[4:] if a != "--no-graph"]
        run(args[0], Path(args[1]), Path(args[2]), int(args[3]), int(rest[0]) if rest else None, "--no-graph" not in args)
    elif cmd == "report":
        report([Path(a) for a in args])
    elif cmd == "selftest":
        selftest()
