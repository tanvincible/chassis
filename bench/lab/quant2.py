"""The follow-up to quant.py: eight bits alone, with ranges that leave room, and centered (lab only).

    python bench/lab/quant2.py run <dataset> <data_dir> <out_dir> <n> [dims] [--no-graph]
    python bench/lab/quant2.py report <out_dir>...
    python bench/lab/quant2.py selftest

Data comes from `quant.py prepare`. Everything but the schemes, the arms and the decision is
quant.py's, unchanged. The plan, predictions and decision rule are in quant-plan.md, "The
follow-up", committed before any result.
"""

import sys
from pathlib import Path

import numpy as np
import quant

original = quant.rounded


def rounded(scheme: str, x: np.ndarray, calib: np.ndarray):
    # i8-dim+P: each dimension's range in the calibration set, widened by P% of it on each side.
    if scheme.startswith("i8-dim+"):
        lo, hi = calib.min(axis=0), calib.max(axis=0)
        pad = (hi - lo) * int(scheme[len("i8-dim+"):]) / 100
        return quant.affine(x, lo - pad, hi + pad, 256)
    # i8-vec-c: the calibration set's mean subtracted, then each vector's own range (LVQ-8).
    if scheme == "i8-vec-c":
        c = calib.mean(axis=0)
        y, clipped = original("i8-vec", x - c, calib)
        y += c
        return y, clipped
    return original(scheme, x, calib)


def decide(results: dict) -> str:
    alone = [(s, c) for s, c in quant.ARMS[2:] if quant.passes(results, s, c, [1])]
    both = sorted({s for s, c in alone if c == "first1k"} & {s for s, c in alone if c == "cluster1k"})
    if both:
        return f"int8 alone, calibrated on the first vectors added: {both} (every arm passing: {alone})"
    if alone:
        return f"int8 alone only with a representative calibration: {alone}"
    return "int8 alone is not worth building with these calibrations: no arm within 0.01 of half precision at r=1"


quant.rounded = rounded
quant.decide = decide
quant.ARMS = [
    ("half", None),
    ("i8-dim", "first1k"),  # run 1's arm, to compare with run 1; not decided on
    ("i8-dim+10", "first1k"),
    ("i8-dim+10", "cluster1k"),
    ("i8-dim+25", "first1k"),
    ("i8-dim+25", "cluster1k"),
    ("i8-vec-c", "first1k"),
    ("i8-vec-c", "cluster1k"),
]
quant.CALIBRATED = ["i8-dim", "i8-dim+10", "i8-dim+25", "i8-vec-c"]
quant.UNCALIBRATED = []


def selftest() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2000, 64)).astype(np.float32)
    span = x.max(axis=0) - x.min(axis=0)
    for p in (10, 25):
        y, clipped = rounded(f"i8-dim+{p}", x, x)
        assert clipped == 0, p
        assert np.all(np.abs(y - x) <= span * (1 + 2 * p / 100) / 255 / 2 + 1e-5), p
        assert len(np.unique(y[:, 0])) <= 256, p
        narrow = rounded(f"i8-dim+{p}", x, x[:10])[1]
        assert 0 < narrow < rounded("i8-dim", x, x[:10])[1], "a margin clips less"
    # An offset every vector shares in one dimension, as ada-002 has: centering takes it out of
    # each vector's range, so its steps are as fine as without it.
    shifted = x.copy()
    shifted[:, 0] -= 20
    y, clipped = rounded("i8-vec-c", shifted, shifted)
    assert clipped == 0
    err_c = np.abs(y - shifted).max()
    err = np.abs(rounded("i8-vec", shifted, shifted)[0] - shifted).max()
    assert err_c <= np.ptp(x - x.mean(axis=0), axis=1).max() / 255 / 2 + 1e-4 and err_c * 3 < err, (err_c, err)
    quant.selftest()


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "run":
        rest = [a for a in args[4:] if a != "--no-graph"]
        quant.run(args[0], Path(args[1]), Path(args[2]), int(args[3]), int(rest[0]) if rest else None, "--no-graph" not in args)
    elif cmd == "report":
        quant.report([Path(a) for a in args])
    elif cmd == "selftest":
        selftest()
