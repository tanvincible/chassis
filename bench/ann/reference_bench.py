"""Runs hnswlib or usearch on a dataset written by prepare.py with the same settings as Chassis's
examples/ann.rs: M 16, ef_construction 200, one thread, recall@10 against queries per second.

    pip install numpy hnswlib usearch
    python bench/ann/reference_bench.py {hnswlib,usearch} <data_dir> <dataset>
"""

import sys
import tempfile
import time
from pathlib import Path

import numpy as np

K = 10
EF_SEARCH = [10, 16, 32, 64, 128, 256, 512]


class Hnswlib:
    def __init__(self, train: np.ndarray):
        import hnswlib

        self.index = hnswlib.Index(space="l2", dim=train.shape[1])
        self.index.init_index(max_elements=len(train), M=16, ef_construction=200, random_seed=0)
        self.index.add_items(train, np.arange(len(train)), num_threads=1)

    def save(self, path: Path) -> None:
        self.index.save_index(str(path))

    def search(self, queries: np.ndarray, ef: int) -> np.ndarray:
        self.index.set_ef(ef)
        return self.index.knn_query(queries, k=K, num_threads=1)[0]


class Usearch:
    def __init__(self, train: np.ndarray):
        from usearch.index import Index

        self.index = Index(
            ndim=train.shape[1], metric="l2sq", dtype="f32", connectivity=16, expansion_add=200
        )
        self.index.add(np.arange(len(train)), train, threads=1)

    def save(self, path: Path) -> None:
        self.index.save(str(path))

    def search(self, queries: np.ndarray, ef: int) -> np.ndarray:
        self.index.expansion_search = ef
        return self.index.search(queries, K, threads=1).keys


def read(path: Path, dtype: str) -> np.ndarray:
    shape = np.fromfile(path, dtype="<u4", count=2)
    return np.fromfile(path, dtype=dtype, offset=8).reshape(shape)


def main() -> None:
    engine, data_dir, name = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    train = read(data_dir / f"{name}.train.f32", "<f4")
    test = read(data_dir / f"{name}.test.f32", "<f4")
    truth = read(data_dir / f"{name}.gt.u32", "<u4")[:, :K]

    start = time.perf_counter()
    index = {"hnswlib": Hnswlib, "usearch": Usearch}[engine](train)
    build = time.perf_counter() - start

    with tempfile.TemporaryDirectory(dir=data_dir) as tmp:
        path = Path(tmp) / "index"
        index.save(path)
        size_mb = path.stat().st_size / 1e6
    print(f"{engine}\t{name}\tbuild\t{build:.1f}s\t{size_mb:.1f} MB", flush=True)

    index.search(test, EF_SEARCH[0])
    for ef in EF_SEARCH:
        start = time.perf_counter()
        found = index.search(test, ef)
        qps = len(test) / (time.perf_counter() - start)
        recall = np.mean([len(set(f) & set(t)) for f, t in zip(found, truth)]) / K
        print(f"{engine}\t{name}\t{ef}\t{recall:.3f}\t{qps:.0f}", flush=True)


if __name__ == "__main__":
    main()
