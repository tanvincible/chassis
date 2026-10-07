"""Downloads the ANN benchmark datasets and writes them as raw files for bench/ann.

Each output file is a little-endian u32 row count and u32 column count, then the rows: f32
vectors (`<name>.train.f32`, `<name>.test.f32`) or the u32 ids of each query's true nearest
neighbors, nearest first (`<name>.gt.u32`).

    pip install numpy h5py pyarrow
    python bench/ann/prepare.py [data_dir [dataset ...]]

Datasets: sift-128, glove-100, dbpedia-openai3-1536 (default: all). One already written is skipped.
"""

import shutil
import sys
import urllib.request
from pathlib import Path

import numpy as np

QUERIES = 1000
GT_DEPTH = 100
DBPEDIA = (
    "https://huggingface.co/datasets/Qdrant/"
    "dbpedia-entities-openai3-text-embedding-3-large-1536-100K/resolve/main/data"
)


def fetch(url: str, path: Path) -> Path:
    if not path.exists():
        print(f"downloading {url}", flush=True)
        # ann-benchmarks.com rejects urllib's default user agent.
        request = urllib.request.Request(url, headers={"User-Agent": "chassis-bench"})
        partial = path.with_suffix(".partial")
        with urllib.request.urlopen(request) as response, open(partial, "wb") as out:
            shutil.copyfileobj(response, out)
        partial.rename(path)
    return path


def unit(x: np.ndarray) -> np.ndarray:
    # Chassis only does Euclidean distance; on unit vectors it ranks like cosine.
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norms == 0, 1, norms)


def ann_benchmarks(name: str, data_dir: Path, angular: bool):
    import h5py

    with h5py.File(fetch(f"http://ann-benchmarks.com/{name}.hdf5", data_dir / f"{name}.hdf5")) as f:
        train, test, gt = f["train"][:], f["test"][:QUERIES], f["neighbors"][:QUERIES]
    if angular:
        train, test = unit(train), unit(test)
    return train, test, gt


def dbpedia(data_dir: Path):
    import pyarrow.parquet as pq

    column = "text-embedding-3-large-1536-embedding"
    shards = []
    for i in range(3):
        path = fetch(f"{DBPEDIA}/train-0000{i}-of-00003.parquet", data_dir / f"dbpedia-{i}.parquet")
        shards.append(np.array(pq.read_table(path, columns=[column])[column].to_pylist()))
    data = unit(np.concatenate(shards).astype(np.float32))

    # Hold out random rows as queries and find their true neighbors by brute force.
    order = np.random.default_rng(0).permutation(len(data))
    test, train = data[order[:QUERIES]], data[order[QUERIES:]]
    nearest = np.argpartition(-(test @ train.T), GT_DEPTH, axis=1)[:, :GT_DEPTH]
    scores = np.take_along_axis(test @ train.T, nearest, axis=1)
    gt = np.take_along_axis(nearest, np.argsort(-scores, axis=1), axis=1)
    return train, test, gt


def write(path: Path, rows: np.ndarray, dtype: str) -> None:
    with open(path, "wb") as f:
        np.array(rows.shape, dtype="<u4").tofile(f)
        np.ascontiguousarray(rows, dtype=dtype).tofile(f)


def main() -> None:
    data_dir = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "data")
    data_dir.mkdir(parents=True, exist_ok=True)
    datasets = {
        "sift-128": lambda: ann_benchmarks("sift-128-euclidean", data_dir, angular=False),
        "glove-100": lambda: ann_benchmarks("glove-100-angular", data_dir, angular=True),
        "dbpedia-openai3-1536": lambda: dbpedia(data_dir),
    }
    for name in sys.argv[2:] or datasets:
        if all((data_dir / f"{name}.{part}").exists() for part in ("train.f32", "test.f32", "gt.u32")):
            continue
        train, test, gt = datasets[name]()
        write(data_dir / f"{name}.train.f32", train, "<f4")
        write(data_dir / f"{name}.test.f32", test, "<f4")
        write(data_dir / f"{name}.gt.u32", gt, "<u4")
        print(f"{name}: {train.shape[0]} x {train.shape[1]}, {test.shape[0]} queries", flush=True)


if __name__ == "__main__":
    main()
