"""Other engines on the lab's datasets, each through its Python package (experiment branch only).

    python bench/lab/incumbents.py <engine> <data_dir> <dataset> <n> <truth.u32> <workdir> [cold]

Every engine builds on every core, with M 16 and ef_construction 200 where it has such settings,
saves to disk, opens what it saved, and searches on one thread. With PAGETOOL set, its files are
then put out of memory and a fresh process times opening them and the first queries.

Lines are tab-separated: inc engine n what setting recall value.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

K = 10
EF = [32, 64, 128, 256]


def read(path: Path, dtype: str) -> np.ndarray:
    shape = np.fromfile(path, dtype="<u4", count=2)
    return np.fromfile(path, dtype=dtype, offset=8).reshape(shape)


def rows(found) -> np.ndarray:
    """Result ids as a matrix, short rows padded with -1."""
    found = list(found)
    out = np.full((len(found), K), -1, dtype=np.int64)
    for i, ids in enumerate(found):
        ids = list(ids)[:K]
        out[i, : len(ids)] = ids
    return out


class Hnswlib:
    file, settings, queries = "hnswlib.bin", EF, 1000

    def build(self, train, path):
        import hnswlib

        index = hnswlib.Index(space="l2", dim=train.shape[1])
        index.init_index(max_elements=len(train), M=16, ef_construction=200, random_seed=0)
        index.add_items(train, np.arange(len(train)), num_threads=-1)
        index.save_index(str(path))

    def load(self, path, dims, n):
        import hnswlib

        self.index = hnswlib.Index(space="l2", dim=dims)
        self.index.load_index(str(path), max_elements=n)

    def search(self, queries, ef):
        self.index.set_ef(ef)
        return self.index.knn_query(queries, k=K, num_threads=1)[0]


class Usearch:
    file, settings, queries = "usearch.index", EF, 1000
    view = False

    def new(self, dims):
        from usearch.index import Index

        return Index(ndim=dims, metric="l2sq", dtype="f32", connectivity=16, expansion_add=200)

    def build(self, train, path):
        index = self.new(train.shape[1])
        index.add(np.arange(len(train)), train, threads=os.cpu_count())
        index.save(str(path))

    def load(self, path, dims, n):
        self.index = self.new(dims)
        # `view` serves the file through a memory mapping; `load` reads it all in.
        (self.index.view if self.view else self.index.load)(str(path))

    def search(self, queries, ef):
        self.index.expansion_search = ef
        return self.index.search(queries, K, threads=1).keys


class UsearchView(Usearch):
    view = True


class Faiss:
    file, settings, queries = "faiss.index", EF, 1000

    def build(self, train, path):
        import faiss

        index = faiss.IndexHNSWFlat(train.shape[1], 16)
        index.hnsw.efConstruction = 200
        index.add(train)
        faiss.write_index(index, str(path))

    def load(self, path, dims, n):
        import faiss

        faiss.omp_set_num_threads(1)
        self.index = faiss.read_index(str(path))

    def search(self, queries, ef):
        self.index.hnsw.efSearch = ef
        return self.index.search(queries, K)[1]


class Annoy:
    file, settings, queries = "annoy.ann", [2000, 8000, 32000, 128000], 1000

    def build(self, train, path):
        from annoy import AnnoyIndex

        index = AnnoyIndex(train.shape[1], "euclidean")
        for i, vector in enumerate(train):
            index.add_item(i, vector)
        index.build(32, n_jobs=-1)
        index.save(str(path))

    def load(self, path, dims, n):
        from annoy import AnnoyIndex

        self.index = AnnoyIndex(dims, "euclidean")
        self.index.load(str(path))

    def search(self, queries, search_k):
        return rows(self.index.get_nns_by_vector(q, K, search_k=search_k) for q in queries)


class SqliteVec:
    """Exact search: sqlite-vec scans every vector."""

    file, settings, queries = "sqlitevec.db", ["exact"], 100

    def open(self, path):
        import sqlite3

        import sqlite_vec

        db = sqlite3.connect(str(path))
        db.enable_load_extension(True)
        sqlite_vec.load(db)
        db.enable_load_extension(False)
        return db

    def build(self, train, path):
        db = self.open(path)
        db.execute(f"create virtual table v using vec0(embedding float[{train.shape[1]}])")
        with db:
            db.executemany(
                "insert into v(rowid, embedding) values (?, ?)",
                ((i + 1, vector.tobytes()) for i, vector in enumerate(train)),
            )
        db.close()

    def load(self, path, dims, n):
        self.db = self.open(path)

    def search(self, queries, _):
        sql = f"select rowid - 1 from v where embedding match ? and k = {K} order by distance"
        return rows([r[0] for r in self.db.execute(sql, (q.tobytes(),))] for q in queries)


class Lance:
    """LanceDB with its IVF_HNSW_SQ index, which keeps 8-bit vectors. The setting is ef; "256r"
    also re-ranks ten times as many candidates by their full vectors."""

    file, settings, queries = "lance", [32, 64, 128, 256, "256r"], 200

    def build(self, train, path):
        import lancedb
        import pyarrow as pa

        flat = pa.array(train.reshape(-1), type=pa.float32())
        data = pa.table(
            {
                "id": pa.array(np.arange(len(train), dtype=np.int64)),
                "vector": pa.FixedSizeListArray.from_arrays(flat, train.shape[1]),
            }
        )
        from lancedb.index import IvfHnswSq

        table = lancedb.connect(str(path)).create_table("v", data=data, mode="overwrite")
        table.create_index("vector", config=IvfHnswSq(distance_type="l2", m=16, ef_construction=200))

    def load(self, path, dims, n):
        import lancedb

        self.table = lancedb.connect(str(path)).open_table("v")

    def search(self, queries, setting):
        def ask(q):
            query = self.table.search(q).limit(K)
            if setting == "256r":
                query = query.ef(256).refine_factor(10)
            else:
                query = query.ef(setting)
            return query.to_arrow()["id"].to_pylist()

        return rows(ask(q) for q in queries)


class ChassisPy:
    """Chassis through its Python package, one call a query, opened as a reader."""

    file, settings, queries = "chassis-full.chassis", EF, 1000
    precision, warm = "full", False

    def options(self, ef):
        from chassis import IndexOptions

        return IndexOptions(
            max_connections=16,
            ef_construction=200,
            ef_search=ef,
            precision=self.precision,
            warm=self.warm,
        )

    def build(self, train, path):
        from chassis import VectorIndex

        index = VectorIndex(path, dimensions=train.shape[1], options=self.options(64))
        index.add_batch(train)
        index.flush()
        index.close()

    def load(self, path, dims, n):
        self.path, self.dims, self.index, self.ef = path, dims, None, None
        self.reopen(self.settings[1])

    def reopen(self, ef):
        from chassis import VectorIndex

        if self.index is not None:
            self.index.close()
        self.index = VectorIndex(
            self.path, dimensions=self.dims, options=self.options(ef), read_only=True
        )
        self.ef = ef

    def search(self, queries, ef):
        if ef != self.ef:
            self.reopen(ef)
        return rows([r.id for r in self.index.search(q, k=K)] for q in queries)


class ChassisPyHalf(ChassisPy):
    file, precision = "chassis-half.chassis", "half"


class ChassisPyWarm(ChassisPy):
    """The same file as chassis-py, opened with `warm`: only its cold run means anything new."""

    warm = True


class ChassisPyHalfWarm(ChassisPyHalf):
    warm = True


ENGINES = {
    "hnswlib": Hnswlib,
    "usearch": Usearch,
    "usearch-view": UsearchView,
    "faiss": Faiss,
    "annoy": Annoy,
    "sqlitevec": SqliteVec,
    "lancedb": Lance,
    "chassis-py": ChassisPy,
    "chassis-py-half": ChassisPyHalf,
    "chassis-py-warm": ChassisPyWarm,
    "chassis-py-half-warm": ChassisPyHalfWarm,
}


def files(path: Path):
    return [p for p in path.rglob("*") if p.is_file()] if path.is_dir() else [path]


def main() -> None:
    name, data_dir, dataset, n = sys.argv[1], Path(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    truth_path, work = Path(sys.argv[5]), Path(sys.argv[6])
    phase = sys.argv[7] if len(sys.argv) > 7 else "all"
    engine = ENGINES[name]()
    work.mkdir(parents=True, exist_ok=True)
    path = work / engine.file
    test = read(data_dir / f"{dataset}.test.f32", "<f4")[: engine.queries]
    truth = read(truth_path, "<u4")[: engine.queries, :K]
    dims = test.shape[1]

    def line(what, setting, recall, value):
        print(f"inc\t{name}\t{n}\t{what}\t{setting}\t{recall:.4f}\t{value:.3f}", flush=True)

    ms = lambda since: (time.perf_counter() - since) * 1e3
    if phase == "cold":
        setting = engine.settings[min(1, len(engine.settings) - 1)]
        start = time.perf_counter()
        engine.load(path, dims, n)
        line("cold_open_ms", setting, 0, ms(start))
        engine.search(test[:1], setting)
        line("cold_first_ms", setting, 0, ms(start))
        engine.search(test[1:100], setting)
        line("cold_100_ms", setting, 0, ms(start))
        return

    if not path.exists():
        train = read(data_dir / f"{dataset}.train.f32", "<f4")[:n]
        start = time.perf_counter()
        engine.build(train, path)
        line("build_s", 0, 0, ms(start) / 1e3)
        del train
    line("size_mb", 0, 0, sum(f.stat().st_size for f in files(path)) / 1e6)

    start = time.perf_counter()
    engine.load(path, dims, n)
    line("open_ms", 0, 0, ms(start))
    for setting in engine.settings:
        engine.search(test[:10], setting)
        rates = []
        for _ in range(3):
            start = time.perf_counter()
            found = engine.search(test, setting)
            rates.append(len(test) / (time.perf_counter() - start))
        hits = sum(len(set(f) & set(t)) for f, t in zip(np.asarray(found).tolist(), truth.tolist()))
        line("search", setting, hits / (len(test) * K), sorted(rates)[1])

    pagetool = os.environ.get("PAGETOOL")
    if pagetool:
        # A file this process still maps stays in memory whatever it is told.
        del engine
        import gc

        gc.collect()
        for f in files(path):
            subprocess.run([pagetool, "evict", str(f)], check=False, stdout=subprocess.DEVNULL)
            resident = subprocess.run([pagetool, "resident", str(f)], capture_output=True, text=True)
            print(f"inc-resident\t{name}\t{resident.stdout.strip()}", flush=True)
        subprocess.run([sys.executable, __file__, *sys.argv[1:7], "cold"], check=False)


if __name__ == "__main__":
    main()
