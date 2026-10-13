"""Semantic search over notes: text in SQLite, vectors in Chassis, joined by id.

Runs as is: `embed` below hashes words into a vector, so it needs no model. To search by meaning,
replace it with your embedding model, for example:

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")  # 384 dimensions
    def embed(texts):
        return model.encode(texts, normalize_embeddings=True)
"""

import hashlib
import re
import sqlite3
import tempfile
from contextlib import closing
from pathlib import Path

import numpy as np

from chassisdb import IndexOptions, VectorIndex

DIMENSIONS = 256  # Your model's output size


def embed(texts):
    """A stand-in for an embedding model: each word longer than 3 letters adds 1 to one of
    DIMENSIONS buckets."""
    vectors = np.zeros((len(texts), DIMENSIONS), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in re.findall(r"[a-z]{4,}", text.lower()):
            bucket = int.from_bytes(hashlib.blake2b(word.encode(), digest_size=4).digest(), "little")
            vectors[row, bucket % DIMENSIONS] += 1.0
    return vectors


NOTES = [
    "Buy milk, eggs and bread on the way home",
    "The quarterly report is due on Friday",
    "Call the dentist to move Tuesday's appointment",
    "Ideas for the team offsite: hiking or a cooking class",
    "Draft the report summary and send it to the team",
    "Pick up bread and coffee beans",
]


def main(folder: Path) -> None:
    with closing(sqlite3.connect(folder / "notes.db")) as db:
        search_notes(folder, db)


def search_notes(folder: Path, db: sqlite3.Connection) -> None:
    # AUTOINCREMENT: a deleted note's id is never given out again.
    db.execute(
        "CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT NOT NULL)"
    )

    # Most text embedding models are compared by cosine distance.
    options = IndexOptions(metric="cosine")
    with VectorIndex(folder / "notes.chassis", dimensions=DIMENSIONS, options=options) as index:
        # Each note's SQLite row id is its id in the index too.
        ids = [db.execute("INSERT INTO notes (text) VALUES (?)", (t,)).lastrowid for t in NOTES]
        db.commit()
        index.add_batch(embed(NOTES), ids=ids)
        # Leaving the `with` block flushes; call index.flush() yourself to save sooner.

    # Open it again, as a later run of the app would. The metric has to be the same.
    with VectorIndex(folder / "notes.chassis", dimensions=DIMENSIONS, options=options) as index:
        query = "when is the report due"
        print(f"Notes like {query!r}:")
        for hit in index.search(embed([query])[0], k=3):
            (text,) = db.execute("SELECT text FROM notes WHERE id = ?", (hit.id,)).fetchone()
            print(f"  {hit.distance:.3f}  {text}")

        # "More like this": start from a stored note, and leave that note out.
        first = ids[0]
        print(f"\nMore like {NOTES[0]!r}:")
        for hit in index.search(index.get(first), k=2, excluded=[first]):
            (text,) = db.execute("SELECT text FROM notes WHERE id = ?", (hit.id,)).fetchone()
            print(f"  {hit.distance:.3f}  {text}")

        # Deleting a note: from both stores.
        db.execute("DELETE FROM notes WHERE id = ?", (first,))
        db.commit()
        index.delete(first)
        assert first not in index
        print(f"\n{len(index)} notes left, {index.deleted_count()} deleted in the index file")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as folder:
        main(Path(folder))
