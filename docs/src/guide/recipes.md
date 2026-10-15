# Recipes

Short programs for what people most often build with Chassis: search over a folder of notes,
retrieval for a language model, an agent's memory, "more like this", and leaving out duplicates.
They are in Python, keep text in SQLite and vectors in Chassis under the same ids, and continue from
the setup below, so you can run them one after another.

## Setup

`embed` stands for your embedding model. This one counts words, so the recipes run without a model
to download; replace it with a real one to search by meaning.

```python
import hashlib
import re
import sqlite3

import numpy as np
from chassisdb import IndexOptions, VectorIndex

DIMENSIONS = 256  # your model's output size


def embed(texts):
    """Stands for your embedding model: each word of four letters or more counts in a bucket."""
    vectors = np.zeros((len(texts), DIMENSIONS), dtype=np.float32)
    for row, text in enumerate(texts):
        for word in re.findall(r"[a-z]{4,}", text.lower()):
            digest = hashlib.blake2b(word.encode(), digest_size=4).digest()
            vectors[row, int.from_bytes(digest, "little") % DIMENSIONS] += 1.0
    return vectors


db = sqlite3.connect("recipes.db")
db.execute("CREATE TABLE IF NOT EXISTS docs (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT)")
index = VectorIndex("recipes.chassis", dimensions=DIMENSIONS, options=IndexOptions(metric="cosine"))


def add(texts):
    """Rows first, then their vectors under the same ids; flush before the commit."""
    ids = [db.execute("INSERT INTO docs (text) VALUES (?)", (t,)).lastrowid for t in texts]
    index.add_batch(embed(texts), ids=ids)
    index.flush()
    db.commit()
    return ids


def search(query, k=3):
    """The nearest texts to the query, nearest first, with their distances."""
    hits = index.search(embed([query])[0], k=k)
    return [(db.execute("SELECT text FROM docs WHERE id = ?", (h.id,)).fetchone()[0], h.distance)
            for h in hits]
```

To search by meaning with [fastembed](https://github.com/qdrant/fastembed), for example, define
`embed` this way in the setup instead, and set `DIMENSIONS` to 384:

```python
from fastembed import TextEmbedding

model = TextEmbedding("BAAI/bge-small-en-v1.5")


def embed(texts):
    return np.array(list(model.embed(texts)), dtype=np.float32)
```

[With a database](./with-a-database.md) explains the write order in `add`, and what a crash
between the flush and the commit leaves.

## Notes

Search a folder of notes: one row per file.

```python
from pathlib import Path

folder = Path("notes")
folder.mkdir(exist_ok=True)
(folder / "groceries.md").write_text("Buy milk, eggs and bread on the way home")
(folder / "report.md").write_text("The quarterly report is due on Friday")
(folder / "dentist.txt").write_text("Call the dentist to move Tuesday's appointment")

files = sorted(p for p in folder.iterdir() if p.suffix in {".md", ".txt"})
add([p.read_text() for p in files])
print(search("when is the report due", k=1))
```

## Retrieval

For a language model's prompt: split a document into chunks, keep each chunk as a row, and put the
chunks nearest the question in the prompt.

```python
document = """Refunds take five working days from the day the return arrives.

We ship to the European Union and the United Kingdom, in three to seven days.

Returns are free within thirty days, in the original packaging."""

chunks = [part.strip() for part in document.split("\n\n")]
add(chunks)

question = "How many days do refunds take?"
context = "\n\n".join(text for text, _ in search(question, k=2))
prompt = f"Answer from this context:\n\n{context}\n\nQuestion: {question}"
print(prompt)  # send it to your language model
```

## Memory

An agent's memory is the same three operations. For coding agents there is
[chassisdb-mcp](https://github.com/tanvincible/chassis/tree/main/chassisdb-mcp), an MCP server built
this way; in your own agent:

```python
def remember(text):
    return add([text])[0]


def recall(query, k=3):
    return search(query, k=k)


def forget(memory_id):
    db.execute("DELETE FROM docs WHERE id = ?", (memory_id,))
    index.delete(memory_id)
    index.flush()
    db.commit()


memory = remember("The staging database is called orders_staging")
print(recall("which database is staging", k=1))
forget(memory)
```

## Similar

"More like this": search with a stored vector, and leave that vector out.

```python
(first,) = db.execute("SELECT id FROM docs ORDER BY id LIMIT 1").fetchone()
for hit in index.search(index.get(first), k=3, excluded=[first]):
    print(hit.id, hit.distance)
```

## Duplicates

Before adding a text, look for one already stored within a small distance, and keep the first. With
the cosine metric, 0 is the same direction; how close counts as a duplicate depends on your model,
so measure it on texts you know.

```python
def add_unless_stored(text, within=0.05):
    hits = index.search(embed([text])[0], k=1)
    if hits and hits[0].distance <= within:
        return hits[0].id
    return add([text])[0]


first = add_unless_stored("Pick up bread and coffee beans")
assert add_unless_stored("pick up bread and coffee beans") == first

index.close()
db.close()
```
