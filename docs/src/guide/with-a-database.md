# Chassis Next to Your Database

Chassis keeps vectors and integer ids; the rows they belong to live in your database. This page
is how to keep the two in step, through a crash at any point. The examples use SQLite from
Python; the same order works with Postgres, and from Rust or C.

## One id for both

Use the row's integer key as the vector's id. In SQLite that is `INTEGER PRIMARY KEY
AUTOINCREMENT`: `AUTOINCREMENT` never gives a deleted row's id out again, which plain `INTEGER
PRIMARY KEY` does.

```python
import sqlite3

from chassisdb import IndexOptions, VectorIndex

db = sqlite3.connect("notes.db", isolation_level=None)  # Transactions only where we BEGIN them
db.execute("CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT)")
index = VectorIndex("notes.chassis", dimensions=384, options=IndexOptions(metric="cosine"))
```

`embed(texts)` below stands for your embedding model.

## Writing: the index flushes inside the transaction

Write the rows in a transaction, change the index, `flush()` it, and only then commit:

```python
def add_notes(texts):
    vectors = embed(texts)  # Before the transaction: a model can take a while
    db.execute("BEGIN IMMEDIATE")
    try:
        ids = [db.execute("INSERT INTO notes (text) VALUES (?)", (t,)).lastrowid for t in texts]
        for id in ids:
            if id in index:  # Left by a crash after a flush whose commit never happened
                index.delete(id)
        index.add_batch(vectors, ids=ids)
        index.flush()
    except BaseException:
        db.execute("ROLLBACK")
        raise
    db.execute("COMMIT")
    return ids


def delete_note(id):
    db.execute("BEGIN IMMEDIATE")
    try:
        index.delete(id)
        index.flush()
        db.execute("DELETE FROM notes WHERE id = ?", (id,))
    except BaseException:
        db.execute("ROLLBACK")
        raise
    db.execute("COMMIT")
```

What a crash leaves:

- **Before the flush**: nothing. The transaction rolls back, and the index discards what wasn't
  flushed, as it does whenever it is closed without one.
- **Between the flush and the commit**: vectors whose rows don't exist, or a row whose vector is
  gone. The rolled-back insert also gives its id back, so the next insert gets it again: that is
  what `if id in index: index.delete(id)` is for, since adding an id the index holds is an
  `IdInUseError`.

To replace a vector, `delete(id)` then `add(vector, id=id)`. Until the next flush neither is
saved, and the flush saves both or neither.

## Reading: search among the rows that exist

Pass the database's ids as `allowed`, so a vector a crash left takes no result's place, and join
the rows back by id:

```python
def search(query, k=5):
    ids = [row[0] for row in db.execute("SELECT id FROM notes")]
    if not ids:
        return []
    hits = index.search(embed([query])[0], k=k, allowed=ids)
    rows = dict(db.execute(
        f"SELECT id, text FROM notes WHERE id IN ({','.join('?' * len(hits))})",
        [hit.id for hit in hits],
    ).fetchall()) if hits else {}
    return [(hit.distance, rows[hit.id]) for hit in hits if hit.id in rows]
```

A filter works the same way: `SELECT id FROM notes WHERE owner = ?` gives the `allowed` ids.

## Mending at start

What a crash left is found by comparing the two sets of ids, which takes one pass over each:

```python
def mend():
    rows = {id: text for id, text in db.execute("SELECT id, text FROM notes")}
    held = set(index.ids())
    for id in held - rows.keys():  # Vectors without a row
        index.delete(id)
    missing = sorted(rows.keys() - held)  # Rows without a vector
    if missing:
        index.add_batch(embed([rows[id] for id in missing]), ids=missing)
    index.flush()
    return len(held - rows.keys()), len(missing)
```

## Readers in other processes

Processes that only search open the index with `read_only=True` (Python), `IndexReader` (Rust)
or `chassis_open_reader` (C). They see each flush, so with the order above a reader can see a
vector a moment before its row is committed: join on the rows, as `search` does, and it never
shows.
