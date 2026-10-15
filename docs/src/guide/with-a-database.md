# With a database

Chassis keeps vectors and integer ids; the rows they belong to live in your database. This page
is how to keep the two in step, through a crash at any point. The examples use SQLite from
Python; the same order works with Postgres, and from Rust or C.

## One id

Use the row's integer key as the vector's id. In SQLite that is `INTEGER PRIMARY KEY
AUTOINCREMENT`: `AUTOINCREMENT` never gives a deleted row's id out again, which plain `INTEGER
PRIMARY KEY` does.

```python
import sqlite3

from chassisdb import IndexOptions, VectorIndex

db = sqlite3.connect("notes.db", isolation_level=None)  # Transactions only where we BEGIN them
db.execute("CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY AUTOINCREMENT, text TEXT)")
# The number of the last write, which the index keeps too; see Opening.
db.execute("CREATE TABLE IF NOT EXISTS sync (version INTEGER NOT NULL)")
db.execute("INSERT INTO sync SELECT 0 WHERE NOT EXISTS (SELECT 1 FROM sync)")
index = VectorIndex("notes.chassis", dimensions=384, options=IndexOptions(metric="cosine"))
```

`embed(texts)` below stands for your embedding model.

## Writing

Write the rows in a transaction, change the index, `flush()` it, and only then commit. Each write
also counts itself in `sync`, and gives the index the same number as its `user_version`, which the
flush commits with the vectors:

```python
def count_write():
    """Counts this write in the open transaction; the index's next flush commits the same number."""
    db.execute("UPDATE sync SET version = version + 1")
    index.user_version = db.execute("SELECT version FROM sync").fetchone()[0]


def add_notes(texts):
    vectors = embed(texts)  # Before the transaction: a model can take a while
    db.execute("BEGIN IMMEDIATE")
    try:
        ids = [db.execute("INSERT INTO notes (text) VALUES (?)", (t,)).lastrowid for t in texts]
        for id in ids:
            if id in index:  # Left by a commit that failed after its flush
                index.delete(id)
        index.add_batch(vectors, ids=ids)
        count_write()
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
        count_write()
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
  gone, and an index one write ahead of the database. The next start sees that and mends it, as
  Opening shows. A commit that fails without a crash leaves the same; its rolled-back insert gives
  its id back for the next insert, which is what `if id in index: index.delete(id)` is for, since
  adding an id the index holds is an `IdInUseError`.

To replace a vector, `delete(id)` then `add(vector, id=id)`. Until the next flush neither is
saved, and the flush saves both or neither.

## Reading

Search the index, then join the rows back by id:

```python
def search(query, k=5):
    hits = index.search(embed([query])[0], k=k)
    rows = dict(db.execute(
        f"SELECT id, text FROM notes WHERE id IN ({','.join('?' * len(hits))})",
        [hit.id for hit in hits],
    ).fetchall()) if hits else {}
    return [(hit.distance, rows[hit.id]) for hit in hits if hit.id in rows]
```

A vector a crash left without its row finds no row in the join and is dropped, and the next start
mends it, so a search needs no filter for it. Don't pass every row's id as `allowed`
to keep such vectors out: building the filter costs a pass over all of them on every search.

For a filter of your own, pass the ids it selects as `allowed`, in id order:
`SELECT id FROM notes WHERE owner = ? ORDER BY id`. Ids in ascending order are searched where they
lie; in any other order they are copied into a set first
([ADR-0022](https://github.com/tanvincible/chassis/blob/main/docs/adr/022-allowed-ids.md)).

## Opening

When the app starts, compare the two numbers. They differ only after a crash between a flush and
its commit, the one case that leaves anything to mend:

```python
def check():
    (version,) = db.execute("SELECT version FROM sync").fetchone()
    if index.user_version != version:
        return mend()
    return 0, 0
```

## Mending

What a crash left is found by comparing the two sets of ids, which takes one pass over each, so
run it only when `check` finds the numbers differ:

```python
def mend():
    rows = {id: text for id, text in db.execute("SELECT id, text FROM notes")}
    held = set(index.ids())
    for id in held - rows.keys():  # Vectors without a row
        index.delete(id)
    missing = sorted(rows.keys() - held)  # Rows without a vector
    if missing:
        index.add_batch(embed([rows[id] for id in missing]), ids=missing)
    index.user_version = db.execute("SELECT version FROM sync").fetchone()[0]
    index.flush()
    return len(held - rows.keys()), len(missing)
```

## Other processes

Processes that only search open the index with `read_only=True` (Python), `IndexReader` (Rust)
or `chassis_open_reader` (C). They see each flush, so with the order above a reader can see a
vector a moment before its row is committed: join on the rows, as `search` does, and it never
shows.
