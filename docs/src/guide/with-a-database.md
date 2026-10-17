# With a database

Chassis keeps vectors and integer ids; the rows they belong to live in your database. This page
is how to keep the two in step, through a crash at any point. The examples use SQLite from
Python; the same order works with Postgres, and from Rust or C.

## Ids

Give each vector an id of its own, kept in its row, and a new one whenever the row's text
changes. Mending (below) compares the ids the index holds with the ids the rows hold, so an id
must never stand for two texts: an edit that kept its id would look the same before and after,
and a crash between the two would leave the old text with the new vector, with nothing to show
it.

```python
import sqlite3
from contextlib import contextmanager

from chassisdb import IndexOptions, VectorIndex

db = sqlite3.connect("notes.db", isolation_level=None)  # Transactions only where we BEGIN them
db.execute(
    "CREATE TABLE IF NOT EXISTS notes ("
    "id INTEGER PRIMARY KEY, text TEXT NOT NULL, vector INTEGER UNIQUE NOT NULL)"
)
# The number of the last write, which the index keeps too; see Opening.
db.execute("CREATE TABLE IF NOT EXISTS sync (version INTEGER NOT NULL)")
db.execute("INSERT INTO sync SELECT 0 WHERE NOT EXISTS (SELECT 1 FROM sync)")
index = VectorIndex("notes.chassis", dimensions=384, options=IndexOptions(metric="cosine"))


def new_vector_ids(n):
    """Ids no row holds: past the largest, so a new text never takes an id an old one had."""
    (top,) = db.execute("SELECT coalesce(max(vector), 0) FROM notes").fetchone()
    return list(range(top + 1, top + 1 + n))
```

`embed(texts)` below stands for your embedding model.

## Writing

Each write changes the rows in a transaction, changes the index, `flush()`es it, and only then
commits. It also counts itself in `sync` and gives the index the same number as its
`user_version`, which the flush commits with the vectors. If any of it fails, the commit too, it
rolls back the transaction and then the index: what the write did to the index would otherwise
wait there for the next write's flush. `writing()` does all of that around a write's own changes:

```python
@contextmanager
def writing():
    """The rows and the index change together, or neither does."""
    db.execute("BEGIN IMMEDIATE")
    try:
        check()  # A write whose commit failed left the index one ahead: mend that first
        yield
        db.execute("UPDATE sync SET version = version + 1")
        index.user_version = db.execute("SELECT version FROM sync").fetchone()[0]
        index.flush()
        db.execute("COMMIT")
    except BaseException:
        if db.in_transaction:  # A failed COMMIT, as a busy reader can cause, leaves it open
            db.execute("ROLLBACK")
        index.rollback()
        raise


def add_notes(texts):
    vectors = embed(texts)  # Before the transaction: a model can take a while
    with writing():
        ids = new_vector_ids(len(texts))
        notes = [
            db.execute("INSERT INTO notes (text, vector) VALUES (?, ?)", (text, id)).lastrowid
            for text, id in zip(texts, ids)
        ]
        index.add_batch(vectors, ids=ids)
    return notes


def edit_note(note, text):
    vector = embed([text])[0]
    with writing():
        (old,) = db.execute("SELECT vector FROM notes WHERE id = ?", (note,)).fetchone()
        (new,) = new_vector_ids(1)
        db.execute("UPDATE notes SET text = ?, vector = ? WHERE id = ?", (text, new, note))
        index.delete(old)
        index.add(vector, id=new)


def delete_note(note):
    with writing():
        (vector,) = db.execute("SELECT vector FROM notes WHERE id = ?", (note,)).fetchone()
        db.execute("DELETE FROM notes WHERE id = ?", (note,))
        index.delete(vector)
```

What a crash leaves:

- **Before the flush**: nothing. The transaction rolls back, and the index discards what wasn't
  flushed, as it does whenever it is closed without one.
- **Between the flush and the commit**: an index one write ahead of the rows. A new note's vector
  has no row, a deleted note's row has no vector, and an edited note has its new vector while its
  row keeps the old text and the old vector's id. The next start sees the numbers differ and
  mends it, as Opening shows. A commit that fails without a crash leaves the same, and the next
  write mends it before it counts itself, which is why `writing()` calls `check()` first.

## Reading

Search the index, then join the rows back by their vectors' ids:

```python
def search(query, k=5):
    hits = index.search(embed([query])[0], k=k)
    rows = dict(db.execute(
        f"SELECT vector, text FROM notes WHERE vector IN ({','.join('?' * len(hits))})",
        [hit.id for hit in hits],
    ).fetchall()) if hits else {}
    return [(hit.distance, rows[hit.id]) for hit in hits if hit.id in rows]
```

A vector a crash left without its row finds no row in the join and is dropped, and the next start
mends it, so a search needs no filter for it. Don't pass every row's vector id as `allowed` to
keep such vectors out: building the filter costs a pass over all of them on every search.

For a filter of your own, pass the vector ids it selects as `allowed`, in id order:
`SELECT vector FROM notes WHERE owner = ? ORDER BY vector`. Ids in ascending order are searched
where they lie; in any other order they are copied into a set first
([ADR-0022](https://github.com/tanvincible/chassis/blob/main/docs/adr/022-allowed-ids.md)).

## Opening

When the app starts, and as each write begins, compare the two numbers. They differ only when a
write flushed the index and its commit never happened, after a crash or a failed commit, the one
case that leaves anything to mend:

```python
def check():
    (version,) = db.execute("SELECT version FROM sync").fetchone()
    if index.user_version != version:
        return mend()
    return 0, 0
```

## Mending

What a crash left is found by comparing the two sets of ids, which takes one pass over each, so
run it only when `check` finds the numbers differ. Since each text has its own id, the comparison
finds every difference, an edit's too: its new vector is one without a row, and its old one is
missing.

```python
def mend():
    rows = dict(db.execute("SELECT vector, text FROM notes"))
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
