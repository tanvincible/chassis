"""The code agents copy: the skill's Python blocks and the examples run as written."""

import hashlib
import re
import runpy
import sqlite3
from pathlib import Path

import numpy as np
import pytest

from chassisdb import InvalidArgumentError, VectorIndex

ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "plugins" / "chassis" / "skills" / "chassis" / "SKILL.md"
WITH_A_DATABASE = ROOT / "docs" / "src" / "guide" / "with-a-database.md"
RECIPES = ROOT / "docs" / "src" / "guide" / "recipes.md"
EXAMPLES = sorted((ROOT / "chassisdb" / "examples").glob("[!_]*.py"))


def test_skill_python_blocks_run(tmp_path, monkeypatch):
    blocks = re.findall(r"```python\n(.*?)```", SKILL.read_text(), re.S)
    assert blocks
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    # Stands in for the app's embedding model, as the skill says.
    namespace = {"embed": lambda texts: rng.random((len(texts), 384), dtype=np.float32)}
    for block in blocks:
        exec(compile(block, str(SKILL), "exec"), namespace)


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda p: p.name)
def test_example_runs(example, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(example), run_name="__main__")


def test_cursor_rule_says_what_the_skill_says():
    def body(path):
        return path.read_text().split("---\n", 2)[2]

    rule = ROOT / "plugins" / "chassis" / "cursor" / "chassis.mdc"
    assert body(rule) == body(SKILL), f"copy {SKILL.name}'s text below the front matter into {rule}"


def _embed(texts):
    """A stand-in model that gives each text a vector of its own, every time."""
    seeds = [int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "little") for t in texts]
    return np.array([np.random.default_rng(s).random(384, dtype=np.float32) for s in seeds])


def _guide(page=None):
    """Runs the database guide's code; with `page`, only its first block, as a restart would."""
    blocks = re.findall(r"```python\n(.*?)```", WITH_A_DATABASE.read_text(), re.S)
    page = page if page is not None else {"embed": _embed}
    for block in blocks[:1] if "check" in page else blocks:
        exec(compile(block, str(WITH_A_DATABASE), "exec"), page)
    return page


def _in_step(page):
    """Every row's vector is in the index, under the row's vector id, made from the row's text."""
    index, db = page["index"], page["db"]
    rows = dict(db.execute("SELECT vector, text FROM notes"))
    assert sorted(index.ids()) == sorted(rows)
    for id, text in rows.items():
        held, made = index.get(id), _embed([text])[0]
        assert np.allclose(held / np.linalg.norm(held), made / np.linalg.norm(made), atol=1e-5)


class _BusyCommit:
    """A connection whose first COMMIT fails and leaves the transaction open, as SQLite's does
    while a reader holds the database."""

    def __init__(self, db):
        self.db, self.busy = db, True

    def execute(self, sql, *args):
        if sql == "COMMIT" and self.busy:
            self.busy = False
            raise sqlite3.OperationalError("database is locked")
        return self.db.execute(sql, *args)

    def __getattr__(self, name):
        return getattr(self.db, name)


def test_the_database_guide_survives_what_it_says_a_crash_leaves(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    page = _guide()
    apples, pears = page["add_notes"](["apples", "pears"])
    assert page["check"]() == (0, 0)  # In step: nothing to mend
    _in_step(page)

    flush = VectorIndex.flush

    def flushed_then_lost(index):  # The flush happens; the commit after it never does
        flush(index)
        raise OSError("the commit never happens")

    # An edit, then a crash between its flush and its commit: the index holds the new text's
    # vector, under a new id, and the row keeps the old text and id.
    with monkeypatch.context() as m:
        m.setattr(VectorIndex, "flush", flushed_then_lost)
        with pytest.raises(OSError):
            page["edit_note"](apples, "apricots")
    page["index"].close()
    page["db"].close()
    _guide(page)  # The next start
    assert page["check"]() == (1, 1)  # The new vector goes, the old one comes back
    _in_step(page)

    # A delete whose commit fails without a crash, then the app's next write.
    with monkeypatch.context() as m:
        m.setattr(VectorIndex, "flush", flushed_then_lost)
        with pytest.raises(OSError):
            page["delete_note"](pears)
    page["add_notes"](["plums"])  # Mends first, then writes
    _in_step(page)
    assert {text for _, text in page["search"]("fruit", k=10)} == {"apples", "pears", "plums"}

    # An edit whose add the index refuses after its delete ran, then the app's next write: the
    # rows roll back, and the index has to as well, or that write's flush commits the delete.
    page["embed"] = lambda texts: np.zeros((len(texts), 384), np.float32)  # Cosine refuses these
    with pytest.raises(InvalidArgumentError):
        page["edit_note"](apples, "nothing")
    page["embed"] = _embed
    page["add_notes"](["figs"])
    _in_step(page)

    # A delete whose COMMIT fails and leaves its transaction open, as a busy reader can make it.
    db = page["db"] = _BusyCommit(page["db"])
    with pytest.raises(sqlite3.OperationalError):
        page["delete_note"](pears)
    page["db"] = db.db
    page["add_notes"](["kiwis"])
    _in_step(page)

    # Edits and deletes that commit.
    page["edit_note"](apples, "apricots")
    page["delete_note"](pears)
    _in_step(page)
    assert {text for _, text in page["search"]("fruit", k=10)} == {
        "apricots",
        "plums",
        "figs",
        "kiwis",
    }


def test_recipes_run_one_after_another(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    page: dict = {}
    for block in re.findall(r"```python\n(.*?)```", RECIPES.read_text(), re.S):
        # The fastembed block replaces the stand-in model rather than following it.
        if "fastembed" in block:
            continue
        exec(compile(block, str(RECIPES), "exec"), page)
        if "folder" in block:
            nearest = page["search"]("when is the report due", k=1)
            assert nearest[0][0].startswith("The quarterly report")
