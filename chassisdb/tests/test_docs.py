"""The code agents copy: the skill's Python blocks and the examples run as written."""

import re
import runpy
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SKILL = ROOT / "plugins" / "chassis" / "skills" / "chassis" / "SKILL.md"
WITH_A_DATABASE = ROOT / "docs" / "src" / "guide" / "with-a-database.md"
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


def test_the_database_guide_survives_what_it_says_a_crash_leaves(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = np.random.default_rng(0)
    page = {"embed": lambda texts: rng.random((len(texts), 384), dtype=np.float32)}
    for block in re.findall(r"```python\n(.*?)```", WITH_A_DATABASE.read_text(), re.S):
        exec(compile(block, str(WITH_A_DATABASE), "exec"), page)
    index, db = page["index"], page["db"]

    ids = page["add_notes"](["apples", "pears"])
    # A crash after a flush whose commit never happened: a vector under the id that comes next.
    index.add(page["embed"](["lost"])[0], id=ids[-1] + 1)
    index.flush()
    assert page["add_notes"](["plums"]) == [ids[-1] + 1]
    assert {text for _, text in page["search"]("fruit", k=10)} == {"apples", "pears", "plums"}

    # A vector without a row, and a row without a vector.
    index.add(page["embed"](["stray"])[0], id=99)
    db.execute("INSERT INTO notes (id, text) VALUES (50, 'unindexed')")
    index.flush()
    assert len(page["search"]("fruit", k=10)) == 3  # The join drops the stray
    assert page["mend"]() == (1, 1)
    assert sorted(index.ids()) == sorted(r[0] for r in db.execute("SELECT id FROM notes"))
    page["delete_note"](ids[0])
    assert ids[0] not in index and "apples" not in {t for _, t in page["search"]("x", k=10)}
