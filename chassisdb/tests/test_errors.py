"""Each mistake gets the exception of its kind, a message with the value that was wrong, and a
line starting "help:" that says what to do instead."""

import numpy as np
import pytest

from chassis import (
    ChassisError,
    DimensionMismatchError,
    IdInUseError,
    IndexLockedError,
    IndexNotFoundError,
    IndexOptions,
    InvalidArgumentError,
    NotAnIndexError,
    OptionsMismatchError,
    ReadOnlyError,
    VectorIndex,
)


@pytest.fixture
def index(tmp_path):
    index = VectorIndex(tmp_path / "index.chassis", dimensions=4)
    index.add([1.0, 2.0, 3.0, 4.0], id=7)
    index.flush()
    yield index
    index.close()


MISTAKES = [
    # (what, call, exception, words the message has)
    ("vector of the wrong length", lambda ix, p: ix.add([1.0, 2.0]), DimensionMismatchError, "has 2 components"),
    ("one row as a 2-D array", lambda ix, p: ix.add(np.ones((1, 4))), DimensionMismatchError, "vector[0]"),
    ("NaN in a vector", lambda ix, p: ix.add([np.nan] * 4), InvalidArgumentError, "NaN at component 0"),
    ("id in use", lambda ix, p: ix.add([1.0] * 4, id=7), IdInUseError, "Id 7 is already in use"),
    ("string id", lambda ix, p: ix.add([1.0] * 4, id="doc-1"), TypeError, "'doc-1'"),
    ("batch of the wrong width", lambda ix, p: ix.add_batch(np.ones((2, 5))), DimensionMismatchError, "(2, 5)"),
    ("several queries at once", lambda ix, p: ix.search(np.ones((3, 4))), DimensionMismatchError, "search_batch"),
    ("queries of the wrong width", lambda ix, p: ix.search_batch(np.ones((2, 5))), DimensionMismatchError, "(2, 5)"),
    ("one query to search_batch", lambda ix, p: ix.search_batch([1.0] * 4), DimensionMismatchError, "call search"),
    ("NaN in a batch's query", lambda ix, p: ix.search_batch([[1.0] * 4, [np.nan] * 4]), InvalidArgumentError, "Query 1 of the batch has NaN"),
    ("k of 0 for a batch", lambda ix, p: ix.search_batch([[1.0] * 4], k=0), InvalidArgumentError, "k is 0"),
    ("k of 0", lambda ix, p: ix.search([1.0] * 4, k=0), InvalidArgumentError, "k is 0"),
    ("k as a float", lambda ix, p: ix.search([1.0] * 4, k=5.0), TypeError, "k is 5.0"),
    ("a predicate for allowed", lambda ix, p: ix.search([1.0] * 4, allowed=lambda i: True), TypeError, "allowed is a function"),
    ("a second writer", lambda ix, p: VectorIndex(p / "index.chassis", 4), IndexLockedError, "one writer at a time"),
    ("other dimensions", lambda ix, p: VectorIndex(p / "index.chassis", 8, read_only=True), DimensionMismatchError, "holds vectors of 4 dimensions, not 8"),
    ("a reader on a missing file", lambda ix, p: VectorIndex(p / "none.chassis", 4, read_only=True), IndexNotFoundError, "There is no index at"),
    ("a directory for a path", lambda ix, p: VectorIndex(p, 4), InvalidArgumentError, "is a directory"),
    ("a file that isn't an index", lambda ix, p: VectorIndex(_text(p), 4), NotAnIndexError, "is not a Chassis index"),
    ("dimensions of 0", lambda ix, p: VectorIndex(p / "new.chassis", 0), InvalidArgumentError, "dimensions is 0"),
    ("an unknown metric", lambda ix, p: IndexOptions(metric="dot").validate(), InvalidArgumentError, "inner product"),
    ("a misspelled option", lambda ix, p: IndexOptions(metirc="cosine"), TypeError, "did you mean 'metric'"),
    ("hnswlib's name for an option", lambda ix, p: VectorIndex(p / "new.chassis", 4, options={"ef": 64}), TypeError, "did you mean 'ef_search'"),
]


def _text(directory):
    path = directory / "notes.txt"
    path.write_text("not an index")
    return path


@pytest.mark.parametrize("what, call, exception, words", MISTAKES, ids=[m[0] for m in MISTAKES])
def test_a_mistake_says_what_was_wrong_and_what_to_do(index, tmp_path, what, call, exception, words):
    with pytest.raises(exception) as raised:
        call(index, tmp_path)
    message = str(raised.value)
    assert words in message
    assert "\nhelp: " in message


def test_a_reader_says_it_only_searches(index, tmp_path):
    reader = VectorIndex(tmp_path / "index.chassis", 4, read_only=True)
    for write in (lambda: reader.add([1.0] * 4), reader.flush, reader.compact, lambda: reader.delete(7)):
        with pytest.raises(ReadOnlyError, match="opened with read_only=True"):
            write()
    reader.close()


def test_reopening_with_another_metric_says_which_it_was_created_with(tmp_path):
    path = tmp_path / "cosine.chassis"
    VectorIndex(path, 4, options=IndexOptions(metric="cosine")).close()
    with pytest.raises(OptionsMismatchError, match="created with cosine distance"):
        VectorIndex(path, 4)


def test_every_error_is_a_chassis_error_and_a_builtin_of_its_kind(index):
    with pytest.raises(ValueError):
        index.add([1.0])
    with pytest.raises(ChassisError):
        index.add([1.0])


def test_other_libraries_names_for_metric_and_precision_are_taken():
    options = IndexOptions(metric="L2", precision="float16")
    options.validate()
    assert (options.metric, options.precision) == ("euclidean", "half")
