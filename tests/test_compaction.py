"""Compaction grouping, and parity with the one piece of `delta` that is Python.

`executeCompaction` and `executeZOrderBy` both hand off to a JVM builder, but
the Z-order column list is validated in Python first, and that check is exactly
what `mojo_delta_encoding.validate_zorder_columns` reproduces. The parity tests
call the real `delta.tables.DeltaOptimizeBuilder.executeZOrderBy` on a stub and
require the same exception and the same message.
"""

import types

import numpy as np
import pytest

import mojo_delta_encoding as mde


# ---------------------------------------------------------------------------
# Compaction grouping
# ---------------------------------------------------------------------------


def test_compaction_groups_are_a_partition_of_the_files():
    """Every group must be a contiguous, non-overlapping, gap-free cover.

    A dropped trailing run or an off-by-one in the accumulator silently loses
    files from the rewrite, and only a partition check catches that.
    """
    rng = np.random.default_rng(0)
    for n in (1, 2, 17, 100, 999):
        sizes = rng.gamma(2.0, 50.0, size=n)
        groups = mde.compaction_groups(sizes, min_size=100.0)
        assert groups, "at least one group for a non-empty input"
        covered = []
        for start, length in groups:
            assert length >= 1
            covered.extend(range(start, start + length))
        assert covered == list(range(n))


def test_compaction_groups_accumulate_small_files_to_the_target():
    """The worked example: two small files alone stay under 128 MB, the third
    crosses it and closes the group."""
    sizes = np.array([100.0, 30.0, 40.0, 50.0, 200.0, 10.0, 10.0], dtype=np.float64)
    groups = mde.compaction_groups(sizes, min_size=128.0)
    # 100+30 = 130 >= 128 closes the first group; 40+50 = 90 then 200 is already
    # large, so 40, 50 flush as a group and 200 stands alone; the tail 10, 10
    # stays a short final group.
    assert groups == [(0, 2), (2, 2), (4, 1), (5, 2)]


def test_compaction_groups_leave_large_files_alone():
    """A file at or above the target is never merged with its neighbours."""
    sizes = np.array([500.0, 1.0, 1.0, 500.0, 1.0], dtype=np.float64)
    groups = mde.compaction_groups(sizes, min_size=100.0)
    assert groups == [(0, 1), (1, 2), (3, 1), (4, 1)]
    assert all(not (start <= 0 < start + length and length > 1) for start, length in groups)


def test_compaction_groups_close_a_trailing_run():
    """Small files at the end must not be silently dropped."""
    sizes = np.full(4, 10.0, dtype=np.float64)
    groups = mde.compaction_groups(sizes, min_size=1000.0)
    assert groups == [(0, 4)]


def test_compaction_groups_emit_no_group_for_empty_input():
    assert mde.compaction_groups(np.zeros(0, dtype=np.float64), 100.0) == []


def test_compaction_groups_reduce_the_file_count_when_it_can():
    """A directory of only small files must collapse to one group per target's
    worth of data; a directory of large files must not be rewritten at all."""
    rng = np.random.default_rng(1)
    small = rng.uniform(1.0, 20.0, size=200)
    groups = mde.compaction_groups(small, min_size=64.0)
    assert len(groups) < 200
    assert sum(length for _, length in groups) == 200

    large = np.full(50, 1000.0, dtype=np.float64)
    assert len(mde.compaction_groups(large, min_size=64.0)) == 50


def test_compaction_groups_are_order_preserving():
    """The planner never reorders files, so the caller's partition order and
    any merge predicates keyed on it still line up."""
    sizes = np.array([5.0, 60.0, 60.0, 5.0, 5.0, 5.0], dtype=np.float64)
    groups = mde.compaction_groups(sizes, min_size=100.0)
    flat = [i for start, length in groups for i in range(start, start + length)]
    assert flat == sorted(flat)
    assert flat == list(range(sizes.size))


def test_compaction_groups_hold_when_the_target_is_zero():
    """With a zero target every file qualifies, so every file is its own group."""
    sizes = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    assert mde.compaction_groups(sizes, min_size=0.0) == [(0, 1), (1, 1), (2, 1)]


# ---------------------------------------------------------------------------
# Parity with the real `delta-encoding` package
# ---------------------------------------------------------------------------

delta_tables = pytest.importorskip("delta.tables", reason="delta-encoding absent")


class _StubBuilder:
    """Enough of the builder's state for the argument check to run.

    `executeZOrderBy` validates every column before it touches `_jbuilder`, so a
    stub is enough to reach the real validation and a real `AttributeError` past
    it, which is how the accept-path is detected.
    """

    _spark = None
    _jbuilder = None
    _partitionFilters = []


def _real(cols):
    """Return `('ok', normalised)` or `('error', type, message)` from the real code."""
    try:
        delta_tables.DeltaOptimizeBuilder.executeZOrderBy(_StubBuilder(), *cols)
    except TypeError as exc:
        return ("error", type(exc), str(exc))
    except AttributeError:
        return ("past_validation", None, None)
    return ("unexpected", None, None)


@pytest.mark.parametrize(
    "cols",
    [
        ("a",),
        ("a", "b"),
        ("a", "b", "c"),
        (["a", "b"],),
        (("x", "y", "z"),),
    ],
)
def test_validate_zorder_columns_accepts_what_delta_accepts(cols):
    assert _real(cols)[0] == "past_validation"
    assert mde.validate_zorder_columns(cols) == list(
        cols[0] if len(cols) == 1 and isinstance(cols[0], (list, tuple)) else cols
    )


@pytest.mark.parametrize(
    "cols, bad",
    [
        (("a", 3), 3),
        ((3,), 3),
        ((["a", "b", 4.5]), 4.5),
        (("a", None), None),
        (("a", b"bytes"), b"bytes"),
    ],
)
def test_validate_zorder_columns_matches_delta_exactly(cols, bad):
    """Same exception class and same message, character for character.

    The message embeds `str(value)` and `type(value)`, so a formatter that drops
    either, or a `str` subclass accepted where upstream rejects one, fails here.
    """
    kind, exc_type, message = _real(cols)
    assert kind == "error"
    with pytest.raises(TypeError) as mine:
        mde.validate_zorder_columns(cols)
    assert type(mine.value) is exc_type
    assert str(mine.value) == message
    assert str(bad) in message


def test_validate_zorder_columns_rejects_a_str_subclass_like_delta():
    """Upstream uses `type(c) is not str`, so a str subclass is rejected too.
    A `isinstance` check here would wrongly accept it."""
    class Col(str):
        pass

    cols = (Col("a"),)
    assert _real(cols)[0] == "error"
    with pytest.raises(TypeError, match="Z-order column must be str"):
        mde.validate_zorder_columns(cols)


def test_validate_zorder_columns_preserves_order_and_duplicates():
    """The order is the Z-order priority, and Delta does not de-duplicate it."""
    assert mde.validate_zorder_columns(("b", "a", "b")) == ["b", "a", "b"]


def test_validate_zorder_columns_of_nothing_is_empty():
    assert mde.validate_zorder_columns(()) == []
