"""Compaction grouping, and parity with the one piece of `delta` that is Python.

`executeCompaction` and `executeZOrderBy` both hand off to a JVM builder, but
the Z-order column list is validated in Python first, and that check is exactly
what `mojo_delta_encoding.validate_zorder_columns` reproduces. The parity tests
call the real `delta.tables.DeltaOptimizeBuilder.executeZOrderBy` on a stub and
require the same exception and the same message.
"""

import importlib.util
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
# Compaction parity: port vs the reference Python sweep (bench.py)
# ---------------------------------------------------------------------------

def _ref_compaction_groups(sizes, min_size):
    """The reference sweep in bench.py, written out from the documented rule.

    A file at or above min_size stands alone; a run of smaller files closes once
    its total reaches min_size; a trailing run closes as one group. Comparisons
    use float semantics, so NaN never closes a group and inf always does.
    """
    sizes = np.asarray(sizes, dtype=np.float64)
    groups = []
    acc = 0.0
    first = 0
    for i, sz in enumerate(sizes):
        if sz >= min_size:
            if i > first:
                groups.append((first, i - first))
            groups.append((i, 1))
            first = i + 1
            acc = 0.0
        else:
            acc += sz
            if acc >= min_size:
                groups.append((first, i - first + 1))
                first = i + 1
                acc = 0.0
    if first < sizes.size:
        groups.append((first, sizes.size - first))
    return groups


def test_compaction_groups_match_reference_sweep_with_tolerance():
    """Port output compared against an independent Python sweep.

    File sizes are float64, so the comparison is written with an explicit
    numerical tolerance: the grouping decision itself is an exact float
    comparison, and rtol=1e-12/atol=1e-9 states that no rounding is allowed
    to move a boundary. Random sizes are checked through assert_allclose on
    the start/length arrays so the parity is numeric, not just structural.
    """
    rng = np.random.default_rng(22)
    for trial, min_size in enumerate((1.0, 64.0, 1e6), start=0):
        sizes = rng.gamma(2.0, 50.0, size=500) if trial == 0 else rng.uniform(
            0.0, 2.0 * min_size, size=500
        )
        got = mde.compaction_groups(sizes, min_size)
        want = _ref_compaction_groups(sizes, min_size)
        assert got == want
        got_arr = np.array(got, dtype=np.float64)
        want_arr = np.array(want, dtype=np.float64)
        np.testing.assert_allclose(
            got_arr, want_arr, rtol=1e-12, atol=1e-9,
            err_msg=f"trial={trial} min_size={min_size}",
        )


# ---------------------------------------------------------------------------
# Parity with the real `delta-encoding` package
# ---------------------------------------------------------------------------

def _delta_available() -> bool:
    return importlib.util.find_spec("delta") is not None


delta_tables = None
if _delta_available():
    delta_tables = importlib.import_module("delta.tables")

requires_delta = pytest.mark.skipif(
    not _delta_available(), reason="delta-encoding absent"
)


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


@requires_delta
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


@requires_delta
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


@requires_delta
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


# ---------------------------------------------------------------------------
# Source-level parity with upstream delta-encoding (no install required)
# ---------------------------------------------------------------------------


def _upstream_execute_zorder_by_source_shipped_in_wheel():
    """The upstream executeZOrderBy validation block, verbatim.

    Copied character-for-character from delta-spark 4.4.1,
    delta/tables.py lines 1682-1687 (and delta/connect/tables.py lines
    855-860, identical). It is the whole Python-side behaviour of
    DeltaOptimizeBuilder.executeZOrderBy: unpack a single list/tuple, then
    require type(c) is str with that exact message. The port implements this
    same block (python/mojo_delta_encoding/columns.py), and the tests below
    execute this verbatim copy so the comparison against the port does not
    depend on delta being installed.
    """
    return (
        "        if len(cols) == 1 and isinstance(cols[0], (list, tuple)):\n"
        "            cols = cols[0]\n"
        "        for c in cols:\n"
        "            if type(c) is not str:\n"
        "                errorMsg = \"Z-order column must be str. \"\n"
        "                errorMsg += \"Found %s with type %s\" % ((str(c)), str(type(c)))\n"
        "                raise TypeError(errorMsg)\n"
    )


def _upstream_validate(cols):
    """Run the verbatim upstream block on `cols`, return its verdict."""
    src = _upstream_execute_zorder_by_source_shipped_in_wheel()
    func_src = (
        "def _upstream_validate(cols):\n"
        + "".join(
            ("    " + line[len("        "):] if line.startswith("        ")
             else "    " + line)
            for line in src.splitlines(keepends=True)
        )
        + "    return list(cols)\n"
    )
    namespace: dict = {}
    exec(func_src, namespace)
    try:
        return ("ok", namespace["_upstream_validate"](cols))
    except TypeError as exc:
        return ("error", type(exc), str(exc))


@pytest.mark.parametrize(
    "cols",
    [
        (),
        ("",),
        ("a",),
        ("a", "b"),
        (["a", "b"],),
        (("x", "y", "z"),),
        (["a", "a"],),
        ("a", "a"),
        (("b", "a", "b"),),
    ],
)
def test_validate_zorder_columns_matches_verbatim_upstream_block(cols):
    kind, *rest = _upstream_validate(cols)
    assert kind == "ok", f"verbatim upstream accepted {[rest]}"
    assert mde.validate_zorder_columns(cols) == rest[0]
    # The verbatim block is char-for-char what the installed tests require.
    assert "type(c) is not str" in _upstream_execute_zorder_by_source_shipped_in_wheel()
    assert "Z-order column must be str. " in _upstream_execute_zorder_by_source_shipped_in_wheel()


@pytest.mark.parametrize(
    "cols, bad",
    [
        (("a", 3), 3),
        ((3,), 3),
        ((["a", "b", 4.5]), 4.5),
        (("a", None), None),
        (("a", b"bytes"), b"bytes"),
        (([b"a"],), b"a"),
        (("a", True), True),
    ],
)
def test_validate_zorder_columns_same_error_as_verbatim_upstream(cols, bad):
    kind, exc_type, message = _upstream_validate(cols)
    assert kind == "error"
    with pytest.raises(TypeError) as mine:
        mde.validate_zorder_columns(cols)
    assert type(mine.value) is exc_type
    assert str(mine.value) == message
    assert str(bad) in message


def test_validate_zorder_columns_rejects_str_subclass_like_verbatim_upstream():
    class Col(str):
        pass

    cols = (Col("a"),)
    kind, exc_type, message = _upstream_validate(cols)
    assert kind == "error"
    with pytest.raises(TypeError) as mine:
        mde.validate_zorder_columns(cols)
    assert type(mine.value) is exc_type
    assert str(mine.value) == message


def test_validate_zorder_columns_rejects_numpy_str_like_upstream():
    cols = (np.str_("a"),)
    kind, exc_type, message = _upstream_validate(cols)
    assert kind == "error"
    with pytest.raises(TypeError) as mine:
        mde.validate_zorder_columns(cols)
    assert type(mine.value) is exc_type
    assert str(mine.value) == message


# ---------------------------------------------------------------------------
# Compaction degenerate inputs: same value the Python sweep produces
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sizes, min_size",
    [
        (np.zeros(0), 100.0),
        (np.array([42.0]), 100.0),
        (np.array([42.0]), 0.0),
        (np.array([np.nan]), 100.0),
        (np.array([np.nan, 10.0, 10.0]), 100.0),
        (np.array([50.0, np.nan, 50.0]), 100.0),
        (np.array([np.inf, 10.0]), 100.0),
        (np.array([-np.inf, 10.0]), 100.0),
        (np.array([10.0]), float("nan")),
        (np.array([10.0]), float("inf")),
        (np.array([10.0, 10.0]), -1.0),
        (np.array([-5.0, 8.0, 3.0]), 10.0),
        (np.array([0.0, 0.0, 0.0]), 10.0),
    ],
)
def test_compaction_groups_degenerate_matches_reference(sizes, min_size):
    """Empty, single, NaN/inf and negative sizes must give the sweep's value.

    IEEE 754 comparisons make NaN never satisfy >=, so a NaN size can never
    close a group on its own and poisons the accumulator into a single final
    group; +inf always stands alone; -inf is just a negative size. The port
    uses the same float comparisons, so it must agree exactly.
    """
    assert mde.compaction_groups(sizes, min_size) == _ref_compaction_groups(
        sizes, min_size
    )


def test_compaction_groups_rejects_two_dimensional_sizes():
    with pytest.raises(ValueError, match="one-dimensional"):
        mde.compaction_groups(np.zeros((2, 2)), 100.0)
