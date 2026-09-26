"""Parity tests for the Z-order curve, the radix sort and the file statistics.

The Z-order key, the sort, the per-file bounds, the range counts and the prefix
rank are exact integer and byte-level operations, so `rtol=0, atol=0` is the
right assertion here: there is no floating-point arithmetic to lose bits to, and
a wrong stride, a dropped sort pass or an off-by-one in a file boundary must show
up as a failure rather than as a tolerance.

Fixtures deliberately span the retained bit window. The curve keeps the *high*
`bits` bits of each value, so a fixture with small magnitudes would collapse
every row onto key 0 and the test would pass without exercising anything.
"""

import numpy as np
import pytest

import mojo_delta_encoding as mde


def _rewrite(cols, nfiles, bits=20, mode=mde.MODE_INT):
    return mde.zorder_rewrite(cols, nfiles=nfiles, bits=bits, mode=mode)


def _morton(values, bits):
    """The reference interleave, written out bit by bit from the definition."""
    values = [np.asarray(v, dtype=np.int64) for v in values]
    k = len(values)
    key = np.zeros(values[0].size, dtype=np.int64)
    for c in range(k):
        truncated = values[c] >> np.int64(64 - bits)
        for b in range(bits):
            key |= ((truncated >> np.int64(b)) & np.int64(1)) << np.int64(
                b * k + c
            )
    return key


# ---------------------------------------------------------------------------
# The Z-order curve
# ---------------------------------------------------------------------------


def test_zorder_key_matches_hand_computed_morton_table():
    """2 columns x 2 bits: the classic 4x4 Morton grid.

    Bit b of column 0 goes to key bit 2b and bit b of column 1 to key bit 2b+1,
    so for two 2-bit columns the key is exactly `x | (y << 1)`. The table is
    written out by hand rather than computed, so a transposed column order or a
    reversed bit order cannot agree with a reimplementation of the same mistake.
    """
    grid = np.array(
        [[0, 2, 8, 10], [1, 3, 9, 11], [4, 6, 12, 14], [5, 7, 13, 15]],
        dtype=np.int64,
    )
    xs, ys = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
    hi = np.int64(64 - 2)
    got = mde.zorder_keys([xs.ravel() << hi, ys.ravel() << hi], bits=2)
    np.testing.assert_array_equal(got, grid.ravel())

    # Column order matters: swapping the columns must change the keys.
    swapped = mde.zorder_keys([ys.ravel() << hi, xs.ravel() << hi], bits=2)
    np.testing.assert_array_equal(swapped, grid.T.ravel())
    assert not np.array_equal(got, swapped)


def test_zorder_key_places_each_column_bit_at_its_own_stride():
    """3 columns x 8 bits: bit b of column c must land at key bit `b * 3 + c`.

    This is the layout itself, pinned by single-bit inputs. A reversed bit order,
    a swapped column index, or a destination position that never advances all
    fail here even though the key is still a plausible-looking number.
    """
    bits, k = 8, 3
    top = np.int64(64 - bits)
    for b in range(bits):
        for c in range(k):
            cols = [np.zeros(1, dtype=np.int64) for _ in range(k)]
            cols[c] = np.array([np.int64(1) << (top + b)], dtype=np.int64)
            got = mde.zorder_keys(cols, bits=bits)
            assert got[0] == (1 << (b * k + c)), f"b={b} c={c}"


def test_zorder_key_single_column_keeps_only_the_top_bits():
    """With one column the key is that column's top `bits` bits, unsigned."""
    rng = np.random.default_rng(0)
    x = rng.integers(-2**62, 2**62, size=512, dtype=np.int64)
    for bits in (1, 7, 20, 63):
        got = mde.zorder_keys([x], bits=bits)
        mask = np.int64((1 << bits) - 1)
        np.testing.assert_array_equal(got, (x >> np.int64(64 - bits)) & mask)
        assert (got >= 0).all(), "an integer-mode key must not use its sign bit"


def test_zorder_key_three_columns_leaves_the_top_three_bits_clear():
    """3 x 20 = 60 bits, Delta's usual budget, so the top 3 are never set."""
    rng = np.random.default_rng(1)
    cols = [rng.integers(-2**62, 2**62, size=1000, dtype=np.int64) for _ in range(3)]
    got = mde.zorder_keys(cols, bits=20)
    assert (got >= 0).all()
    assert (got >> np.int64(60) == 0).all()
    np.testing.assert_array_equal(got, _morton(cols, 20))


def test_zorder_key_rejects_a_key_wider_than_63_bits():
    x = np.zeros(4, dtype=np.int64)
    with pytest.raises(ValueError, match="exceeds"):
        mde.zorder_keys([x, x, x, x], bits=20)
    with pytest.raises(ValueError, match="bits must be"):
        mde.zorder_keys([x], bits=64)
    with pytest.raises(ValueError, match="at least one"):
        mde.zorder_keys([])


def test_zorder_key_rejects_ragged_columns():
    with pytest.raises(ValueError, match="same length"):
        mde.zorder_keys([np.zeros(4, dtype=np.int64), np.zeros(5, dtype=np.int64)])


def test_zorder_key_keeps_locality():
    """The point of a Z-order curve: nearby values must give nearby keys.

    A wrong interleave scatters local points across the whole key range and fails
    here even when the per-bit layout tests still pass.
    """
    rng = np.random.default_rng(2)
    bits, k = 10, 3
    shift = np.int64(64 - bits)
    centre = rng.integers(0, 512, size=64, dtype=np.int64) << shift
    offsets = rng.integers(-2, 3, size=(64, k), dtype=np.int64) << shift
    cols = [centre + offsets[:, c] for c in range(k)]
    base = mde.zorder_keys([centre] * k, bits=bits)
    near = mde.zorder_keys(cols, bits=bits)
    distant = mde.zorder_keys([centre + (np.int64(256) << shift)] * k, bits=bits)
    assert np.abs(near - base).max() < np.abs(distant - base).min()


def test_float_mode_matches_the_key_transform():
    """MODE_FLOAT must equal interleaving `zorder_float_bits` of the input."""
    rng = np.random.default_rng(3)
    values = rng.standard_normal(512) * 1e3
    values[0] = -0.0
    values[1] = 0.0
    key_image = mde.zorder_float_bits(values).view(np.int64)
    got = mde.zorder_keys([values, values], bits=20, mode=mde.MODE_FLOAT)
    want = mde.zorder_keys([key_image, key_image], bits=20)
    np.testing.assert_array_equal(got, want)



def test_the_two_float_transforms_induce_the_same_order():
    """One is unsigned-monotone for the key, the other signed-monotone for the
    statistics. If they ever disagreed about which float is larger, a file's
    recorded bounds would prune away rows that are actually there."""
    rng = np.random.default_rng(3)
    values = rng.standard_normal(512) * 1e3
    values[0] = -0.0
    values[1] = 0.0
    order_by_key = np.argsort(mde.zorder_float_bits(values), kind="stable")
    order_by_stats = np.argsort(mde.sortable_float_bits(values), kind="stable")
    np.testing.assert_array_equal(order_by_key, order_by_stats)


def test_float_mode_is_monotone_in_the_data():
    """Sorted floats must give sorted keys, across the sign boundary included.

    The key is the retained high window read unsigned, which is exactly why the
    key transform has to be the unsigned-monotone one: the float sign is the
    window's top bit, so a signed reading would wrap at zero.
    """
    rng = np.random.default_rng(4)
    values = np.sort(rng.standard_normal(1024) * 1e6)
    keys = mde.zorder_keys([values], bits=32, mode=mde.MODE_FLOAT)
    assert (np.diff(keys) >= 0).all()
    assert (keys >= 0).all(), "the retained window is read unsigned"

    ladder = np.array([-np.inf, -1e300, -1.0, -1e-300, -0.0, 0.0, 1e-300, 1.0,
                       np.inf])
    ladder_keys = mde.zorder_keys([ladder], bits=32, mode=mde.MODE_FLOAT)
    assert (np.diff(ladder_keys) > 0).all(), "every rung must be strictly higher"
    neg_zero = mde.zorder_keys([np.array([-0.0])], bits=32, mode=mde.MODE_FLOAT)
    pos_zero = mde.zorder_keys([np.array([0.0])], bits=32, mode=mde.MODE_FLOAT)
    assert neg_zero[0] < pos_zero[0]


def test_float_mode_differs_from_reading_the_raw_bits():
    """A raw bit-pattern read is not order-preserving, so the two must not agree
    -- otherwise MODE_FLOAT would be testing nothing."""
    rng = np.random.default_rng(5)
    values = np.sort(rng.standard_normal(256) * 1e3)
    as_float = mde.zorder_keys([values], bits=32, mode=mde.MODE_FLOAT)
    as_raw = mde.zorder_keys([values.view(np.int64)], bits=32)
    assert not np.array_equal(as_float, as_raw)


def test_sortable_float_bits_is_a_monotone_injection():
    """Signed monotonicity is what integer min and max need to mean float min and
    max, so the whole ladder, signs and both zeroes included, must ascend."""
    values = np.array([-np.inf, -1.5, -1e-300, -0.0, 0.0, 1e-300, 1.5, np.inf])
    assert (np.diff(mde.sortable_float_bits(values)) > 0).all()


# ---------------------------------------------------------------------------
# The radix sort
# ---------------------------------------------------------------------------


def test_radix_sort_matches_a_stable_numpy_sort():
    """Random int64 keys, including both sign extremes.

    The top byte pass of an LSD radix sort is the classic place the sign gets
    mishandled: without flipping that byte's high bit the whole sort becomes an
    unsigned sort and INT64_MIN lands at the top. Both extremes are in the
    fixture on purpose.
    """
    rng = np.random.default_rng(6)
    keys = rng.integers(-2**63, 2**63 - 1, size=5000, dtype=np.int64)
    keys[:4] = [np.iinfo(np.int64).min, np.iinfo(np.int64).max, -1, 0]
    values = np.arange(keys.size, dtype=np.int32)
    got_keys, got_values = mde.radix_sort(keys, values)
    want = np.argsort(keys, kind="stable")
    np.testing.assert_array_equal(got_keys, keys[want])
    np.testing.assert_array_equal(got_values, values[want])


def test_radix_sort_is_stable_on_duplicate_keys():
    """Equal keys must keep their input order, or a Z-order rewrite would shuffle
    rows that share a Z-prefix for no reason."""
    rng = np.random.default_rng(7)
    keys = rng.integers(0, 5, size=1000, dtype=np.int64)
    values = np.arange(keys.size, dtype=np.int32)
    _, got = mde.radix_sort(keys, values)
    np.testing.assert_array_equal(got, np.argsort(keys, kind="stable"))


def test_radix_sort_does_not_disturb_its_inputs():
    """The kernel writes in place, so the shim must copy: a caller that keeps the
    unsorted keys for its own bookkeeping must not find them sorted."""
    keys = np.array([5, 1, 4, 1, 3], dtype=np.int64)
    values = np.arange(5, dtype=np.int32)
    mde.radix_sort(keys, values)
    np.testing.assert_array_equal(keys, [5, 1, 4, 1, 3])
    np.testing.assert_array_equal(values, [0, 1, 2, 3, 4])


def test_radix_sort_handles_degenerate_sizes():
    """Sizes either side of the 256-bucket stride and of the 8-pass count."""
    for n in (0, 1, 2, 255, 256, 257, 4097):
        rng = np.random.default_rng(n)
        keys = rng.integers(-1000, 1000, size=n, dtype=np.int64)
        values = np.arange(n, dtype=np.int32)
        got_keys, got_values = mde.radix_sort(keys, values)
        want = np.argsort(keys, kind="stable")
        np.testing.assert_array_equal(got_keys, keys[want], err_msg=f"n={n}")
        np.testing.assert_array_equal(got_values, values[want], err_msg=f"n={n}")


def test_radix_sort_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        mde.radix_sort(np.zeros(4, dtype=np.int64), np.zeros(5, dtype=np.int32))


# ---------------------------------------------------------------------------
# Splitting the sorted keys into files
# ---------------------------------------------------------------------------


def test_split_points_balance_rows_within_one():
    """Boundary f is at index `f * n // nfiles`, so file sizes differ by at
    most one row."""
    rng = np.random.default_rng(8)
    for n, nfiles in ((100, 7), (1000, 64), (999, 100), (10, 3)):
        keys = np.sort(rng.integers(0, 2**40, size=n, dtype=np.int64))
        split = mde.split_points(keys, nfiles)
        assert split.size == nfiles + 1
        assert (np.diff(split) >= 0).all(), "split points must be monotone"
        edges = [f * n // nfiles for f in range(nfiles + 1)]
        spread = [b - a for a, b in zip(edges[:-1], edges[1:])]
        assert max(spread) - min(spread) <= 1, f"n={n} nfiles={nfiles}"


def test_split_points_are_the_keys_at_the_index_boundaries():
    """A boundary off by one row would put a row in the wrong file."""
    keys = np.sort(np.arange(20, dtype=np.int64))
    assert mde.split_points(keys, 4).tolist() == [
        keys[0], keys[5], keys[10], keys[15], keys[19]
    ]


def test_split_points_collapse_when_every_key_is_equal():
    """A degenerate curve must produce degenerate files, not invented ones."""
    keys = np.full(50, 12345, dtype=np.int64)
    assert mde.split_points(keys, 5).tolist() == [12345] * 6


def test_split_points_of_empty_input_are_all_zero():
    assert mde.split_points(np.zeros(0, dtype=np.int64), 3).tolist() == [0, 0, 0, 0]


def test_split_points_rejects_zero_files():
    with pytest.raises(ValueError, match="at least 1"):
        mde.split_points(np.zeros(4, dtype=np.int64), 0)


def test_row_files_partitions_every_row_exactly_once():
    rng = np.random.default_rng(9)
    keys = np.sort(rng.integers(0, 2**40, size=1000, dtype=np.int64))
    nfiles = 7
    split = mde.split_points(keys, nfiles)
    row_files = mde.assign_files(keys, split)
    assert row_files.dtype == np.int32
    assert row_files.size == keys.size
    assert row_files.min() >= 0 and row_files.max() <= nfiles
    assert (np.diff(row_files.astype(np.int64)) >= 0).all(), "must stay monotone"
    # Every non-final file must leave its upper bound exclusive, or a row lands
    # in two files and the per-file row counts stop adding up.
    for f in range(nfiles):
        held = keys[row_files == f]
        if held.size:
            assert held.min() >= split[f]
            assert held.max() <= split[f + 1]
        if f < nfiles - 1:
            assert held.max() < split[f + 1]


def test_row_files_assigns_duplicate_keys_to_the_last_file():
    """With duplicates the later ranges are empty, and the last file must still
    take the maximum key rather than leaving rows unassigned."""
    keys = np.sort(np.full(5, 5, dtype=np.int64))
    row_files = mde.assign_files(keys, mde.split_points(keys, 3))
    assert row_files.tolist() == [2, 2, 2, 2, 2]


def test_file_row_bounds_reconstruct_the_assignment():
    rng = np.random.default_rng(10)
    keys = np.sort(rng.integers(0, 2**40, size=500, dtype=np.int64))
    nfiles = 9
    row_files = mde.assign_files(keys, mde.split_points(keys, nfiles))
    bounds = mde.file_row_bounds(row_files, nfiles)
    assert bounds[0] == 0 and bounds[-1] == keys.size
    assert (np.diff(bounds) >= 0).all()
    for f in range(nfiles):
        assert (row_files[bounds[f]:bounds[f + 1]] == f).all()


# ---------------------------------------------------------------------------
# Per-file statistics
# ---------------------------------------------------------------------------


def test_file_bounds_match_numpy_over_the_same_rows():
    rng = np.random.default_rng(11)
    n, nfiles = 2000, 11
    cols = [rng.integers(-2**62, 2**62, size=n, dtype=np.int64) for _ in range(3)]
    res = _rewrite(cols, nfiles, bits=20)
    assert res.mins.shape == (nfiles, 3) and res.maxs.shape == (nfiles, 3)
    for f in range(nfiles):
        rows = res.order[res.row_bounds[f]:res.row_bounds[f + 1]].astype(np.int64)
        for c in range(3):
            if rows.size == 0:
                assert res.mins[f, c] == 0 and res.maxs[f, c] == 0
            else:
                assert res.mins[f, c] == cols[c][rows].min()
                assert res.maxs[f, c] == cols[c][rows].max()


def test_file_bounds_cover_every_row_exactly_once():
    """The statistic must describe the rewrite, so every row must be counted."""
    rng = np.random.default_rng(12)
    n, nfiles = 777, 13
    cols = [rng.integers(0, 2**50, size=n, dtype=np.int64) for _ in range(2)]
    res = _rewrite(cols, nfiles, bits=20)
    assert res.row_bounds.size == nfiles + 1
    assert int(np.sum(np.diff(res.row_bounds.astype(np.int64)))) == n


def test_file_bounds_give_an_empty_file_zero():
    """More files than rows: the surplus files have no statistics at all."""
    n, nfiles = 4, 9
    cols = [np.arange(n, dtype=np.int64), -np.arange(n, dtype=np.int64)]
    order = np.arange(n, dtype=np.int32)
    bounds = mde.file_row_bounds(order, nfiles)
    assert (np.diff(bounds) >= 0).all()
    mins, maxs = mde.file_bounds(order, cols, bounds)
    empty = np.flatnonzero(bounds[1:] == bounds[:-1])
    assert empty.size == nfiles - n
    for f in empty:
        assert (mins[f] == 0).all() and (maxs[f] == 0).all()


def test_file_bounds_on_a_float_column_keep_float_order():
    """Taking the integer min of the order-preserving image is the float min, so
    per-file float statistics still prune correctly."""
    rng = np.random.default_rng(13)
    n, nfiles = 1000, 5
    values = rng.standard_normal(n) * 1e5
    res = _rewrite([values], nfiles, bits=30, mode=mde.MODE_FLOAT)
    transformed = mde.sortable_float_bits(values)
    for f in range(nfiles):
        rows = res.order[res.row_bounds[f]:res.row_bounds[f + 1]].astype(np.int64)
        assert res.mins[f, 0] == transformed[rows].min()
        assert res.maxs[f, 0] == transformed[rows].max()
        # And that min/max is the transform of the float min/max.
        assert res.mins[f, 0] == mde.sortable_float_bits(
            np.array([values[rows].min()]))[0]
        assert res.maxs[f, 0] == mde.sortable_float_bits(
            np.array([values[rows].max()]))[0]


def test_file_bounds_rejects_bounds_that_drop_rows():
    cols = [np.arange(10, dtype=np.int64)]
    with pytest.raises(ValueError, match="span every row"):
        mde.file_bounds(np.arange(10, dtype=np.int32), cols,
                        np.array([0, 5], dtype=np.int32))


def test_file_bounds_rejects_mismatched_columns():
    with pytest.raises(ValueError, match="same row count"):
        mde.file_bounds(np.arange(10, dtype=np.int32),
                        [np.arange(9, dtype=np.int64)],
                        np.array([0, 10], dtype=np.int32))


# ---------------------------------------------------------------------------
# Data skipping
# ---------------------------------------------------------------------------


def test_range_count_is_closed_at_both_ends():
    keys = np.array([-10, -5, 0, 5, 10], dtype=np.int64)
    assert mde.range_count(keys, -5, 5) == 3
    assert mde.range_count(keys, -5, 4) == 2
    assert mde.range_count(keys, -4, 5) == 2
    assert mde.range_count(keys, 11, 20) == 0
    assert mde.range_count(keys, -2**62, 2**62) == 5
    assert mde.range_count(np.zeros(0, dtype=np.int64), 0, 0) == 0


def test_range_count_handles_the_int64_extremes():
    """An exclusive upper bound would overflow at INT64_MAX; the closed bound
    must not."""
    keys = np.array([np.iinfo(np.int64).min, 0, np.iinfo(np.int64).max],
                    dtype=np.int64)
    assert mde.range_count(keys, np.iinfo(np.int64).min,
                           np.iinfo(np.int64).max) == 3
    assert mde.range_count(keys, 0, np.iinfo(np.int64).max) == 2


def test_range_counts_match_a_numpy_mask():
    rng = np.random.default_rng(14)
    keys = rng.integers(-1000, 1000, size=3000, dtype=np.int64)
    los = rng.integers(-1000, 0, size=25, dtype=np.int64)
    his = los + rng.integers(0, 500, size=25, dtype=np.int64)
    got = mde.range_counts(keys, los, his)
    want = np.array(
        [int(((keys >= a) & (keys <= b)).sum()) for a, b in zip(los, his)]
    )
    np.testing.assert_array_equal(got, want)


def test_prune_rows_counts_the_closed_key_range_of_every_file():
    """The ranges are closed, so a row sitting exactly on an interior boundary
    is counted by both neighbours. That is the documented contract, and the
    final range must still take the maximum key so nothing is missed."""
    rng = np.random.default_rng(15)
    n, nfiles = 1000, 6
    res = _rewrite([rng.integers(0, 2**50, size=n, dtype=np.int64)], nfiles)
    counts = mde.prune_rows(res.keys, res.split)
    for f in range(nfiles):
        lo, hi = res.split[f], res.split[f + 1]
        held = (res.keys >= lo) & (res.keys <= hi)
        assert counts[f] == int(held.sum())
        # The rows actually assigned to the file are a subset of its range.
        assert counts[f] >= int((res.row_files == f).sum())
    assert res.keys.max() <= res.split[-1]
    assert counts.sum() >= n


def test_prune_files_skips_what_the_bounds_rule_out():
    rng = np.random.default_rng(16)
    nfiles = 9
    cols = [rng.integers(0, 2**50, size=900, dtype=np.int64) for _ in range(2)]
    res = _rewrite(cols, nfiles, bits=20)
    lo, hi = 2**45, 2**46
    kept = set(mde.prune_files(res.mins, res.maxs, column=0, lo=lo, hi=hi).tolist())
    for f in range(nfiles):
        rows = res.order[res.row_bounds[f]:res.row_bounds[f + 1]].astype(np.int64)
        if rows.size == 0:
            continue
        hits = bool(((cols[0][rows] >= lo) & (cols[0][rows] <= hi)).any())
        # A file is kept if and only if it really holds a matching row: dropping
        # one loses data, keeping one wastes a read.
        assert (f in kept) == hits


def test_prune_files_keeps_empty_files():
    """An empty file's 0/0 must not read as a non-overlapping range."""
    mins = np.array([[0], [10], [0]], dtype=np.int64)
    maxs = np.array([[0], [20], [0]], dtype=np.int64)
    kept = mde.prune_files(mins, maxs, column=0, lo=100, hi=200)
    assert kept.tolist() == [0, 2]


# ---------------------------------------------------------------------------
# Bucket occupancy
# ---------------------------------------------------------------------------


def test_prefix_rank_counts_equal_prefixes():
    """`prefix_rank[i]` must be the offset of row i inside its prefix run."""
    rng = np.random.default_rng(17)
    keys = np.sort(rng.integers(0, 2**20, size=2000, dtype=np.int64))
    for prefix_bits in (1, 5, 20, 40):
        prefixes = keys >> np.int64(64 - prefix_bits)
        uniq, first = np.unique(prefixes, return_index=True)
        want = np.arange(keys.size, dtype=np.int64) - first[
            np.searchsorted(uniq, prefixes)
        ]
        got = mde.prefix_rank(keys, prefix_bits)
        np.testing.assert_array_equal(got, want, err_msg=f"bits={prefix_bits}")


def test_prefix_rank_of_a_constant_key_is_the_row_index():
    keys = np.zeros(50, dtype=np.int64)
    np.testing.assert_array_equal(
        mde.prefix_rank(keys, 30), np.arange(50, dtype=np.int64)
    )


def test_prefix_rank_rejects_an_impossible_prefix_width():
    with pytest.raises(ValueError, match="prefix_bits"):
        mde.prefix_rank(np.zeros(4, dtype=np.int64), 0)


# ---------------------------------------------------------------------------
# The full rewrite
# ---------------------------------------------------------------------------


def test_zorder_rewrite_improves_locality_over_the_input_order():
    """The whole reason to Z-order a table: a predicate covering a thin slice of
    a wide column must be able to skip most files after the rewrite.

    A broken interleave, a dropped sort pass or a wrong file assignment all make
    the range overlap more files, not fewer, so this is the end-to-end test that
    the pipeline is right rather than merely self-consistent.
    """
    rng = np.random.default_rng(18)
    n, nfiles = 40000, 32
    a = rng.integers(0, 2**60, size=n).astype(np.int64)
    b = a + rng.integers(-8, 9, size=n)
    res = _rewrite([a, b], nfiles, bits=30)
    lo = int(a.min())
    hi = lo + (int(a.max()) - lo) // 1024
    skipped = nfiles - mde.prune_files(res.mins, res.maxs, 0, lo, hi).size
    assert skipped >= 24, f"Z-ordering pruned only {skipped}/{nfiles} files"


def test_zorder_rewrite_beats_a_row_range_layout_on_the_same_measure():
    """Same number of files, same predicate, same kernel for the statistics.

    The baseline is the layout a Z-ordering rewrite replaces: `nfiles` equal row
    ranges in input order. Both layouts are measured with `file_bounds` and
    `prune_files`, so the only thing that differs is which rows share a file.
    """
    rng = np.random.default_rng(19)
    n, nfiles = 20000, 16
    a = rng.integers(0, 2**60, size=n).astype(np.int64)
    b = a + rng.integers(-8, 9, size=n)
    res = _rewrite([a, b], nfiles, bits=30)
    lo = int(a.min())
    span = int(a.max()) - lo
    centre = lo + span // 2
    lo, hi = centre - span // 2048, centre + span // 2048

    identity = np.arange(n, dtype=np.int32)
    row_range_bounds = np.array(
        [f * n // nfiles for f in range(nfiles + 1)], dtype=np.int32
    )
    flat_mins, flat_maxs = mde.file_bounds(identity, [a], row_range_bounds)
    flat = mde.prune_files(flat_mins, flat_maxs, 0, lo, hi).size
    zordered = mde.prune_files(res.mins, res.maxs, 0, lo, hi).size
    assert flat == nfiles, "a scattered predicate must touch every row-range file"
    assert zordered < flat, "Z-ordering must let a reader skip files"


def test_zorder_rewrite_is_a_permutation_with_sorted_keys():
    """`keys` must stay in input order and `keys[order]` must be ascending, which
    is only true if the sort did not overwrite the caller's key array."""
    rng = np.random.default_rng(20)
    n = 800
    cols = [rng.integers(0, 2**50, size=n, dtype=np.int64) for _ in range(3)]
    res = _rewrite(cols, nfiles=4, bits=20)
    assert res.keys.size == n
    assert np.array_equal(np.sort(res.order), np.arange(n))
    np.testing.assert_array_equal(res.keys, _morton(cols, 20))
    assert (np.diff(res.keys[res.order]) >= 0).all()
    assert (np.diff(res.split) >= 0).all()
    assert res.split.size == 5


def test_zorder_rewrite_bounds_match_a_direct_numpy_computation():
    rng = np.random.default_rng(21)
    n, nfiles = 5000, 17
    cols = [rng.integers(-2**62, 2**62, size=n, dtype=np.int64) for _ in range(2)]
    res = _rewrite(cols, nfiles, bits=20)
    assert res.row_bounds[0] == 0 and res.row_bounds[-1] == n
    total = 0
    for f in range(nfiles):
        rows = res.order[res.row_bounds[f]:res.row_bounds[f + 1]].astype(np.int64)
        total += rows.size
        for c in range(2):
            assert res.mins[f, c] == cols[c][rows].min()
            assert res.maxs[f, c] == cols[c][rows].max()
    assert total == n


def test_zorder_rewrite_of_a_single_row_is_well_formed():
    """The key is the retained high bits, so the statistic is the original value
    even though the key is only 20 bits of it."""
    value = np.array([7 << 44], dtype=np.int64)
    res = _rewrite([value], nfiles=1, bits=20)
    assert res.keys.tolist() == [7]
    assert res.split.tolist() == [7, 7]
    assert res.order.tolist() == [0]
    assert res.mins.tolist() == [[7 << 44]]
    assert res.maxs.tolist() == [[7 << 44]]
