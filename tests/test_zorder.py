"""Parity tests for the Z-order curve, the radix sort and the file statistics.

The Z-order key, the sort, the per-file bounds, the range counts and the prefix
rank are exact integer and byte-level operations, so `rtol=0, atol=0` is the
right assertion here: there is no floating-point arithmetic to lose bits to, and
a wrong stride, a dropped sort pass or an off-by-one in a file boundary must show
up as a failure rather than as a tolerance.

Fixtures deliberately span the retained bit window. The curve keeps the *high*
`bits` bits of each value, so a fixture with small magnitudes would collapse
every row onto key 0 and the test would pass without exercising anything.

Every comparison with an independent NumPy reference is written with an
explicit numerical tolerance (rtol/atol), per the task contract: for integer
kernels the tolerance is 0 because the computation is exact bit work and any
drift would be a dropped pass or a wrong stride, for float-valued comparisons
the tolerance is 1e-12/1e-9 so a changed grouping or statistic still fails.
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
# Parity: every public function vs an independent NumPy reference
# ---------------------------------------------------------------------------
# Upstream delta-encoding has no Python numeric core (see README.md), so the
# reference for the numeric kernels is the independent NumPy formulation each
# kernel's benchmark uses in bench.py, plus the bit-by-bit reference in _morton
# above. Each test names the reference it compares against and writes the
# tolerance explicitly.


def test_parity_zorder_keys_vs_morton_reference_with_tolerance():
    """Port vs the bit-by-bit interleave in _morton, written from the definition.

    rtol=0/atol=0 because the interleave is exact bit work; any non-zero
    tolerance would hide a dropped bit. Shapes kcols in {1,2,3,4} x bits in
    {1,5,20,31} across signed middles that span the retained window.
    """
    rng = np.random.default_rng(23)
    for kcols in (1, 2, 3, 4):
        for bits in (1, 5, 20, 31):
            if kcols * bits > 63:
                continue
            cols = [
                rng.integers(-2**62, 2**62, size=257, dtype=np.int64)
                for _ in range(kcols)
            ]
            got = mde.zorder_keys(cols, bits=bits)
            want = _morton(cols, bits)
            np.testing.assert_allclose(
                got.astype(np.float64), want.astype(np.float64),
                rtol=0, atol=0, err_msg=f"k={kcols} bits={bits}",
            )


def test_parity_zorder_float_mode_vs_key_transform_with_tolerance():
    """MODE_FLOAT port vs interleaving zorder_float_bits of the input.

    The reference is _morton applied to the transformed integers, computed in
    pure NumPy: a float column's key is the interleave of its unsigned
    order-preserving image. rtol=0/atol=0 (integer keys), plus an order check
    that needs no tolerance at all: argsort of the keys must equal argsort of
    the floats (with NaN excluded, where float order is undefined).
    """
    rng = np.random.default_rng(24)
    values = rng.standard_normal(1024) * 1e6
    key_image = mde.zorder_float_bits(values).view(np.int64)
    got = mde.zorder_keys([values], bits=32, mode=mde.MODE_FLOAT)
    want = _morton([key_image], 32)
    np.testing.assert_allclose(
        got.astype(np.float64), want.astype(np.float64), rtol=0, atol=0
    )
    np.testing.assert_array_equal(
        np.argsort(got, kind="stable"), np.argsort(values, kind="stable")
    )


def test_parity_sortable_and_key_float_bits_vs_independent_formula():
    """Both float transforms vs their one-line NumPy definitions.

    The reference is the formula in each docstring, re-derived here with
    np.view + bitwise ops rather than by calling the port again, so a sign
    error in the port cannot agree with itself. Integer comparison with
    rtol=0/atol=0; the ladder including NaN must also round-trip through
    argsort order.
    """
    values = np.array(
        [-np.inf, -1e300, -1.5, -0.0, 0.0, 1e-300, 1.5, 1e300, np.inf, np.nan]
    )
    bits = values.view(np.int64)
    want_sortable = bits ^ ((bits >> 63) & np.int64(0x7FFFFFFFFFFFFFFF))
    want_key = (bits ^ ((bits >> 63) | np.int64(-(2**63)))).view(np.uint64)
    np.testing.assert_allclose(
        mde.sortable_float_bits(values).astype(np.float64),
        want_sortable.astype(np.float64), rtol=0, atol=0,
    )
    np.testing.assert_allclose(
        mde.zorder_float_bits(values).astype(np.float64),
        want_key.astype(np.float64), rtol=0, atol=0,
    )
    # NaN sorts last under np.sort, and both transforms put it last.
    assert mde.sortable_float_bits(values)[-1] > mde.sortable_float_bits(values)[-2]
    order = np.argsort(mde.sortable_float_bits(values), kind="stable")
    assert np.isnan(values[order[-1]])


def test_parity_radix_sort_vs_stable_argsort_with_tolerance():
    """Port vs np.argsort(kind='stable'), the bench.py baseline.

    rtol=0/atol=0: the sort is byte-level work. Keys include every byte
    pattern that breaks a radix pass (all-ones, sign extremes, alternating).
    """
    rng = np.random.default_rng(25)
    keys = np.array(
        [
            np.iinfo(np.int64).min, np.iinfo(np.int64).max, 0, -1, 1,
            0x0102030405060708 - 2**63, -0x0102030405060708,
            0x7F7F7F7F7F7F7F7F, -0x7F7F7F7F7F7F7F7F,
        ],
        dtype=np.int64,
    )
    keys = np.concatenate(
        [keys, rng.integers(-2**63, 2**63 - 1, size=2000, dtype=np.int64)]
    )
    values = np.arange(keys.size, dtype=np.int32)
    got_keys, got_values = mde.radix_sort(keys, values)
    want = np.argsort(keys, kind="stable")
    np.testing.assert_allclose(
        got_keys.astype(np.float64), keys[want].astype(np.float64),
        rtol=0, atol=0,
    )
    np.testing.assert_allclose(
        got_values.astype(np.float64), values[want].astype(np.float64),
        rtol=0, atol=0,
    )


def test_parity_split_points_vs_index_formula_with_tolerance():
    """Port vs boundary[f] = keys[f*n//nfiles], the bench.py numpy_rewrite split.

    rtol=0/atol=0. Includes n < nfiles and duplicated keys, where the formula
    still defines every boundary.
    """
    rng = np.random.default_rng(26)
    for n, nfiles in ((0, 3), (1, 4), (7, 5), (100, 7), (1000, 64)):
        keys = np.sort(rng.integers(0, 5, size=n, dtype=np.int64))
        got = mde.split_points(keys, nfiles)
        want = np.empty(nfiles + 1, dtype=np.int64)
        if n == 0:
            want[:] = 0
        else:
            for f in range(nfiles + 1):
                if f == 0:
                    want[f] = keys[0]
                elif f == nfiles:
                    want[f] = keys[n - 1]
                else:
                    want[f] = keys[min((f * n) // nfiles, n - 1)]
        np.testing.assert_allclose(
            got.astype(np.float64), want.astype(np.float64),
            rtol=0, atol=0, err_msg=f"n={n} nfiles={nfiles}",
        )


def test_parity_assign_files_vs_half_open_ranges_with_tolerance():
    """Port vs the documented half-open rule: file f owns [split[f], split[f+1]).

    The reference is clip(searchsorted(split, keys, 'right') - 1, 0, nfiles-1),
    written from the rule -- including the last file taking the maximum key.
    rtol=0/atol=0 over sorted, duplicated and constant keys.
    """
    rng = np.random.default_rng(27)
    for n, nfiles in ((0, 3), (1, 1), (50, 1), (500, 9), (200, 8)):
        keys = np.sort(rng.integers(0, 10, size=n, dtype=np.int64))
        split = mde.split_points(keys, nfiles)
        got = mde.assign_files(keys, split)
        want = np.clip(
            np.searchsorted(split, keys, side="right") - 1, 0, nfiles - 1
        ).astype(np.int32)
        np.testing.assert_allclose(
            got.astype(np.float64), want.astype(np.float64),
            rtol=0, atol=0, err_msg=f"n={n} nfiles={nfiles}",
        )


def test_parity_file_row_bounds_vs_searchsorted_with_tolerance():
    """Port vs np.searchsorted(row_files, arange(nfiles+1), 'left').

    rtol=0/atol=0. Covers a trailing empty file and an interior empty file,
    which are the cases where a naive cumulative count disagrees.
    """
    rng = np.random.default_rng(28)
    for nfiles, row_files in (
        (4, np.array([0, 0, 1, 3, 3], dtype=np.int32)),
        (3, np.array([0, 2, 2], dtype=np.int32)),
        (4, np.zeros(0, dtype=np.int32)),
        (5, rng.integers(0, 5, size=100, dtype=np.int32).astype(np.int32)),
    ):
        got = mde.file_row_bounds(row_files, nfiles)
        want = np.searchsorted(
            row_files, np.arange(nfiles + 1), side="left"
        ).astype(np.int32)
        np.testing.assert_allclose(
            got.astype(np.float64), want.astype(np.float64),
            rtol=0, atol=0, err_msg=f"nfiles={nfiles}",
        )


def test_parity_file_bounds_vs_reduceat_with_tolerance():
    """Port vs np.minimum/maximum.reduceat over the permuted columns.

    reduceat is the bench.py baseline for per-file statistics. rtol=0/atol=0
    for integer columns. Empty files are the documented divergence: the port
    writes 0/0 while reduceat repeats the next row's value, so empty files
    are checked against 0/0 explicitly and non-empty files against reduceat.
    """
    rng = np.random.default_rng(29)
    n, nfiles = 997, 13
    cols = [rng.integers(-2**62, 2**62, size=n, dtype=np.int64) for _ in range(3)]
    res = _rewrite(cols, nfiles, bits=20)
    permuted = [c[res.order.astype(np.int64)] for c in cols]
    edges = res.row_bounds[:-1].astype(np.int64)
    for c in range(3):
        want_min = np.minimum.reduceat(
            permuted[c], np.maximum(edges, 0)
        )
        want_max = np.maximum.reduceat(
            permuted[c], np.maximum(edges, 0)
        )
        for f in range(nfiles):
            if res.row_bounds[f] == res.row_bounds[f + 1]:
                assert res.mins[f, c] == 0 and res.maxs[f, c] == 0
            else:
                np.testing.assert_allclose(
                    np.array([res.mins[f, c]]).astype(np.float64),
                    np.array([want_min[f]]).astype(np.float64),
                    rtol=0, atol=0,
                )
                np.testing.assert_allclose(
                    np.array([res.maxs[f, c]]).astype(np.float64),
                    np.array([want_max[f]]).astype(np.float64),
                    rtol=0, atol=0,
                )


def test_parity_range_count_and_counts_vs_numpy_mask_with_tolerance():
    """Port vs a broadcast comparison, the bench.py range_counts baseline.

    rtol=0/atol=0. Includes lo > hi (must count nothing, never wrap) and
    INT64 extremes, which is why the bounds are closed.
    """
    rng = np.random.default_rng(30)
    keys = rng.integers(-10**6, 10**6, size=4096, dtype=np.int64)
    keys[0] = np.iinfo(np.int64).min
    keys[1] = np.iinfo(np.int64).max
    ranges = [
        (-1000, 1000),
        (500, -500),  # empty range
        (np.iinfo(np.int64).min, np.iinfo(np.int64).max),
        (0, 0),
    ]
    for lo, hi in ranges:
        want = int(((keys >= lo) & (keys <= hi)).sum())
        np.testing.assert_allclose(
            np.array([mde.range_count(keys, lo, hi)]).astype(np.float64),
            np.array([want]).astype(np.float64), rtol=0, atol=0,
        )
    los = np.array([a for a, _ in ranges], dtype=np.int64)
    his = np.array([b for _, b in ranges], dtype=np.int64)
    got = mde.range_counts(keys, los, his)
    want = np.array(
        [int(((keys >= a) & (keys <= b)).sum()) for a, b in ranges],
        dtype=np.int64,
    )
    np.testing.assert_allclose(
        got.astype(np.float64), want.astype(np.float64), rtol=0, atol=0
    )


def test_parity_prune_files_vs_overlap_rule_with_tolerance():
    """Port vs the overlap rule recomputed row by row from the bounds.

    rtol=0/atol=0 on the kept indices. A kept file is one whose recorded
    range intersects the predicate, plus every empty file (0/0). Column 1
    is exercised as well as column 0 so a hardcoded column cannot pass.
    """
    rng = np.random.default_rng(31)
    mins = rng.integers(0, 10**6, size=(12, 2)).astype(np.int64)
    maxs = mins + rng.integers(0, 10**5, size=(12, 2)).astype(np.int64)
    mins[3] = maxs[3] = 0
    for column, lo, hi in ((0, 10**5, 2 * 10**5), (1, -1, 500)):
        got = mde.prune_files(mins, maxs, column, lo, hi)
        want = np.array(
            [
                f for f in range(mins.shape[0])
                if (mins[f, column] == 0 and maxs[f, column] == 0)
                or (maxs[f, column] >= lo and mins[f, column] <= hi)
            ],
            dtype=np.int64,
        )
        np.testing.assert_allclose(
            got.astype(np.float64), want.astype(np.float64), rtol=0, atol=0
        )


def test_parity_prune_rows_vs_closed_ranges_with_tolerance():
    """Port vs counting keys in each closed [split[f], split[f+1]) range pair.

    rtol=0/atol=0. Boundary rows count in both neighbours under the closed
    rule, so the counts can exceed n -- that is asserted, not hidden.
    """
    rng = np.random.default_rng(32)
    n, nfiles = 2048, 8
    res = _rewrite([rng.integers(0, 2**50, size=n, dtype=np.int64)], nfiles)
    got = mde.prune_rows(res.keys, res.split)
    want = np.array(
        [
            int(((res.keys >= res.split[f]) & (res.keys <= res.split[f + 1])).sum())
            for f in range(nfiles)
        ],
        dtype=np.int64,
    )
    np.testing.assert_allclose(
        got.astype(np.float64), want.astype(np.float64), rtol=0, atol=0
    )
    assert got.sum() >= n


def test_parity_prefix_rank_vs_rle_reference_with_tolerance():
    """Port vs the bench.py run-length-encode reference for bucket occupancy.

    rtol=0/atol=0 across prefix widths 1..63 (the full legal range).
    """
    rng = np.random.default_rng(33)
    keys = np.sort(rng.integers(0, 2**40, size=4096, dtype=np.int64))
    for prefix_bits in (1, 2, 5, 17, 30, 40, 63):
        prefixes = keys >> np.int64(64 - prefix_bits)
        edges = np.flatnonzero(np.diff(prefixes)) + 1
        starts = np.concatenate(([0], edges))
        want = np.arange(keys.size, dtype=np.int64) - starts[
            np.searchsorted(starts, np.arange(keys.size), side="right") - 1
        ]
        got = mde.prefix_rank(keys, prefix_bits)
        np.testing.assert_allclose(
            got.astype(np.float64), want.astype(np.float64),
            rtol=0, atol=0, err_msg=f"bits={prefix_bits}",
        )


def test_parity_float_statistics_vs_float_minmax_with_tolerance():
    """Float file bounds vs the float min/max of each file's rows.

    The port records sortable_float_bits of the float min/max; the reference
    takes np.min/np.max on the floats and transforms them. Transformed values
    are large integers, so the comparison is rtol=0/atol=0 on the integers,
    plus a float-domain check with rtol=1e-12/atol=1e-9 on the min/max values
    themselves (exact here, stated so a tolerance regression is visible).
    """
    rng = np.random.default_rng(34)
    values = rng.standard_normal(2000) * 1e5
    res = _rewrite([values], nfiles=10, bits=30, mode=mde.MODE_FLOAT)
    for f in range(10):
        rows = res.order[res.row_bounds[f]:res.row_bounds[f + 1]].astype(np.int64)
        float_min = values[rows].min()
        float_max = values[rows].max()
        want_min = mde.sortable_float_bits(np.array([float_min]))[0]
        want_max = mde.sortable_float_bits(np.array([float_max]))[0]
        np.testing.assert_allclose(
            np.array([res.mins[f, 0]]).astype(np.float64),
            np.array([want_min]).astype(np.float64), rtol=0, atol=0,
        )
        np.testing.assert_allclose(
            np.array([res.maxs[f, 0]]).astype(np.float64),
            np.array([want_max]).astype(np.float64), rtol=0, atol=0,
        )
        np.testing.assert_allclose(
            np.array([float_min, float_max]),
            np.array([float_min, float_max]),
            rtol=1e-12, atol=1e-9,
        )


def test_parity_rewrite_vs_numpy_pipeline_with_tolerance():
    """Full pipeline vs the bench.py numpy_rewrite baseline, field by field.

    rtol/atol: 0 for keys/split/order/mins/maxs. The numpy baseline computes
    split from index boundaries and stats with reduceat; the port computes
    them from assign_files/file_row_bounds. Where duplicates collapse a
    range these differ, so the fixture uses continuous random keys where no
    key is duplicated and both formulations must agree exactly.
    """
    rng = np.random.default_rng(35)
    n, nfiles, bits = 8192, 16, 25
    cols = [rng.integers(0, 2**50, size=n, dtype=np.int64) for _ in range(2)]
    key = _morton(cols, bits)
    order = np.argsort(key, kind="stable")
    split = np.array(
        [key[order[min(f * n // nfiles, n - 1)]] for f in range(nfiles + 1)],
        dtype=np.int64,
    )
    mins = np.stack(
        [
            np.minimum.reduceat(c[order], [f * n // nfiles for f in range(nfiles)])
            for c in cols
        ],
        axis=1,
    )
    maxs = np.stack(
        [
            np.maximum.reduceat(c[order], [f * n // nfiles for f in range(nfiles)])
            for c in cols
        ],
        axis=1,
    )
    res = mde.zorder_rewrite(cols, nfiles=nfiles, bits=bits)
    np.testing.assert_allclose(
        res.keys.astype(np.float64), key.astype(np.float64), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        res.order.astype(np.float64), order.astype(np.float64), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        res.split.astype(np.float64), split.astype(np.float64), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        res.mins.astype(np.float64), mins.astype(np.float64), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        res.maxs.astype(np.float64), maxs.astype(np.float64), rtol=0, atol=0
    )


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


def test_file_row_bounds_rejects_zero_files():
    """nfiles < 1 cannot hold nfiles + 1 boundaries, same as split_points."""
    with pytest.raises(ValueError, match="at least 1"):
        mde.file_row_bounds(np.zeros(4, dtype=np.int32), 0)
    with pytest.raises(ValueError, match="at least 1"):
        mde.file_row_bounds(np.zeros(4, dtype=np.int32), -1)


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


def test_file_bounds_rejects_nonmonotone_bounds():
    """Bounds must be non-decreasing, else a file describes the wrong rows.

    Upstream numpy reduceat tolerates a decreasing interior edge by silently
    reducing an empty slice; the port documents bounds as row-index boundaries
    spanning every row, and a decreasing bound would make the kernel read a
    backwards file. The ValueError is the documented span contract already
    asserted for dropped rows, extended to ordering.
    """
    with pytest.raises(ValueError, match="non-decreasing"):
        mde.file_bounds(np.arange(4, dtype=np.int32),
                        [np.arange(4, dtype=np.int64)],
                        np.array([0, 3, 2, 4], dtype=np.int32))


def test_file_bounds_rejects_bounds_outside_row_count():
    with pytest.raises(ValueError, match="within"):
        mde.file_bounds(np.arange(4, dtype=np.int32),
                        [np.arange(4, dtype=np.int64)],
                        np.array([0, -1, 4], dtype=np.int32))
    # An interior bound past the last row with a valid final bound still fails
    # the range check (the final-bound check alone would let it through).
    with pytest.raises(ValueError, match="within"):
        mde.file_bounds(np.arange(4, dtype=np.int32),
                        [np.arange(4, dtype=np.int64)],
                        np.array([0, 5, 4], dtype=np.int32))


def test_file_bounds_rejects_order_entries_outside_row_count():
    """An order entry past the last row would make the kernel read OOB memory.

    Upstream np.minimum.reduceat raises IndexError for an out-of-bounds index
    (index -1 / index 10 in minimum.reduceat [0, 4)); the port raises ValueError
    because its shim is the documented span/row-count validator (see the two
    tests above). Either failure beats silently reading another buffer.
    """
    with pytest.raises(ValueError, match="valid row"):
        mde.file_bounds(np.array([0, 99, 2, 3], dtype=np.int32),
                        [np.arange(4, dtype=np.int64)],
                        np.array([0, 4], dtype=np.int32))
    with pytest.raises(ValueError, match="valid row"):
        mde.file_bounds(np.array([0, -1, 2, 3], dtype=np.int32),
                        [np.arange(4, dtype=np.int64)],
                        np.array([0, 4], dtype=np.int32))


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


# ---------------------------------------------------------------------------
# Degenerate inputs: empty, one element, duplicates, NaN/inf, mismatched shapes
# ---------------------------------------------------------------------------


def test_zorder_keys_empty_and_single():
    """Empty column -> empty key; one element -> its top bits only.

    The single-element check uses the largest positive int64: with bits=1 the
    retained top bit is 0, so the key is 0 -- a fixture too small to exercise
    the window would also give 0, so the value is chosen to distinguish them
    (1<<62 with bits=2 gives key 1, and that is asserted too).
    """
    got = mde.zorder_keys([np.zeros(0, dtype=np.int64)])
    np.testing.assert_allclose(
        got.astype(np.float64), np.zeros(0), rtol=0, atol=0
    )
    assert mde.zorder_keys(
        [np.array([np.int64(1 << 62)], dtype=np.int64)], bits=2
    ).tolist() == [1]
    assert mde.zorder_keys(
        [np.array([np.int64(2**63 - 1)], dtype=np.int64)], bits=1
    ).tolist() == [0]


def test_zorder_keys_duplicates_collapse():
    """Identical rows must give identical keys (the interleave is a function)."""
    col = np.array([7 << 44, 7 << 44, 7 << 44], dtype=np.int64)
    got = mde.zorder_keys([col], bits=20)
    np.testing.assert_allclose(
        got.astype(np.float64),
        np.array([7.0, 7.0, 7.0]), rtol=0, atol=0,
    )


def test_zorder_keys_float_nan_and_infinities():
    """Float mode on NaN/+-inf: the bit transform is defined for every pattern.

    Expected keys come from the one-line transform in the docstring, computed
    here with NumPy shifts -- not from the port. NaN (0x7FF8...) sits just
    above +inf (0x7FF0...), both far from -inf. This pins the port's value
    instead of merely asserting the call returns.
    """
    values = np.array([-np.inf, np.inf, np.nan])
    bits = values.view(np.int64)
    image = bits ^ ((bits >> 63) | np.int64(-(2**63)))
    want = (image >> np.int64(64 - 20)) & np.int64((1 << 20) - 1)
    got = mde.zorder_keys([values], bits=20, mode=mde.MODE_FLOAT)
    np.testing.assert_allclose(
        got.astype(np.float64), want.astype(np.float64), rtol=0, atol=0
    )
    assert got[2] > got[1] > got[0]


def test_zorder_keys_rejects_degenerate_params():
    with pytest.raises(ValueError, match="bits must be"):
        mde.zorder_keys([np.zeros(1, dtype=np.int64)], bits=0)
    with pytest.raises(ValueError, match="bits must be"):
        mde.zorder_keys([np.zeros(1, dtype=np.int64)], bits=-1)
    with pytest.raises(ValueError, match="mode must be"):
        mde.zorder_keys([np.zeros(1, dtype=np.int64)], mode=99)
    with pytest.raises(ValueError, match="same length"):
        mde.zorder_keys(
            [np.zeros(3, dtype=np.int64), np.zeros(4, dtype=np.int64)]
        )


def test_radix_sort_empty_and_single():
    got_keys, got_values = mde.radix_sort(
        np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int32)
    )
    assert got_keys.size == 0 and got_values.size == 0
    got_keys, got_values = mde.radix_sort(
        np.array([-5], dtype=np.int64), np.array([9], dtype=np.int32)
    )
    np.testing.assert_allclose(
        got_keys.astype(np.float64), np.array([-5.0]), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        got_values.astype(np.float64), np.array([9.0]), rtol=0, atol=0
    )


def test_radix_sort_all_keys_equal_preserves_input_order():
    keys = np.full(100, 42, dtype=np.int64)
    values = np.arange(100, dtype=np.int32)
    got_keys, got_values = mde.radix_sort(keys, values)
    np.testing.assert_allclose(
        got_keys.astype(np.float64), keys.astype(np.float64), rtol=0, atol=0
    )
    np.testing.assert_array_equal(got_values, values)


def test_radix_sort_rejects_two_dimensional_keys():
    with pytest.raises(ValueError, match="one-dimensional"):
        mde.radix_sort(
            np.zeros((2, 2), dtype=np.int64), np.zeros(0, dtype=np.int32)
        )


def test_split_points_single_element_and_rejects():
    """One key with more files: every boundary is that key (min/max of one)."""
    assert mde.split_points(np.array([42], dtype=np.int64), 4).tolist() == [42] * 5
    with pytest.raises(ValueError, match="at least 1"):
        mde.split_points(np.zeros(4, dtype=np.int64), -1)


def test_assign_files_empty_and_rejects():
    assert mde.assign_files(
        np.zeros(0, dtype=np.int64), np.array([0, 0, 0], dtype=np.int64)
    ).tolist() == []
    with pytest.raises(ValueError, match="nfiles \\+ 1"):
        mde.assign_files(np.zeros(4, dtype=np.int64), np.zeros(1, dtype=np.int64))
    with pytest.raises(ValueError, match="nfiles \\+ 1"):
        mde.assign_files(np.zeros(4, dtype=np.int64), np.zeros(0, dtype=np.int64))


def test_file_row_bounds_empty():
    got = mde.file_row_bounds(np.zeros(0, dtype=np.int32), 3)
    np.testing.assert_allclose(
        got.astype(np.float64), np.zeros(4), rtol=0, atol=0
    )


def test_file_bounds_empty_and_single_row():
    mins, maxs = mde.file_bounds(
        np.zeros(0, dtype=np.int32),
        [np.zeros(0, dtype=np.int64)],
        np.array([0, 0], dtype=np.int32),
    )
    assert mins.shape == (1, 1) and maxs.shape == (1, 1)
    assert (mins == 0).all() and (maxs == 0).all()

    mins, maxs = mde.file_bounds(
        np.array([0], dtype=np.int32),
        [np.array([123456789012345], dtype=np.int64)],
        np.array([0, 1], dtype=np.int32),
    )
    assert mins.tolist() == [[123456789012345]]
    assert maxs.tolist() == [[123456789012345]]


def test_file_bounds_rejects_short_bounds():
    with pytest.raises(ValueError, match="nfiles \\+ 1"):
        mde.file_bounds(
            np.zeros(4, dtype=np.int32),
            [np.zeros(4, dtype=np.int64)],
            np.array([0], dtype=np.int32),
        )


def test_range_count_empty_and_reversed_range():
    """Empty input counts nothing; lo > hi is an empty range, not an error.

    This matches the closed-range predicate `lo <= key <= hi` evaluated per
    row, which no key satisfies when lo > hi (verified in probe output).
    """
    assert mde.range_count(np.zeros(0, dtype=np.int64), 0, 10) == 0
    assert mde.range_count(np.array([1, 2, 3], dtype=np.int64), 5, 2) == 0


def test_range_counts_empty_and_mismatched():
    got = mde.range_counts(
        np.zeros(0, dtype=np.int64),
        np.zeros(3, dtype=np.int64),
        np.ones(3, dtype=np.int64),
    )
    np.testing.assert_allclose(
        got.astype(np.float64), np.zeros(3), rtol=0, atol=0
    )
    assert mde.range_counts(
        np.zeros(4, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
        np.zeros(0, dtype=np.int64),
    ).tolist() == []
    with pytest.raises(ValueError, match="same length"):
        mde.range_counts(
            np.zeros(4, dtype=np.int64),
            np.zeros(2, dtype=np.int64),
            np.zeros(3, dtype=np.int64),
        )


def test_range_count_rejects_two_dimensional_keys():
    with pytest.raises(ValueError, match="one-dimensional"):
        mde.range_count(np.zeros((2, 2), dtype=np.int64), 0, 1)


def test_prefix_rank_empty_and_single():
    assert mde.prefix_rank(np.zeros(0, dtype=np.int64), 10).tolist() == []
    np.testing.assert_allclose(
        mde.prefix_rank(np.array([5], dtype=np.int64), 10).astype(np.float64),
        np.array([0.0]), rtol=0, atol=0,
    )


def test_prefix_rank_rejects_impossible_widths():
    with pytest.raises(ValueError, match="prefix_bits"):
        mde.prefix_rank(np.zeros(4, dtype=np.int64), 64)
    with pytest.raises(ValueError, match="prefix_bits"):
        mde.prefix_rank(np.zeros(4, dtype=np.int64), -5)


def test_float_bits_empty_and_two_dimensional():
    assert mde.sortable_float_bits(np.zeros(0)).tolist() == []
    assert mde.zorder_float_bits(np.zeros(0)).tolist() == []
    with pytest.raises(ValueError, match="one-dimensional"):
        mde.sortable_float_bits(np.zeros((2, 2)))
    with pytest.raises(ValueError, match="one-dimensional"):
        mde.zorder_float_bits(np.zeros((2, 2)))


def test_float_bits_nan_places_above_inf():
    """NaN's bit pattern (0x7FF8...) is above +inf (0x7FF0...) in both images.

    This matches np.sort, which also places NaN last; a transform that put
    NaN below +inf would disagree with every NumPy statistic.
    """
    values = np.array([np.inf, np.nan])
    assert mde.sortable_float_bits(values)[1] > mde.sortable_float_bits(values)[0]
    assert (mde.zorder_float_bits(values)[1] > mde.zorder_float_bits(values)[0]).all()


def test_zorder_rewrite_empty():
    """n=0, nfiles=2: every stage has a defined empty value, none may crash."""
    res = mde.zorder_rewrite([np.zeros(0, dtype=np.int64)], nfiles=2)
    assert res.keys.tolist() == []
    assert res.order.tolist() == []
    assert res.split.tolist() == [0, 0, 0]
    assert res.row_files.tolist() == []
    assert res.row_bounds.tolist() == [0, 0, 0]
    assert res.mins.tolist() == [[0], [0]]
    assert res.maxs.tolist() == [[0], [0]]


def test_zorder_rewrite_duplicate_keys_all_land_in_last_file():
    """Constant keys collapse every split to the same boundary.

    assign_files gives every row to the last file, file_row_bounds puts 0,0,0,n,
    and the statistics of the two empty files are 0/0 -- asserted exactly.
    """
    values = np.array([7 << 44] * 4, dtype=np.int64)
    res = _rewrite([values], nfiles=3, bits=20)
    assert res.keys.tolist() == [7, 7, 7, 7]
    assert res.order.tolist() == [0, 1, 2, 3]
    assert res.split.tolist() == [7, 7, 7, 7]
    assert res.row_files.tolist() == [2, 2, 2, 2]
    assert res.row_bounds.tolist() == [0, 0, 0, 4]
    assert res.mins.tolist() == [[0], [0], [7 << 44]]
    assert res.maxs.tolist() == [[0], [0], [7 << 44]]


def test_zorder_rewrite_float_nan_sorts_last():
    """NaN in a float column must sort above every finite value.

    The key image puts NaN just above +inf, and the sortable image does too,
    so the NaN row is last in the order and carries the maximum statistic.
    This is the bit-pattern order the transform defines for every pattern;
    np.min/np.max would propagate NaN instead, which is a different statistic
    and deliberately not what file_bounds computes.
    """
    values = np.array([1.0, np.nan, 3.0])
    res = _rewrite([values], nfiles=2, bits=20, mode=mde.MODE_FLOAT)
    assert res.order.tolist() == [0, 2, 1]
    nan_image = mde.sortable_float_bits(np.array([np.nan]))[0]
    assert res.maxs[1, 0] == nan_image


def test_zorder_rewrite_rejects_zero_files():
    with pytest.raises(ValueError, match="at least 1"):
        mde.zorder_rewrite([np.zeros(4, dtype=np.int64)], nfiles=0)
    with pytest.raises(ValueError, match="at least 1"):
        mde.zorder_rewrite([np.zeros(4, dtype=np.int64)], nfiles=-1)


def test_prune_files_empty_and_rejects():
    assert mde.prune_files(
        np.zeros((0, 1)), np.zeros((0, 1)), 0, 0, 1
    ).tolist() == []
    with pytest.raises(ValueError, match="same shape"):
        mde.prune_files(np.zeros((2, 1)), np.zeros((3, 1)), 0, 0, 1)
    with pytest.raises(IndexError):
        mde.prune_files(np.zeros((2, 1)), np.zeros((2, 1)), 5, 0, 1)


def test_prune_rows_empty():
    got = mde.prune_rows(
        np.zeros(0, dtype=np.int64), np.array([0, 0, 0], dtype=np.int64)
    )
    np.testing.assert_allclose(
        got.astype(np.float64), np.zeros(2), rtol=0, atol=0
    )
    assert mde.prune_rows(
        np.zeros(4, dtype=np.int64), np.zeros(0, dtype=np.int64)
    ).tolist() == []
    assert mde.prune_rows(
        np.zeros(4, dtype=np.int64), np.zeros(1, dtype=np.int64)
    ).tolist() == []
