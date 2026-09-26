"""Correctness-gated benchmark for mojo-delta-encoding.

Every case checks its result against an independent NumPy formulation before
timing, so a regression in the Mojo kernels shows up as a correctness failure
rather than as a suspiciously good number.

The baselines are the fastest reasonable NumPy formulations, not Python loops:
a bit-interleave written as 20 vectorised passes, `argsort(kind="stable")` for
the sort, `np.minimum.reduceat` for the per-file statistics, and a broadcast
comparison for the range counts. A Python loop would be a strawman, and the point
of the comparison is what a reader could realistically do without a compiler.
"""

from __future__ import annotations

import gc
import pathlib
import sys
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "python"))

import mojo_delta_encoding as mde  # noqa: E402
from mojo_delta_encoding import _lib  # noqa: E402


def _time(fn, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def _ref_zorder(cols, bits):
    """NumPy bit-interleave: one vectorised pass per bit."""
    n = cols[0].size
    k = len(cols)
    key = np.zeros(n, dtype=np.int64)
    for c in range(k):
        q = cols[c] >> np.int64(64 - bits)
        for b in range(bits):
            key |= ((q >> np.int64(b)) & np.int64(1)) << np.int64(b * k + c)
    return key


def bench_zorder_keys(n: int = 1 << 21, kcols: int = 3, bits: int = 20):
    """The Z-order curve: Mojo's per-row bit loop against a 60-pass NumPy
    interleave, which needs 60 full-size temporaries."""
    rng = np.random.default_rng(0)
    cols = [rng.integers(-(2**62), 2**62, size=n, dtype=np.int64)
            for _ in range(kcols)]
    want = _ref_zorder(cols, bits)
    got = mde.zorder_keys(cols, bits=bits)
    assert np.array_equal(got, want), "zorder_keys mismatch"

    numpy_time = _time(lambda: _ref_zorder(cols, bits), 3)
    mojo_time = _time(lambda: mde.zorder_keys(cols, bits=bits))
    return f"zorder_keys n={n} k={kcols} b={bits}", numpy_time, mojo_time


def bench_radix_sort(n: int = 1 << 21):
    """Sorting the keys: an 8-pass stable radix sort against NumPy's stable sort.
    NumPy uses introsort for int64, which is a comparison sort with an index
    array, so this is the kernel's real competitor."""
    rng = np.random.default_rng(1)
    keys = rng.integers(-(2**62), 2**62, size=n, dtype=np.int64)
    values = np.arange(n, dtype=np.int32)
    want = np.argsort(keys, kind="stable")
    got_keys, got_values = mde.radix_sort(keys, values)
    assert np.array_equal(got_keys, keys[want]), "radix sort keys mismatch"
    assert np.array_equal(got_values, values[want]), "radix sort payload mismatch"

    numpy_time = _time(lambda: np.argsort(keys, kind="stable"), 3)
    mojo_time = _time(lambda: mde.radix_sort(keys, values))
    return f"radix_sort n={n}", numpy_time, mojo_time


def bench_file_bounds(n: int = 1 << 20, kcols: int = 3, nfiles: int = 64):
    """Per-file min and max: a sequential kernel against `reduceat`, which is the
    fastest way to get grouped reductions out of NumPy."""
    rng = np.random.default_rng(2)
    cols = [rng.integers(-(2**40), 2**40, size=n, dtype=np.int64)
            for _ in range(kcols)]
    order = rng.permutation(n).astype(np.int32)
    edges = np.array([f * n // nfiles for f in range(nfiles + 1)], dtype=np.int32)

    def numpy_bounds():
        return [
            (np.minimum.reduceat(col[order.astype(np.int64)], edges[:-1]),
             np.maximum.reduceat(col[order.astype(np.int64)], edges[:-1]))
            for col in cols
        ]

    mins, maxs = mde.file_bounds(order, cols, edges)
    for c, col in enumerate(cols):
        permuted = col[order.astype(np.int64)]
        assert np.array_equal(mins[:, c], np.minimum.reduceat(permuted,
                                                             edges[:-1]))
        assert np.array_equal(maxs[:, c], np.maximum.reduceat(permuted,
                                                             edges[:-1]))

    numpy_time = _time(numpy_bounds, 3)
    mojo_time = _time(lambda: mde.file_bounds(order, cols, edges))
    return (f"file_bounds n={n} k={kcols} f={nfiles}", numpy_time, mojo_time)


def bench_range_counts(n: int = 1 << 20, nfiles: int = 128):
    """Data skipping: `nfiles` closed key ranges, Mojo against a broadcast
    comparison and a sum, which is what a reader would write in NumPy."""
    rng = np.random.default_rng(3)
    keys = np.sort(rng.integers(0, 2**40, size=n, dtype=np.int64))
    los = np.sort(rng.integers(0, 2**40, size=nfiles, dtype=np.int64))
    his = los + rng.integers(0, 2**20, size=nfiles, dtype=np.int64)

    def numpy_counts():
        return ((keys[None, :] >= los[:, None]) & (keys[None, :] <= his[:, None])
                ).sum(axis=1)

    got = mde.range_counts(keys, los, his)
    assert np.array_equal(got, numpy_counts()), "range_counts mismatch"

    numpy_time = _time(numpy_counts, 3)
    mojo_time = _time(lambda: mde.range_counts(keys, los, his))
    return f"range_counts n={n} f={nfiles}", numpy_time, mojo_time


def bench_compaction_groups(nfiles: int = 1 << 16):
    """Compaction planning: the kernel against the obvious Python sweep, which is
    what a caller has today. NumPy has no grouped-prefix-sum here, so the
    honest baseline is a single pass that closes each group as it goes."""
    rng = np.random.default_rng(4)
    sizes = rng.gamma(2.0, 1e6, size=nfiles)
    min_size = 4e6

    def python_groups():
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
        if first < nfiles:
            groups.append((first, nfiles - first))
        return groups

    got = mde.compaction_groups(sizes, min_size)
    assert got == python_groups(), "compaction grouping mismatch"

    numpy_time = _time(python_groups, 3)
    mojo_time = _time(lambda: mde.compaction_groups(sizes, min_size))
    return f"compaction_groups n={nfiles}", numpy_time, mojo_time


def bench_rewrite(n: int = 1 << 20, nfiles: int = 32):
    """The whole OPTIMIZE Z-ORDER BY pipeline against the same five steps in
    NumPy. This is the number a table writer would actually feel."""
    rng = np.random.default_rng(5)
    cols = [rng.integers(0, 2**50, size=n, dtype=np.int64) for _ in range(2)]
    bits = 25

    def numpy_rewrite():
        keys = _ref_zorder(cols, bits)
        order = np.argsort(keys, kind="stable")
        split = np.array([keys[order[min(f * n // nfiles, n - 1)]]
                          for f in range(nfiles + 1)], dtype=np.int64)
        bounds = np.array([f * n // nfiles for f in range(nfiles + 1)],
                          dtype=np.int32)
        mins = np.stack([np.minimum.reduceat(c[order], bounds[:-1]) for c in cols],
                        axis=1)
        maxs = np.stack([np.maximum.reduceat(c[order], bounds[:-1]) for c in cols],
                        axis=1)
        return keys, order, split, mins, maxs

    keys, order, split, mins, maxs = numpy_rewrite()
    res = mde.zorder_rewrite(cols, nfiles=nfiles, bits=bits)
    assert np.array_equal(res.keys, keys), "rewrite keys mismatch"
    assert np.array_equal(res.order, order), "rewrite order mismatch"
    assert np.array_equal(res.split, split), "rewrite split mismatch"
    assert np.array_equal(res.mins, mins), "rewrite mins mismatch"
    assert np.array_equal(res.maxs, maxs), "rewrite maxs mismatch"

    numpy_time = _time(numpy_rewrite, 3)
    mojo_time = _time(lambda: mde.zorder_rewrite(cols, nfiles=nfiles, bits=bits))
    return f"zorder_rewrite n={n} f={nfiles}", numpy_time, mojo_time


    keys, order, split, mins, maxs = numpy_rewrite()
    res = mde.zorder_rewrite(cols, nfiles=nfiles, bits=bits)
    assert np.array_equal(res.keys, keys), "rewrite keys mismatch"
    assert np.array_equal(res.order, order), "rewrite order mismatch"
    assert np.array_equal(res.split, split), "rewrite split mismatch"
    assert np.array_equal(res.mins, mins), "rewrite mins mismatch"
    assert np.array_equal(res.maxs, maxs), "rewrite maxs mismatch"

    # Drop the reference arrays before timing. Holding a dozen live 8 MB buffers
    # through both timed runs measures the allocator, not either pipeline.
    del keys, order, split, mins, maxs, res
    gc.collect()

    numpy_time = _time(numpy_rewrite, 3)
    gc.collect()
    mojo_time = _time(lambda: mde.zorder_rewrite(cols, nfiles=nfiles, bits=bits), 3)
    return f"zorder_rewrite n={n} f={nfiles}", numpy_time, mojo_time


def bench_prefix_rank(n: int = 1 << 22, prefix_bits: int = 30):
    """Bucket occupancy: the kernel against a NumPy run-length encode."""
    rng = np.random.default_rng(6)
    keys = np.sort(rng.integers(0, 2**40, size=n, dtype=np.int64))

    def numpy_rank():
        prefixes = keys >> np.int64(64 - prefix_bits)
        edges = np.flatnonzero(np.diff(prefixes)) + 1
        starts = np.concatenate(([0], edges))
        return np.arange(n, dtype=np.int64) - starts[
            np.searchsorted(starts, np.arange(n), side="right") - 1
        ]

    got = mde.prefix_rank(keys, prefix_bits)
    assert np.array_equal(got, numpy_rank()), "prefix_rank mismatch"

    numpy_time = _time(numpy_rank, 3)
    mojo_time = _time(lambda: mde.prefix_rank(keys, prefix_bits))
    return f"prefix_rank n={n} b={prefix_bits}", numpy_time, mojo_time


def main(samples: int = 3):
    """Report the best of `samples` runs per case.

    This box is shared, so a single reading is worth little: contention shows up
    as a whole case being 3x slower, not as a uniform slowdown. Taking the best
    of several independent runs and printing the spread keeps a noisy neighbour
    from being mistaken for a result.
    """
    cases = (
        bench_zorder_keys,
        bench_radix_sort,
        bench_file_bounds,
        bench_range_counts,
        bench_compaction_groups,
        bench_prefix_rank,
        bench_rewrite,
    )
    print(f"{'case':<34}{'reference':>12}{'mojo-delta-encoding':>21}{'ratio':>10}")
    print("-" * 77)
    for fn in cases:
        refs, gots, label = [], [], None
        for _ in range(samples):
            label, ref, got = fn()
            refs.append(ref)
            gots.append(got)
        ref, got = min(refs), min(gots)
        ratio = ref / got if got else float("nan")
        verdict = f"{ratio:.2f}x" if ratio >= 1 else f"{ratio:.2f}x SLOWER"
        spread = ""
        if max(refs) > 1.5 * min(refs) or max(gots) > 1.5 * min(gots):
            spread = f"   (spread ref {min(refs)*1e3:.0f}-{max(refs)*1e3:.0f}ms," \
                     f" mojo {min(gots)*1e3:.0f}-{max(gots)*1e3:.0f}ms)"
        print(f"{label:<34}{ref*1e3:>10.2f}ms{got*1e3:>19.2f}ms{verdict:>10}{spread}")


if __name__ == "__main__":
    main()
