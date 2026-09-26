"""ctypes bridge to the compiled Mojo kernels.

The shared library owns no memory. Every buffer crosses the C ABI as a 64-bit
address, so the argtypes below must stay `c_int64` for addresses; `c_int`
truncates them and segfaults.

Every entry point here allocates and owns its own buffers and returns fresh
arrays, so callers never have to think about the Mojo side's memory.

Row-major layout is shared by every kernel: `kcols` rows of `n` values, so
column c of row i lives at flat index `c * n + i`.
"""

from __future__ import annotations

import ctypes
import pathlib

import numpy as np

_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[2]
_LIB_PATH = _ROOT / "dist" / "libmojo-delta-encoding.so"

#: Read Z-order columns as signed integers.
MODE_INT = 0
#: Read Z-order columns as float64 bit patterns, order-preserving first.
MODE_FLOAT = 1

#: A Z-order key is int64, so all columns' bit widths must fit in 63 bits.
MAX_ZORDER_BITS = 63

_i64p = ctypes.c_int64
_dbl = ctypes.c_double


def _load():
    if not _LIB_PATH.exists():
        raise RuntimeError(
            f"{_LIB_PATH} not found; run `bash build/build.sh` first"
        )
    lib = ctypes.CDLL(str(_LIB_PATH))

    def decl(name, *argtypes):
        fn = getattr(lib, name)
        fn.restype = None
        fn.argtypes = list(argtypes)

    decl("de_zorder_keys", _i64p, _i64p, _i64p, _i64p, _i64p, _i64p)
    decl("de_radix_sort_u64", _i64p, _i64p, _i64p, _i64p, _i64p, _i64p)
    decl("de_split_points", _i64p, _i64p, _i64p, _i64p)
    decl("de_row_files", _i64p, _i64p, _i64p, _i64p, _i64p)
    decl("de_file_bounds", _i64p, _i64p, _i64p, _i64p, _i64p, _i64p, _i64p,
         _i64p)
    decl("de_range_count", _i64p, _i64p, _i64p, _i64p, _i64p)
    decl("de_range_counts", _i64p, _i64p, _i64p, _i64p, _i64p, _i64p)
    decl("de_compaction_groups", _i64p, _i64p, _dbl, _i64p, _i64p, _i64p)
    decl("de_prefix_rank", _i64p, _i64p, _i64p, _i64p)
    return lib


lib = _load()


def _addr(a: np.ndarray) -> int:
    if not a.flags["C_CONTIGUOUS"]:
        raise ValueError("buffer must be C-contiguous")
    return a.ctypes.data


def _i64(a) -> np.ndarray:
    out = np.ascontiguousarray(a, dtype=np.int64)
    if out.ndim != 1:
        raise ValueError("expected a one-dimensional array")
    return out


def _f64(a) -> np.ndarray:
    out = np.ascontiguousarray(a, dtype=np.float64)
    if out.ndim != 1:
        raise ValueError("expected a one-dimensional array")
    return out


def _i32(a) -> np.ndarray:
    out = np.ascontiguousarray(a, dtype=np.int32)
    if out.ndim != 1:
        raise ValueError("expected a one-dimensional array")
    return out


def _check_key_shape(kcols: int, bits: int, mode: int) -> None:
    if mode not in (MODE_INT, MODE_FLOAT):
        raise ValueError("mode must be MODE_INT or MODE_FLOAT")
    if not 1 <= bits <= MAX_ZORDER_BITS:
        raise ValueError(f"bits must be in 1..{MAX_ZORDER_BITS}")
    if kcols * bits > MAX_ZORDER_BITS:
        raise ValueError(
            f"{kcols} columns x {bits} bits exceeds the "
            f"{MAX_ZORDER_BITS}-bit Z-order key"
        )


def _stack(columns, bits: int, mode: int) -> tuple[np.ndarray, int, int]:
    """Normalise a column sequence to one row-major int64 buffer."""
    if isinstance(columns, np.ndarray) and columns.ndim == 1:
        columns = [columns]
    cols = list(columns)
    if not cols:
        raise ValueError("at least one Z-order column is required")
    prepared = []
    n = None
    for c in cols:
        c = _f64(c).view(np.int64) if mode == MODE_FLOAT else _i64(c)
        if n is None:
            n = c.size
        elif c.size != n:
            raise ValueError("all Z-order columns must have the same length")
        prepared.append(c)
    _check_key_shape(len(prepared), bits, mode)
    return np.ascontiguousarray(np.stack(prepared)), len(prepared), n or 0


def sortable_float_bits(values: np.ndarray) -> np.ndarray:
    """The signed `int64` order-preserving image of a float64 array.

    This is the transform for *statistics*: `u ^ ((u >> 63) & 0x7FFF_...)` maps
    a negative double to `~u` with the sign bit set and leaves a non-negative one
    alone, so plain signed comparison reproduces float comparison. That is what
    makes per-file statistics on a float column computable with integer min and
    max -- the smallest transformed value in a file is the transform of that
    file's smallest float.

    It is deliberately *not* the transform `zorder_keys` uses in MODE_FLOAT; that
    one, `zorder_float_bits`, is monotone as an unsigned value instead, because
    the retained key window is read unsigned. The two induce the same total
    order on finite doubles, so the key and the per-file bounds never disagree
    about which row is smaller.
    """
    bits = _f64(values).view(np.int64)
    return bits ^ ((bits >> 63) & np.int64(0x7FFFFFFFFFFFFFFF))


def zorder_float_bits(values: np.ndarray) -> np.ndarray:
    """The unsigned order-preserving image a float64 column takes in a key.

    `u ^ ((u >> 63) | 1 << 63)`: a negative double maps to `~u`, a
    non-negative one to `u | 1 << 63`, so the result is increasing as an
    *unsigned* 64-bit value across the whole range including the sign boundary.
    `zorder_keys(..., mode=MODE_FLOAT)` applies exactly this before keeping the
    top `bits` bits. For min and max use `sortable_float_bits` instead.
    """
    bits = _f64(values).view(np.int64)
    return (bits ^ ((bits >> 63) | np.int64(-(2**63)))).view(np.uint64)


def zorder_keys(columns, bits: int = 20, mode: int = MODE_INT) -> np.ndarray:
    """Interleave the top `bits` of each column into one Z-order key per row.

    `columns` is a sequence of equal-length 1-D arrays. MODE_INT reads them as
    signed integers; MODE_FLOAT reads them as float64 and applies
    `zorder_float_bits` first, so keys ascend with the data.
    """
    buf, kcols, n = _stack(columns, bits, mode)
    out = np.empty(n, dtype=np.int64)
    lib.de_zorder_keys(_addr(buf), n, kcols, bits, mode, _addr(out))
    return out


def radix_sort(keys, values) -> tuple[np.ndarray, np.ndarray]:
    """Stable sort `int64` keys, permuting the int32 `values` array with them.

    Returns fresh `(keys, values)` arrays. The sort is an LSD radix sort over
    the 8 bytes of the key, so equal keys keep their input order.
    """
    keys = _i64(keys).copy()
    values = _i32(values).copy()
    if keys.size != values.size:
        raise ValueError("keys and values must have the same length")
    n = keys.size
    skey = np.empty(n, dtype=np.int64)
    sval = np.empty(n, dtype=np.int32)
    cnt = np.empty(256, dtype=np.int64)
    lib.de_radix_sort_u64(
        _addr(keys), _addr(values), _addr(skey), _addr(sval), _addr(cnt), n
    )
    return keys, values


def split_points(keys, nfiles: int) -> np.ndarray:
    """`nfiles + 1` monotone key boundaries that cut sorted `keys` evenly.

    Boundary `f` is the key at index `f * n // nfiles`, so every file holds
    either `n // nfiles` or one more row. Duplicated keys can collapse a range
    to nothing, which is correct.
    """
    keys = _i64(keys)
    if nfiles < 1:
        raise ValueError("nfiles must be at least 1")
    out = np.empty(nfiles + 1, dtype=np.int64)
    lib.de_split_points(_addr(keys), keys.size, nfiles, _addr(out))
    return out


def assign_files(sorted_keys, split) -> np.ndarray:
    """File index of each sorted key, given the `nfiles + 1` split boundaries.

    File f owns the key range `[split[f], split[f + 1])`, except the last file,
    which also takes the maximum key, so every row lands in exactly one file.
    """
    sorted_keys = _i64(sorted_keys)
    split = _i64(split)
    nfiles = split.size - 1
    if nfiles < 1:
        raise ValueError("split must hold nfiles + 1 boundaries")
    out = np.empty(sorted_keys.size, dtype=np.int32)
    lib.de_row_files(_addr(sorted_keys), sorted_keys.size, _addr(split), nfiles,
                     _addr(out))
    return out


def file_row_bounds(row_files, nfiles: int) -> np.ndarray:
    """`nfiles + 1` row-index boundaries implied by `assign_files`.

    This is the exact complement of the key-range assignment: the two agree even
    when the Z-order key has duplicates, which is why `file_bounds` is driven by
    this and not by recomputing an index split, and it stays monotone even when
    a trailing file ends up empty.
    """
    row_files = _i32(row_files)
    return np.searchsorted(
        row_files, np.arange(nfiles + 1), side="left"
    ).astype(np.int32)


def file_bounds(order, columns, row_bounds) -> tuple[np.ndarray, np.ndarray]:
    """Per-file, per-column `(min, max)` after a Z-order rewrite.

    `order` is the sorted row permutation, `columns` the columns the keys came
    from, and `row_bounds` the `nfiles + 1` row-index boundaries from
    `file_row_bounds`. For a float column pass it through
    `sortable_float_bits` first: min and max in that integer image are the
    transformed min and max of the float values, so the bounds still prune
    correctly. A file with no rows gets 0/0.
    """
    order = _i32(order)
    bounds = _i32(row_bounds)
    nfiles = bounds.size - 1
    if nfiles < 1:
        raise ValueError("row_bounds must hold nfiles + 1 boundaries")
    if bounds[0] != 0 or bounds[-1] != order.size:
        raise ValueError("row_bounds must span every row of order")
    buf, kcols, n = _stack(columns, 1, MODE_INT)
    if order.size != n:
        raise ValueError("order and columns must have the same row count")
    omin = np.zeros(nfiles * kcols, dtype=np.int64)
    omax = np.zeros(nfiles * kcols, dtype=np.int64)
    lib.de_file_bounds(_addr(order), _addr(buf), _addr(bounds), n, kcols,
                       nfiles, _addr(omin), _addr(omax))
    shape = (nfiles, kcols)
    return omin.reshape(shape), omax.reshape(shape)


def range_count(keys, lo: int, hi: int) -> int:
    """Rows whose Z-order key lies in the closed range `[lo, hi]`."""
    keys = _i64(keys)
    out = np.empty(1, dtype=np.int64)
    lib.de_range_count(_addr(keys), keys.size, int(lo), int(hi), _addr(out))
    return int(out[0])


def range_counts(keys, los, his) -> np.ndarray:
    """Rows per closed key range, one count per `(los[f], his[f])` pair."""
    keys = _i64(keys)
    los = _i64(los)
    his = _i64(his)
    if los.size != his.size:
        raise ValueError("los and his must have the same length")
    out = np.empty(los.size, dtype=np.int64)
    lib.de_range_counts(_addr(keys), keys.size, _addr(los), _addr(his),
                        los.size, _addr(out))
    return out


def compaction_groups(sizes, min_size: float) -> list[tuple[int, int]]:
    """Group files for OPTIMIZE compaction as a list of `(start, length)`.

    A file at or above `min_size` is left alone and becomes its own group; a run
    of smaller files accumulates until its total reaches `min_size` and then
    closes as the rewrite unit; a trailing run of small files becomes one final
    group.
    """
    sizes = _f64(sizes)
    n = sizes.size
    start = np.empty(n, dtype=np.int32)
    length = np.empty(n, dtype=np.int32)
    ngroups = np.empty(1, dtype=np.int64)
    lib.de_compaction_groups(
        _addr(sizes), n, _dbl(min_size), _addr(start), _addr(length),
        _addr(ngroups)
    )
    return [(int(start[i]), int(length[i])) for i in range(int(ngroups[0]))]


def prefix_rank(sorted_keys, prefix_bits: int) -> np.ndarray:
    """For each sorted key, how many earlier keys share its top `prefix_bits`."""
    sorted_keys = _i64(sorted_keys)
    if not 1 <= prefix_bits <= MAX_ZORDER_BITS:
        raise ValueError(f"prefix_bits must be in 1..{MAX_ZORDER_BITS}")
    out = np.empty(sorted_keys.size, dtype=np.int64)
    lib.de_prefix_rank(_addr(sorted_keys), sorted_keys.size, int(prefix_bits),
                       _addr(out))
    return out
