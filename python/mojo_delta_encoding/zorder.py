"""The OPTIMIZE Z-ORDER BY pipeline, end to end.

`zorder_rewrite` is the whole numeric path a Z-ordering table writer runs: build
one Z-order key per row, sort by it, cut the sorted keys into files, and record
each file's per-column statistics so a reader can skip it. Each stage is also
available on its own so a caller can stop after the part it needs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import _lib


def _as_list(columns) -> list:
    if isinstance(columns, np.ndarray) and columns.ndim == 1:
        return [columns]
    return list(columns)


@dataclass(frozen=True)
class ZOrderResult:
    """Everything a Z-order rewrite produces.

    keys
        The Z-order key of every input row, in input order.
    order
        The permutation that puts rows in Z-order. `table[order]` is the
        rewritten table.
    split
        `nfiles + 1` key boundaries; file f owns `[split[f], split[f + 1])`,
        with the last file also taking the maximum key.
    row_files
        File index of each row in sorted order.
    row_bounds
        `nfiles + 1` row-index boundaries in sorted order, for `file_bounds`.
    mins, maxs
        `(nfiles, kcols)` per-file statistics, 0/0 for an empty file. A float
        column is measured in its `sortable_float_bits` image, so `mins[f, c]`
        is the transform of the smallest float in file f.
    """

    keys: np.ndarray
    order: np.ndarray
    split: np.ndarray
    row_files: np.ndarray
    row_bounds: np.ndarray
    mins: np.ndarray
    maxs: np.ndarray


def zorder_rewrite(
    columns,
    nfiles: int,
    bits: int = 20,
    mode: int = _lib.MODE_INT,
) -> ZOrderResult:
    """Cluster `columns` into `nfiles` Z-ordered files with per-file statistics.

    `columns` is a sequence of equal-length 1-D arrays, the Z-order columns in
    priority order. `bits` is how many high-order bits of each column feed the
    key; the key is int64, so `len(columns) * bits` must not exceed 63. Values
    below the retained width collapse onto the same key, which is the point of
    choosing `bits` for the data rather than for the type.
    """
    columns = _as_list(columns)
    keys = _lib.zorder_keys(columns, bits=bits, mode=mode)
    order = np.arange(keys.size, dtype=np.int32)
    sorted_keys, order = _lib.radix_sort(keys, order)
    split = _lib.split_points(sorted_keys, nfiles)
    row_files = _lib.assign_files(sorted_keys, split)
    row_bounds = _lib.file_row_bounds(row_files, nfiles)
    # Statistics are integer min/max, so a float column is measured in its
    # order-preserving image: the smallest transformed value in a file is the
    # transform of that file's smallest float, so the bounds still prune.
    if mode == _lib.MODE_FLOAT:
        stat_columns = [_lib.sortable_float_bits(c) for c in columns]
    else:
        stat_columns = columns
    mins, maxs = _lib.file_bounds(order, stat_columns, row_bounds)
    return ZOrderResult(keys, order, split, row_files, row_bounds, mins, maxs)


def prune_files(mins: np.ndarray, maxs: np.ndarray, column: int,
                lo: int, hi: int) -> np.ndarray:
    """File indices that a predicate `lo <= column <= hi` cannot rule out.

    This is the data-skipping decision a reader makes from the statistics
    `file_bounds` wrote: a file whose recorded range for `column` does not
    intersect the predicate can be skipped without being read. A file with no
    rows is never skipped, because an empty range must not look like a
    non-overlapping one.
    """
    mins = np.atleast_2d(np.asarray(mins))
    maxs = np.atleast_2d(np.asarray(maxs))
    if mins.shape != maxs.shape:
        raise ValueError("mins and maxs must have the same shape")
    empty = (mins[:, column] == 0) & (maxs[:, column] == 0)
    overlaps = (maxs[:, column] >= lo) & (mins[:, column] <= hi)
    return np.flatnonzero(overlaps | empty)


def prune_rows(keys, split: np.ndarray) -> np.ndarray:
    """Rows of a Z-ordered table inside each file's key range, as a count."""
    split = np.asarray(split, dtype=np.int64)
    return _lib.range_counts(keys, split[:-1], split[1:])
