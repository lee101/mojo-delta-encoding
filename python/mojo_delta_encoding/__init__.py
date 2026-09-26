"""mojo-delta-encoding: Z-order curves, data-skipping statistics and compaction
planning with Mojo kernels.

Installable alongside the real `delta` package, which it is tested against for
parity where the real package has Python-side behaviour to compare.

The Python surface of `delta-encoding` is Spark plumbing: it forwards column
lists to a JVM builder and has no numeric core of its own. This package
implements the Z-order and compaction algorithms that `executeZOrderBy` and
`executeCompaction` name, and the README says exactly which is which.
"""

from ._lib import (
    MAX_ZORDER_BITS,
    MODE_FLOAT,
    MODE_INT,
    assign_files,
    compaction_groups,
    file_bounds,
    file_row_bounds,
    prefix_rank,
    radix_sort,
    range_count,
    range_counts,
    sortable_float_bits,
    split_points,
    zorder_float_bits,
    zorder_keys,
)
from .columns import validate_zorder_columns
from .zorder import ZOrderResult, prune_files, prune_rows, zorder_rewrite

__all__ = [
    "MAX_ZORDER_BITS",
    "MODE_FLOAT",
    "MODE_INT",
    "ZOrderResult",
    "assign_files",
    "compaction_groups",
    "file_bounds",
    "file_row_bounds",
    "prefix_rank",
    "prune_files",
    "prune_rows",
    "radix_sort",
    "range_count",
    "range_counts",
    "sortable_float_bits",
    "split_points",
    "validate_zorder_columns",
    "zorder_float_bits",
    "zorder_keys",
    "zorder_rewrite",
]
__version__ = "0.1.0"
