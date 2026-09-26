"""Column-list validation for OPTIMIZE Z-ORDER BY.

`delta-encoding` validates the Z-order column list in
`DeltaOptimizeBuilder.executeZOrderBy` and raises a `TypeError` naming the
offending value and its type. That check is the only part of `executeZOrderBy`
that lives in Python -- everything after it is a hand-off to the JVM builder --
so it is the one place this port can be a true parity test against the real
package.

The loop below is the upstream one, unchanged in behaviour: a single list or
tuple argument is unpacked into the column list, and every element must be a
`str`.
"""

from __future__ import annotations

from typing import Iterable, Union

ColumnSpec = Union[str, Iterable[str]]


def validate_zorder_columns(cols: tuple[ColumnSpec, ...]) -> list[str]:
    """Normalise and validate the `executeZOrderBy(*cols)` argument list.

    Mirrors `delta.tables.DeltaOptimizeBuilder.executeZOrderBy`: one list or
    tuple argument is unpacked, and each remaining element must be a `str`.
    """
    if len(cols) == 1 and isinstance(cols[0], (list, tuple)):
        cols = tuple(cols[0])
    for c in cols:
        if type(c) is not str:
            errorMsg = "Z-order column must be str. "
            errorMsg += "Found %s with type %s" % ((str(c)), str(type(c)))
            raise TypeError(errorMsg)
    return list(cols)
