# mojo-delta-encoding

`mojo-delta-encoding` implements the numeric core behind Delta Lake's
`OPTIMIZE Z-ORDER BY` and `OPTIMIZE` compaction in Mojo, behind a C ABI, and
exposes it from a Python package named `mojo_delta_encoding` so it installs
alongside the real `delta` package.

```python
import numpy as np
import mojo_delta_encoding as mde

a = np.random.default_rng(0).integers(0, 2**50, size=1_000_000)
b = a + np.random.default_rng(1).integers(-8, 9, size=1_000_000)

res = mde.zorder_rewrite([a, b], nfiles=32, bits=25)
table = np.stack([a, b], axis=1)[res.order]     # the rewritten, clustered table
res.mins, res.maxs                               # per-file statistics to skip with
mde.prune_files(res.mins, res.maxs, 0, 0, 2**40) # files a predicate must read
```

## What this is, and what it is not

**The real `delta-encoding` package has no numeric core in Python.** Its whole
public surface is Spark plumbing: `DeltaTable`, `DeltaTableBuilder`,
`DeltaMergeBuilder` and `DeltaOptimizeBuilder` build and submit plans, and every
numeric decision is made by the JVM. Reading `delta/tables.py` and
`delta/connect/tables.py` end to end, there is not one loop over an array or
over a byte buffer: `DeltaOptimizeBuilder.executeZOrderBy` validates that its
column list is a list of `str` and then hands it to a Java builder, and
`executeCompaction` does not even read its arguments. Grepping the package for
`import numpy`, `for ... in range(`, `int.from_bytes` and `struct.pack` returns
only generated protobuf stubs.

So there are two honest things to do, and this port does both:

1. **Port the algorithms the package names.** The Z-order curve, the sort by
   Z-key, the file split, the per-file column statistics, the data-skipping
   range count and the compaction planner are all pure integer and `float64`
   work with no Spark dependency at all. They live in `delta-core` in Scala;
   here they are one Mojo compilation unit. This is the substantive part of the
   port and the reason the package exists.
2. **Parity-test the part that really is Python.** `validate_zorder_columns`
   reproduces the upstream column check, and the tests call the real
   `delta.tables.DeltaOptimizeBuilder.executeZOrderBy` and require the same
   exception class and the same message, character for character.

Nothing here is presented as a port of a Python function that does not exist.

## Covered subset

| area | implemented API | kernel |
| --- | --- | --- |
| Z-order curve | `zorder_keys`, `zorder_rewrite` | `de_zorder_keys` |
| Sorting by Z-key | `radix_sort` | `de_radix_sort_u64` |
| File split | `split_points`, `assign_files`, `file_row_bounds` | `de_split_points`, `de_row_files` |
| Data-skipping statistics | `file_bounds` | `de_file_bounds` |
| Row-level data skipping | `range_count`, `range_counts`, `prune_rows` | `de_range_count`, `de_range_counts` |
| File-level data skipping | `prune_files` | (consumes `file_bounds`) |
| Bucket occupancy | `prefix_rank` | `de_prefix_rank` |
| Compaction planning | `compaction_groups` | `de_compaction_groups` |
| Float ordering | `zorder_float_bits`, `sortable_float_bits` | (inside `de_zorder_keys` mode 1) |
| Column validation | `validate_zorder_columns` | (parity with the real `delta`) |

### Not implemented

* The Spark surface itself: `DeltaTable`, `DeltaTableBuilder`,
  `DeltaMergeBuilder`, `DeltaTable.forPath` / `forName`, Spark Connect plans,
  `ConvertToDelta`, `RestoreTable`, `CloneTable`, `Generate`, `vacuum`. These are
  session plumbing with no numeric content; use the real `delta` package.
* Reading or writing the Delta log, Parquet files, or transaction state.
* Delta's *candidate selection* heuristics: `AutoCompact`'s
  `maxFileSize` / `minFileSize` policy, `DeltaOptimize`'s rewrites-per-file
  scoring, and the closest-vector search that picks a Z-order column set. Those
  are JVM heuristics over catalog state, not kernels, and this port does not
  pretend to have them.
* Multi-column Z-order beyond 63 bits of key. `kcols * bits` must fit in the
  `int64` key; the shim rejects anything wider rather than silently wrapping.
* Deletion vectors, bloom filters, and `OPTIMIZE WHERE` predicate pushdown.

## Contracts worth knowing before you use it

**The key keeps the high bits.** `zorder_keys` retains the top `bits` bits of
each column, not the low bits, so a column whose magnitudes are far below `2**(64-bits)`
collapses onto key 0 and Z-ordering does nothing. Choose `bits` for the data.
In integer mode the key is non-negative; in float mode the retained window is
read unsigned, and the key transform is the unsigned-monotone one so that a
negative float still sorts below a positive one.

**Float statistics use a different transform from float keys, on purpose.**
`zorder_float_bits` is monotone as an unsigned 64-bit value and is what the key
is built from. `sortable_float_bits` is monotone under *signed* comparison, which
is what `np.minimum` and integer `min`/`max` do, so it is what the per-file
statistics use. The two induce the same total order, and a test asserts it: if
they ever disagreed, a file's recorded bounds could prune away rows that are
really there.

**`file_bounds` for a float column is in the transform's image.** `mins` and
`maxs` hold `sortable_float_bits` of the smallest and largest float in the file.
Convert a predicate bound with `sortable_float_bits` before comparing, or the
sign of the comparison is wrong.

**Range bounds are closed.** `range_count` and `range_counts` count
`lo <= key <= hi`. That avoids the overflow an exclusive upper bound hits at
`INT64_MAX`, at the cost of counting a row sitting exactly on an interior
boundary in both of its neighbours. `prune_rows` documents this.

**A file with no rows has bounds 0/0, and `prune_files` keeps it.** An empty
range must not read as a non-overlapping one, or the caller would "skip" a file
that a concurrent writer might have just filled.

## Install

The repository pins its own Mojo toolchain:

```bash
pixi install
pixi run build
pixi run test
```

`pixi run build` produces `dist/libmojo-delta-encoding.so`. Set
`PYTHONPATH=python` when using the package outside a Pixi task. To build and
test against the shared toolchain instead:

```bash
bash build/build.sh
PYTHONPATH=python python -m pytest tests -q
```

## Performance

Best of three runs per case, on a shared 36-core box, against the fastest
reasonable NumPy formulation of each step: a vectorised bit-interleave for the
curve, `np.argsort(kind="stable")` for the sort, `np.minimum.reduceat` for the
grouped statistics, a broadcast comparison for the range counts, and a run-length
encode for the prefix rank. Every case verifies its result against that
reference before timing.

| case | reference | mojo-delta-encoding | result |
| --- | ---: | ---: | ---: |
| zorder_keys n=2097152 k=3 b=20 | 1210.93 ms | 280.76 ms | 4.31x faster |
| radix_sort n=2097152 | 468.93 ms | 647.48 ms | **0.72x, slower** |
| file_bounds n=1048576 k=3 f=64 | 104.19 ms | 59.27 ms | 1.76x faster |
| range_counts n=1048576 f=128 | 460.93 ms | 273.23 ms | 1.69x faster |
| compaction_groups n=65536 | 42.87 ms | 22.24 ms | 1.93x faster |
| prefix_rank n=4194304 b=30 | 237.24 ms | 25.06 ms | 9.47x faster |
| zorder_rewrite n=1048576 f=32 | 658.32 ms | 315.94 ms | 2.08x faster |

The radix sort is a genuine loss and is reported as one. NumPy's stable integer
sort is itself a radix sort and its digit loop vectorises, while this kernel
moves 2M keys and 2M payloads through memory eight times; at 8 passes it is
bandwidth-bound and the vectorised version wins. It is still the right kernel
here because the pipeline needs the payload permutation and because its
worst case does not degrade, but it is not a speedup.

`prefix_rank` wins by the largest margin because the NumPy run-length encode
needs a `searchsorted` per row while the kernel walks the sorted array once.
The end-to-end rewrite wins because the curve is the dominant cost and NumPy
needs one full-size temporary per bit.

The benchmark reports the best of three runs and prints the observed spread when
it is wide; this box is shared, and contention moves a whole case by 3x rather
than shifting everything uniformly.

Reproduce with:

```bash
pixi run bench
```

## How it works

All kernels live in `src/kernels.mojo`, one compilation unit, because shared
library build cost is largely fixed. `build/build.sh` compiles it with
`mojo build --emit shared-lib` into `dist/libmojo-delta-encoding.so`.

The `python/mojo_delta_encoding` layer owns every array. It normalises columns to
contiguous `float64` or `int64`, allocates the output and the radix-sort
scratch, and copies the results out, so a caller can pass views and strided
slices without thinking about the Mojo side. Buffers cross the C ABI as 64-bit
addresses and are reconstructed in Mojo as
`Pointer[Int64, AnyOrigin[mut=True]]`, which keeps the exported symbols
non-parametric.

No kernel allocates. The radix sort takes its scratch buffers from the caller,
and the eight passes ping-pong between the caller's arrays and the scratch an
even number of times, so the sorted result lands back where the caller can see
it. The top byte pass flips the digit's high bit, which is what makes the sort
signed: without it `INT64_MIN` lands at the top of the list.

Every kernel here is a memory-bound or bit-manipulation loop, so each is a plain
serial loop. Threading was not used: 1.2.0 cannot carry pointers into a
`parallelize` body, and a bandwidth-bound loop does not benefit from threads
anyway.

## Tests

`tests/test_zorder.py` covers the curve, the sort, the file split, the
statistics, the skipping and the pipeline. `tests/test_compaction.py` covers
compaction planning and the parity tests against the real `delta` package.

Every assertion is exact (`rtol=0, atol=0`): the Z-order key, the sort, the
bounds and the counts are integer and byte-level work with no floating-point
arithmetic to lose bits to, and a wrong stride, a dropped sort pass or an
off-by-one in a file boundary should be a hard failure rather than a tolerance.
The tests that would catch a plausible bug include the hand-written 4x4 Morton
grid, the single-bit column-stride layout, a stability fixture with duplicates,
`INT64_MIN` and `INT64_MAX` in the sort, a partition check on every compaction
grouping, and an end-to-end test that Z-ordering actually lets a reader skip
files compared against a row-range layout with the same file count.

## License

MIT
