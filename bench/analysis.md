# ANALYZE — mojo-delta-encoding hotspot map

Measured, not estimated. Every time below comes from a run in this session
(best of 5–7 repeats, `time.perf_counter`, single-threaded caller).
`pixi run bench` (flock-wrapped, `flock /tmp/mojo-bench.lock python bench/bench.py`)
was run 4 times and used for the headline comparison; the per-function
breakdown (wrapper vs. direct `lib.*` call) was timed in separate `pixi run python`
harnesses against the same sizes. Machine: Intel Xeon E5-2697 v4 (72 vCPU,
x86-64, AVX2), NumPy 2.5.1, Python 3.13, Mojo `1.2.0.dev2026092905`.
This box is shared (per `bench/bench.py`); absolute numbers wobble run to run,
so ratios and the wrapper-vs.-kernel split are the evidence. No algorithm was
changed (nothing in `src/` or `tests/` was touched).

## Upstream `delta-encoding` — not importable here

`importlib.util.find_spec("delta")` is `None` and `pip show delta-encoding`
finds nothing in this environment. There is therefore no upstream numeric
baseline to time. Per `README.md` (verified by reading the package in the
repo's tests), upstream `delta-encoding` has **no numeric core in Python** —
`DeltaOptimizeBuilder.executeZOrderBy` validates the column list and hands it
to a JVM builder; there is no loop over an array or buffer to compare against.
Every comparison below is therefore against the fastest reasonable **NumPy /
Python baseline from `bench/bench.py`** (vectorised bit-interleave,
`argsort(kind="stable")`, `minimum.reduceat`, broadcast comparison,
run-length encode, Python sweep), which is the honest stand-in. The one
genuine parity point, `validate_zorder_columns`, is Python work in both
packages and was timed directly (0.33 µs).

## Ranked hotspot table

Sizes are the sizes real callers pass: the README example and `bench/bench.py`
use n = 1M–4M rows, nfiles = 32–128, compaction nfiles = 65,536. Bytes touched
counts kernel reads + writes (plus wrapper copies where flagged). Flops are
simple integer/float ops counted from the kernel source; flops per byte
(flops/byte) is the roofline discriminator (GPU is only interesting well
above ~2 flops/byte).
"Kernel" = time inside `src/kernels.mojo`; "Python/ctypes glue" = the
`python/mojo_delta_encoding/_lib.py` (and `zorder.py`) wrapper around it.
Rows are ranked by wall time at the listed size, descending.
**SLOWER** marks where the port loses to the baseline. Nothing was dropped.

| # | function | input size | time (measured) | bytes touched | flops | flops/byte | bound | time sits in |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | `radix_sort` | n=2,097,152 keys+payload | 370–460 ms (bench: 392/369/376/420 ms; numpy 284–306 ms) | ~512 MiB kernel (8 passes × ~32 B/elem) + ~48 MiB wrapper copies | ~168 M (n×10×8 passes) | 0.31 | memory-bound (scatter-latency) | kernel 88%, glue ~12% (copies 12.7 ms + allocs) |
| 2 | `zorder_keys` | n=2,097,152, k=3, bits=20 | 130–147 ms (bench: 134–142 ms; numpy 568–606 ms) | 64 MiB (50.3 read + 16.8 write) + `_stack` copy 50.3 MiB when n=2M | ~755 M (n×k×bits×~6) | 11.2 | **compute-bound** (serial bit loop, 0.49 GB/s) | kernel ~82%, glue ~18% (`_stack` 23.5 ms dominates glue) |
| 3 | `zorder_rewrite` (pipeline) | n=1,048,576, k=2, bits=25, f=32 | 150–160 ms (bench: 136–160 ms; numpy 347–366 ms) | sum of stages (keys 16.8 MiB + sort ~256 MiB + bounds ~28 MiB) | sum of stages | ~0.5 blended | memory-bound overall (dominated by sort) | kernel ~85%, glue ~15% |
| 4 | `range_counts` | n=1,048,576, f=128 | 149–158 ms (bench: 150–156 ms; numpy 312–341 ms) | 1,024 MiB (keys re-read f×) | ~537 M (n×f×4) | 0.50 | memory-bound (sequential re-scan, 6.8 GB/s) | kernel ~100%, glue <1% |
| 5 | `prune_rows` | n=1,048,576, f=32 (77 ms at f=64) | 39 ms | 256 MiB (f=32 × 8 MiB) | ~134 M | 0.50 | memory-bound — **not in `bench/bench.py`**, scales O(n·f) | kernel (via `range_counts`) ~100%, glue <1% |
| 6 | `file_bounds` | n=1,048,576, k=3, f=64 | 19–25 ms (bench: 19–21 ms; numpy 48–54 ms) | ~28 MiB kernel (gather) + 24 MiB `_stack` copy | ~12.6 M (n×k×4) | 0.43 | memory-bound (random gather, ~1.2 GB/s) | kernel ~48% (9.5 ms), **Python glue ~52%** (wrapper 19.5 ms) |
| 7 | `prefix_rank` | n=4,194,304, bits=30 | 14–16 ms (bench: 14–16 ms; numpy 160–172 ms) | 64 MiB (32 read + 32 write) | ~21 M (n×5) | 0.31 | memory-bound (4.1 GB/s) | kernel ~47% (6.9 ms), glue ~53% (wrapper 14.4 ms) |
| 8 | `compaction_groups` | nfiles=65,536 | 9.4–12.8 ms (bench: 9.9–10.6 ms; python sweep 20–22 ms) | 1.0 MiB kernel (0.5 read + 0.5 write) | ~0.33 M (n×5) | 0.31 | latency-bound (serial FP-add chain, 1.3 GB/s direct) | **Python glue ~96%** (direct 0.39 ms, wrapper 9.4–10.7 ms) |
| 9 | `zorder_rewrite` FLOAT mode | n=1,048,576, k=2, bits=25, f=32 | 203 ms | keys 16.8 + sort ~256 + float-bits 2×16 MiB | blended | ~0.5 | memory-bound (same as INT rewrite + two NumPy float transforms) | kernel ~85%, glue (incl. NumPy float bits) ~15% |
| 10 | `zorder_keys` FLOAT mode | n=1,048,576, k=1, bits=20 | 23–25 ms | 16 MiB | ~126 M | 7.9 | compute-bound (same bit loop + 1 xor per col) | kernel ~97%, glue ~3% |
| 11 | `sortable_float_bits` | n=1,048,576 float64 | 2.1–5.2 ms | 16 MiB (8 read + 8 write) | ~3.1 M | 0.19 | memory-bound (3.6 GB/s) | **no kernel — 100% NumPy in Python wrapper** |
| 12 | `zorder_float_bits` | n=1,048,576 float64 | 2.0–5.3 ms | 16 MiB | ~3.1 M | 0.19 | memory-bound (same) | **no kernel — 100% NumPy in Python wrapper** |
| 13 | `assign_files` | n=1,048,576, f=64 | 2.1 ms | 12 MiB (8 keys + 0.5 split + 4 out) | ~3.1 M | 0.25 | memory-bound (5.8 GB/s) | kernel ~100%, glue <1% |
| 14 | `range_count` (single) | n=1,048,576 | 1.1–1.2 ms | 8 MiB | ~2.1 M | 0.25 | memory-bound (7.5 GB/s) | kernel ~100%, glue <1% |
| 15 | `file_row_bounds` | n=1,048,576, f=64 | 0.58 ms | 4 MiB (`searchsorted`) | — (zippier than a scan; binary search) | — | memory-bound | **no kernel — 100% NumPy (`np.searchsorted`)** |
| 16 | `split_points` | n=1,048,576, f=64 | 5.2 µs | 0.5 KiB out (O(f) reads) | ~65 | ~0.13 | latency-bound trivial | kernel dominant; 5 µs incl. ctypes |
| 17 | `prune_files` | f=64, 2 cols | 10–13 µs | 1 KiB | — | — | trivial | **pure NumPy in `zorder.py`, no kernel** |
| 18 | `validate_zorder_columns` | 3 cols | 0.33 µs | — | — | — | trivial | pure Python (parity point, no kernel) |

`pixi run bench` (final run, best of 3 per case, this box):

```
case                          reference  mojo-delta-encoding     ratio
zorder_keys n=2097152 k=3 b=20  605.85ms             139.64ms     4.34x
radix_sort n=2097152          304.96ms             419.55ms  0.73x SLOWER
file_bounds n=1048576 k=3 f=64  54.04ms              19.89ms    2.72x
range_counts n=1048576 f=128   340.67ms             155.24ms    2.19x
compaction_groups n=65536      21.69ms               9.75ms    2.22x
prefix_rank n=4194304 b=30     172.05ms              15.64ms   11.00x
zorder_rewrite n=1048576 f=32  365.90ms             160.17ms    2.28x
```

**`radix_sort` is SLOWER than the NumPy baseline (0.61–0.83× across runs).**
NumPy's `argsort(kind="stable")` on int64 is itself a vectorised radix sort;
this kernel does 8 passes of scalar counting + random scatter with no SIMD,
so it does not win the bandwidth race. It stays in the table as a loss.
Every other row beats its baseline. **`prune_rows`, `sortable_float_bits`,
`zorder_float_bits`, `file_row_bounds`, `prune_files`, `validate_zorder_columns`,
`split_points`, `assign_files`, `range_count`, and float-mode paths are not
timed by `bench/bench.py`** — they are timed here so the next phases do not
optimise blind.

### Where each millisecond goes (stage split, measured)

`zorder_rewrite` n=1M, k=2, bits=25, f=32 (isolated stage timings):

| stage | time | share of 157 ms total |
| --- | --- | --- |
| `radix_sort` | 95 ms | 60% |
| `zorder_keys` | 49 ms | 31% |
| `file_bounds` | 16 ms | 10% |
| `assign_files` | 2.1 ms | 1% |
| `file_row_bounds` | 0.6 ms | <1% |
| `split_points` | 0.006 ms | ~0% |

(Stages sum to ~163 ms; the fused `zorder_rewrite` total is 157 ms — the
difference is `keys` buffer reuse. The sort is the pipeline bottleneck,
the curve is second.)

Scaling that matters (measured): `range_counts` is exactly O(n·f) —
39 ms at f=32, 156 ms at f=128, 619 ms at f=512 (n=1M). `zorder_keys` scales
linearly in k·bits (19 ms for k=1/b=20, 49 ms for k=2/b=25, 61 ms for k=3/b=20
at n=1M) — pure compute per key bit. `radix_sort` is 196 ms at n=1M vs. 370–460 ms
at n=2M (NumPy 138/284–306 ms at the same sizes), so the gap widens with n.

## Per-call allocations and copies (measured)

- `zorder_keys` / `file_bounds` call `_stack()`, which does
  `np.stack(prepared)` — a **full copy of every column on every call**,
  even when the caller already passed contiguous int64 arrays
  (`_i64` per column is a no-op at ~1 µs total, but `np.stack` copies).
  Measured: 1.3 ms for 16.8 MiB (k=2, n=1M), 2.0 ms for 25.2 MiB (k=3, n=1M),
  23.5 ms for 50.3 MiB (k=3, n=2M). At n=2M this is ~18% of the wrapper time.
  `np.empty(n)` for `out` is negligible (2 perf_counter resolution scale).
  Small-n cost: 30 µs at n=1,000 (vs. 10 µs ctypes round-trip floor, see below).
- `radix_sort` copies **both** inputs (`keys.copy()`, `values.copy()` — 24 MiB
  at n=2M) and allocates two scratch arrays (24 MiB) plus the 256-entry count
  array. Copies + allocs alone measure 6–13 ms (~3% of the 370–420 ms kernel
  time), so they are not the bottleneck here, but they are pure overhead for
  any caller who could hand over a scratch buffer.
- `file_bounds` also allocates `omin`/`omax` zeros (negligible, f·k = 192 values)
  but runs **four whole-array validation passes in Python**
  (`bounds.min()`, `bounds.max()`, `bounds[1:] < bounds[:-1]`,
  `order.min()`, `order.max()` — 0.31 ms) plus the `_stack` copy on top of the
  9.5 ms kernel. That is why the wrapper is 19.5 ms vs. 9.5 ms direct
  (see #6 above). A caller who just rewrote a table revalidates data they
  know is well-formed.
- `compaction_groups` allocates `start`/`length` (n each, 256 KiB total at
  65,536 — ~1.4 µs) — not the problem. The problem is the return path
  (see below).
- `prefix_rank`, `range_count(s)`, `split_points`, `assign_files` each do one
  `np.empty(out)` — negligible. `zorder_rewrite` (float mode) additionally
  allocates an intermediate (`sortable_float_bits` per column, 2×8 MiB at n=1M).

## Is there any Python loop that re-enters ctypes per element?

**No per-element ctypes re-entry anywhere in the shim.** All 10 `lib.*` call
sites in `_lib.py` are one call per public function (one call per kernel
invocation), and `zorder.py`'s only loops call `_lib.range_counts` once
(`prune_rows`) or call NumPy (`prune_files`). `columns.py` has no ctypes at
all. Grep evidence (`bench/analysis.md` run, this session):
`python/mojo_delta_encoding/_lib.py`: 10 `lib.` sites, one per function;
the only `for` loops are column normalisation (`_stack`, no ctypes) and the
list-build below. `bench/bench.py`'s loops are in the baselines, not the port.

**There is one per-element Python loop, and it is the whole compaction
hotspot.** `_lib.compaction_groups` finishes with

```python
return [(int(start[i]), int(length[i])) for i in range(int(ngroups[0]))]
```

27,639 groups at nfiles=65,536 (63,000–422,000 groups measured across sizes:
65k files → ~27.6k groups; 1M files → 422k groups). Measured: the list-build
alone is **9.1 ms**; the kernel is **0.39 ms**; the wrapper is 9.4–10.7 ms.
At nfiles=1M the wrapper is 200 ms vs. 6.4 ms direct — **97% of the time is
the Python list comprehension**, growing linearly in group count.
End-to-end `compaction_groups` still beats the Python sweep baseline (2.1–2.2×)
only because the sweep pays the same per-file Python cost plus a Python sink
per file; the kernel itself is ~25× faster than the wrapper that calls it.
Tiny-n single-call floor: ~15 µs at n=1,000, ~7 µs at n=10 (ctypes + `_f64`
+ 3 allocs), so the wrapper is not overhead-bound at small n — it is
group-count-bound at large n.

## Next phase per hotspot

Phase vocabulary: `unroll`, `simd`, `parallel`, `gpu`, `none`. A GPU verdict
requires the flops/byte figure behind it. No kernel here exceeds ~11 flops/byte,
and the only one above ~1 (the zorder curve) is still serial-latency-bound,
not GPU-throughput-bound — **no hotspot earns `gpu`** on today's arithmetic.
If a future phase changes the algorithm (e.g. a parallel-scan compaction or a
binary-search range count), the roofline changes and GPU must be re-evaluated,
but under the current kernels the verdict is: memory-bound kernels get
`parallel`, compute/latency kernels get `simd`/`unroll`, glue gets `none`
(fix the copy/return, not the kernel).

| hotspot | next phase | evidence (measured) |
| --- | --- | --- |
| `radix_sort` (#1) | `parallel` | 0.31 flops/byte → memory-bound, scatter-latency. 370–460 ms for 512 MiB (1.26 GB/s vs. ~7 GB/s for streaming scans). 8 passes are embarrassingly parallel across buckets *except* the scatter needs atomic histograms; a K-way parallel or 16-bit-digit variant cuts passes 8→4. Not `simd` (scatter is random by construction) and not `gpu` (0.31 f/B). The NumPy loss (0.73×) is a bandwidth loss, and bandwidth is what threads buy here. |
| `zorder_keys` (#2) | `simd` (with `unroll`) | 11.2 flops/byte, 0.49 GB/s — the **only** compute-bound kernel, and it is a scalar bit-loop with a branch per bit (`if ((q>>b)&1)==1: z |= 1<<pos`). Branchless PDEP/magic-multiply interleave + SIMD across rows is where the 130 ms goes; `parallel` alone would just run the same branchy loop on more cores at 0.49 GB/s/core. Not `gpu`: 11 f/B is latency/ILP-bound, not throughput-bound, and a 64 MiB working set does not amortise launch + copy. |
| `zorder_rewrite` pipeline (#3) | `parallel` (inherits #1/#2) | 60% sort + 31% curve. Fixing #1 (parallel radix) and #2 (SIMD curve) fixes the pipeline; no separate phase needed. |
| `range_counts` / `prune_rows` (#4, #5) | `parallel` | 0.50 flops/byte, O(n·f) re-reads (1 GiB at f=128, 6.8 GB/s — already at streaming speed, so the *only* win is doing fewer bytes or more streams). The kernel is a perfect work-steal: f independent linear scans → thread over ranges. `simd` second (two compares vectorise trivially). `gpu` needs >~2 f/B; this is 0.50. Note: the *algorithmic* fix (binary search on sorted keys, O(f·log n)) is out of scope for ANALYZE (no algorithm changes), but flag it: at f=512 this row alone is 619 ms and dwarfs the rewrite. |
| `file_bounds` (#6) | `none` (fix glue, then `parallel`) | 0.43 flops/byte, gather-bound at 1.2 GB/s; but **52% of wrapper time is Python** (stack copy + 4 validation passes). Cheapest next step is removing the copy/validation, not the kernel. After glue: `parallel` (gather scales across files). Not `gpu` (0.43 f/B). |
| `prefix_rank` (#7) | `none` (fix glue) | 0.31 flops/byte, 4.1 GB/s, already 11× faster than NumPy. 53% of *wrapper* time is glue (`_i64` + `np.empty` + ctypes envelope, 14.4 vs. 6.9 ms), but at 15 ms total it is not worth a kernel phase. Kernel is a linear scan — `parallel` would need a scan prefix to split runs, real work for a 15 ms row. Lowest priority of the large rows. |
| `compaction_groups` (#8) | `none` (fix the return) | 0.31 flops/byte, kernel 0.39 ms vs. wrapper 9.4 ms (**96% Python**). Fix the list-build (e.g. return arrays, or build the list in the kernel / a single vectorised conversion) and the true cost is <1 ms. No SIMD/parallel/GPU verdict is possible until the wrapper stops being 96% of the measurement. The kernel's serial FP-add chain (`acc += sz`) also blocks SIMD/parallel without an algorithm change. |
| float-mode rows (#9, #10, #11, #12) | `none` | Float bits are NumPy in the wrapper (no kernel exists to optimise); float `zorder_keys` inherits the #2 verdict (compute-bound, 7.9 f/B → `simd`). The rewrite-float total is dominated by the same sort as #3. |
| small rows (#13–#18) | `none` | `assign_files` (5.8 GB/s), `range_count` (7.5 GB/s) are already at streaming speed for 1–2 ms rows. `file_row_bounds` / `prune_files` / `validate_zorder_columns` have no kernel and are ≤0.6 ms at bench sizes. `split_points` is 5 µs. None justifies a phase on its own. |

### Single-line priority for downstream phases (not a recommendation in this phase)

1. `radix_sort` — 370–460 ms, SLOWER than NumPy (0.73×), memory-bound → `parallel`.
2. `zorder_keys` — 130–147 ms, only compute-bound kernel (11.2 f/B), branchy bit loop → `simd`.
3. `range_counts` / `prune_rows` — 150 ms / O(n·f) scaling, memory-bound → `parallel`.
4. `compaction_groups` — 96% Python list-build → `none` (fix glue first).
5. `file_bounds` — 52% Python validation + stack copy → `none` (fix glue first).
6. Everything else is ≤15 ms at bench sizes or has no kernel; no phase.

## Reproduction

```bash
pixi run bench                                   # flock-wrapped; do not run alongside
pixi run python bench/bench.py                   # same, unwrapped — do not do this in parallel
# Per-function split (wrapper vs. direct lib.*) and stage breakdown were timed
# in ad-hoc `pixi run python` harnesses this session (best of 5–9 repeats,
# same inputs as bench/bench.py plus the un-benched entry points above).
```

No file under `src/` or `tests/` was modified; this is the only new file.
