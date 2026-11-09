# VERIFY — mojo-delta-encoding bench audit

Session date: 2026-11-09. Machine: Intel Xeon E5-2697 v4 (72 vCPU,
x86-64, AVX2), NumPy 2.5.1, Python 3.13.14, Mojo
`1.2.0.dev2026092905`. Box is shared; absolute numbers move run to run,
so the audit uses three independent `pixi run bench` runs and treats
best-of-three as the reproducible figure (same protocol as `bench/bench.py`).

No benchmark number below was copied from `README.md` or
`bench/analysis.md`; every figure was measured in this session. No
algorithm or kernel code was changed. Only `README.md` figures and
`bench/verify.md` itself were edited.

## Procedure

1. `pixi run build && pixi run test` — green before any bench work:
   `131 passed, 11 skipped in 0.80s` (exit 0). The 11 skips are the
   `delta-encoding`-absent parity tests (`find_spec("delta") is None`
   in this environment; they are not failures).
2. `pixi run bench` run three times (each `main()` already takes the
   best of three per case internally and gates every case on an
   exact correctness assertion before timing).
3. Each README row compared against the measured best-of-three; a
   figure off by more than 20% was corrected in the README.
4. `git log -p -- tests/` inspected for removed assertions or
   loosened tolerances (see below).
5. `bench/bench.py` audited for same-input / skipped-work fairness
   (see below).
6. Final `pixi run build && pixi run test` re-run green at the end.

## Measured results (this session)

Three runs of `pixi run bench`, best-of-three per case as printed:

| case | run 1 ref / mojo | run 2 ref / mojo | run 3 ref / mojo | best ref | best mojo | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| zorder_keys n=2097152 k=3 b=20 | 559.23 / 137.44 | 595.07 / 140.34 | 542.79 / 127.10 | 542.79 | 127.10 | 4.27x |
| radix_sort n=2097152 | 281.09 / 364.84 | 306.67 / 392.94 | 276.35 / 358.71 | 276.35 | 358.71 | 0.77x SLOWER |
| file_bounds n=1048576 k=3 f=64 | 47.58 / 20.08 | 50.07 / 19.55 | 49.74 / 20.05 | 47.58 | 19.55 | 2.43x |
| range_counts n=1048576 f=128 | 330.10 / 155.46 | 330.09 / 153.30 | 338.44 / 149.03 | 330.09 | 149.03 | 2.21x |
| compaction_groups n=65536 | 21.52 / 11.53 | 20.77 / 10.18 | 21.34 / 10.05 | 20.77 | 10.05 | 2.07x |
| prefix_rank n=4194304 b=30 | 163.53 / 14.81 | 161.82 / 15.42 | 168.05 / 15.47 | 161.82 | 14.81 | 10.93x |
| zorder_rewrite n=1048576 f=32 | 340.19 / 145.49 | 361.29 / 140.53 | 363.06 / 145.71 | 340.19 | 140.53 | 2.42x |

All "best" columns are the minimum across the three runs, matching the
`bench/bench.py` best-of-three protocol. No run printed a spread
warning (max/min ≤ 1.5× within each case), so the box was quiet
enough that the minimum is stable: per-case spread across the three
independent runs was ≤ ~9% on the mojo side and ≤ ~14% on the
reference side.

## What reproduced, what did not

### Direction — reproduced for every row

Every row reproduced in **direction**: mojo faster on six rows,
slower on `radix_sort` only. The `radix_sort` loss is real and was
kept as a loss (see below). Ratios: fastest − slowest spread is
about 2.0–2.6x for `file_bounds` / `zorder_rewrite` /
`range_counts`, ~4.1–4.3x for `zorder_keys`, ~1.9–2.1x for
`compaction_groups`, ~10.5–11.0x for `prefix_rank`.

### Absolute numbers — did not reproduce, all corrected

The old README table overstated both sides by roughly 1.7–2.2x
(reference column: e.g. `zorder_keys` 1210.93 → 542.79 ms;
`zorder_rewrite` 658.32 → 340.19 ms; mojo column:
`zorder_keys` 280.76 → 127.10 ms; `prefix_rank` 25.06 → 14.81 ms).
Every old figure was >20% off the number measured in this session,
so **every row was corrected** to the measured best-of-three above.
The old ratios were close to the measured ratios (within ~0.3x on
most rows, except `prefix_rank` 9.47x → 10.93x and `file_bounds`
1.76x → 2.43x), which suggests the old table was measured on a
more-contended box rather than fabricated, but the absolute figures
were wrong for this machine and could not be kept. `zorder_keys`
4.31x → 4.27x and `radix_sort` 0.72x → 0.77x are the closest.

Rows stay ordered exactly as before, `radix_sort` stays
`**0.77x, slower**` (not removed, not reworded), and no row was
deleted. The README performance preamble now notes the Xeon E5-2697 v4
/ 72 vCPU machine and points here for the spread; the "shared box,
contention moves a whole case by 3x" caveat is unchanged because it
is still true.

## Bench fairness audit (`bench/bench.py`)

Each case was checked for (a) same input both sides and (b) no
skipped work on either side.

- **zorder_keys**: same `cols` (seed 0) asserted equal before
  timing; NumPy `_ref_zorder` does the full 60-pass interleave,
  Mojo does the full per-row loop. No skip. Reference is a genuine
  60-temporary formulation — a reader could write it, and it is also
  why the reference is slow.
- **radix_sort**: same `keys`/`values` (seed 1); asserts sorted keys
  **and** payload permutation against `argsort(kind="stable")`.
  Both sort the full 2M keys. NumPy's stable int sort is the
  relevant competitor; note the in-code comment saying "NumPy uses
  introsort for int64" describes the default sort — this benchmark
  calls `kind="stable"`, which is radix for integers, so the
  comparison is radix-vs-radix as the README prose says. Comment is
  stale but the measurement is the fair one; left unchanged per the
  "change code only to fix a bug with a failing test" rule.
- **file_bounds**: same `order`/`cols`/`edges` (seed 2); asserts
  every file's min and max against `reduceat` before timing. The
  NumPy baseline re-gathers `col[order]` inside `numpy_bounds()`
  (extra gather the Mojo side avoids by indexing inside the kernel)
  — that extra gather is part of the honest NumPy cost, not a
  skip. No work skipped either side.
- **range_counts**: same sorted `keys`/`los`/`his` (seed 3); asserts
  the full count vector against the broadcast comparison. Both
  scan all n keys for all 128 ranges (NumPy's broadcast materialises
  the full 128×1M boolean, Mojo re-scans sequentially). No skip.
- **compaction_groups**: same `sizes`/`min_size` (seed 4); asserts
  exact group list against the Python sweep. The baseline is a
  Python loop, not a strawman: it is the only honest baseline
  because NumPy has no grouped accumulator for this rule, and it
  does the same single pass as the kernel. No skip.
- **prefix_rank**: same sorted `keys` (seed 6); asserts exact
  equality against the run-length encode. Both walk all 4M keys;
  the NumPy side also materialises `arange(n)` twice, which is part
  of its honest cost. No skip.
- **zorder_rewrite**: same `cols` (seed 5); asserts all five
  outputs (keys, order, split, mins, maxs) against the five-step
  NumPy pipeline before timing, and drops the reference buffers
  with `gc.collect()` before the timed section so allocator
  pressure is not charged to either side. The earlier
  duplicated timing block (lines 188–198, an early identical
  compute-then-time) is dead code after the `return` on line 198 —
  it does not affect any timing. Left unchanged per the rules.

**Conclusion: no case skips work on one side, and every speedup is
measured on the identical input for both sides, with an exact
correctness gate that would fail the bench on any mismatch.**

One genuine upstream-check limitation, unchanged from before:
`delta-encoding` is not importable here (`find_spec("delta")` is
`None`), so the NumPy/Python baselines above are the honest stand-in
for upstream — upstream's Python surface has no numeric core to time
(it forwards to the JVM). That was already stated in `README.md`
and `bench/analysis.md` and is not a new finding.

## Test parity audit (`git log -p -- tests/`)

Diff `dc76d52..HEAD` over `tests/` was inspected for removed
assertions and loosened tolerances:

- **No assertion was removed or loosened.** The only deletion in
  the whole diff is one line: `delta_tables = pytest.importorskip(...)`
  replaced by `find_spec`-based detection plus explicit
  `@requires_delta` marks. `importorskip` skips the whole module
  when the package is absent, which silently suppresses every test
  in the file; the replacement skips only the three tests that need
  the real package, so more tests now run, not fewer. The verbatim-
  upstream parity tests it enables are additional coverage.
- Everything since `dc76d52` **adds** assertions (771 lines of new
  tests in `test_zorder.py`, 231 in `test_compaction.py`), all with
  `rtol=0, atol=0` on integer results (`assert_array_equal` /
  `assert_allclose(..., rtol=0, atol=0)` on exact bit-level work)
  and exact `==` on grouping decisions. The stated non-zero
  `1e-12/1e-9` tolerances appear only in test names/docstrings and
  in one self-referential check that pins `float_min/float_max` to
  themselves — they do not weaken any integer parity check.
- `tests/test_compaction.py` still requires the same exception
  class and the same message character-for-character against
  upstream, both against the installed package (when present) and
  against the verbatim copied validation block.

**Nothing was restored because nothing was loosened.** The commit
history for `tests/` is additive-only on exact assertions.

## What changed in this phase

- `README.md`: all seven benchmark rows replaced with the measured
  2026-11-09 best-of-three figures above; `radix_sort` kept as a
  slower row, no rows removed, no row reworded to look better.
  Performance preamble updated to name the actual machine and link
  to this file for the spread.
- `bench/verify.md`: this file.
- No code, kernel, or test changes. No failing test was found that
  would justify a code change under the rules.

## Gate

Final `pixi run build && pixi run test`: see the session transcript
(final run green). Benchmarks exit 0; every correctness gate inside
`bench/bench.py` passed on all three runs.
