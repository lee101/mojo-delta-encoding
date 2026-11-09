# VERIFY — mojo-delta-encoding bench audit

Session date: 2026-11-09. Machine: Intel Xeon E5-2697 v4 (72 vCPU,
x86-64, AVX2), NumPy 2.5.1, Python 3.13.14, Mojo
`1.2.0.dev2026092905`. Box is shared; absolute numbers move run to
run, so the audit uses three independent `pixi run bench` runs and treats
best-of-three as the reproducible figure (same protocol as `bench/bench.py`).

Every figure below was measured in this session; nothing was copied from
the old `README.md` or `bench/analysis.md`. No algorithm, kernel, or test
code was changed. Only `README.md` figures and this file were edited.

## Procedure

1. `pixi run build && pixi run test` — green before any bench work:
   `131 passed, 11 skipped in 0.76s` (exit 0). The 11 skips are the
   `delta-encoding`-absent parity tests (`find_spec("delta") is None`
   here; they are not failures).
2. `pixi run bench` run three times (each `main()` already takes the
   best of three samples per case internally and gates every case on an
   exact correctness assertion before timing).
3. Each README row compared against the measured best-of-three; the
   deviation is recorded below.
4. `git log -p -- tests/` inspected for removed assertions or loosened
   tolerances (see below).
5. `bench/bench.py` audited for same-input / skipped-work fairness
   (see below).
6. Final `pixi run build && pixi run test` re-run green at the end.

## Measured results (this session)

Three runs of `pixi run bench`, best-of-three per case as printed:

| case | run 1 ref / mojo | run 2 ref / mojo | run 3 ref / mojo | best ref | best mojo | ratio |
| --- | --- | --- | --- | --- | --- | --- |
| zorder_keys n=2097152 k=3 b=20 | 580.00 / 133.34 | 565.55 / 137.50 | 561.81 / 141.35 | 561.81 | 133.34 | 4.21x |
| radix_sort n=2097152 | 286.89 / 366.29 | 297.31 / 379.97 | 303.54 / 386.65 | 286.89 | 366.29 | 0.78x SLOWER |
| file_bounds n=1048576 k=3 f=64 | 46.95 / 18.06 | 47.17 / 18.20 | 48.65 / 19.21 | 46.95 | 18.06 | 2.60x |
| range_counts n=1048576 f=128 | 303.47 / 147.89 | 323.64 / 155.60 | 342.25 / 155.94 | 303.47 | 147.89 | 2.05x |
| compaction_groups n=65536 | 20.47 / 9.99 | 21.42 / 10.18 | 21.08 / 10.11 | 20.47 | 9.99 | 2.05x |
| prefix_rank n=4194304 b=30 | 158.42 / 15.22 | 161.74 / 14.98 | 165.90 / 15.18 | 158.42 | 14.98 | 10.58x |
| zorder_rewrite n=1048576 f=32 | 357.10 / 148.34 | 350.29 / 137.04 | 363.09 / 145.16 | 350.29 | 137.04 | 2.56x |

All "best" columns are the minimum across the three runs, matching the
`bench/bench.py` best-of-three protocol. No run printed a spread warning
(max/min ≤ 1.5× within each case). Per-case spread across the three
independent runs was ≤ 1.13× on the reference side and ≤ 1.08× on the
mojo side.

## What reproduced, what did not

### Direction — reproduced for every row

Every row reproduced in **direction**: mojo faster on six rows, slower
on `radix_sort` only. The `radix_sort` loss is real and was kept as a
loss (see below).

### Absolute numbers — reproduced within 20%, refreshed to this session

Against the table that was in the README at the start of this session
(measured 2026-11-09 in the previous VERIFY pass), the measured
best-of-three above deviates by:

| case | ref deviation | mojo deviation | ratio deviation |
| --- | --- | --- | --- |
| zorder_keys | +3.5% | +4.9% | −1.3% |
| radix_sort | +3.8% | +2.1% | +1.7% |
| file_bounds | −1.3% | −7.6% | +7.0% |
| range_counts | −8.1% | −0.8% | −7.1% |
| compaction_groups | −1.4% | −0.6% | −1.0% |
| prefix_rank | −2.1% | +1.1% | −3.2% |
| zorder_rewrite | +3.0% | −2.5% | +5.6% |

Every figure reproduced within 20%, so no row met the correction
threshold, but the README figures are now the numbers measured in this
session (the rule that a number not measured in this session does not
go in the README overrides keeping the earlier session's best-of-three
verbatim). Rows stay ordered exactly as before, `radix_sort` stays
`**0.78x, slower**` (not removed, not reworded), and no row was
deleted.

### What could not be reproduced

- Exact millisecond figures: on a shared box the absolute times move a
  few percent between runs (spread above), so no earlier figure
  reproduces bit-for-bit. The figures that do reproduce are the
  direction, the ratios to within ~7%, and the order of magnitude.
- An upstream timing: `delta-encoding` is not installed here
  (`find_spec("delta") is None`), and upstream's Python surface has no
  numeric core to time (it forwards to the JVM). The NumPy/Python
  baselines in `bench/bench.py` are therefore the honest stand-in,
  and the README says so. Nothing beyond that is claimed.

## Bench fairness audit (`bench/bench.py`)

Each case was checked for (a) same input both sides and (b) no skipped
work on either side.

- **zorder_keys**: same `cols` (seed 0) asserted equal before timing;
  NumPy `_ref_zorder` does the full 60-pass interleave, Mojo does the
  full per-row loop. No skip.
- **radix_sort**: same `keys`/`values` (seed 1); asserts sorted keys
  **and** payload permutation against `argsort(kind="stable")`. Both
  sort the full 2M keys. The in-code comment saying "NumPy uses
  introsort for int64" describes the default sort; the benchmark calls
  `kind="stable"`, which is radix for integers, so the comparison is
  radix-vs-radix as the README prose says. Comment is stale but the
  measurement is the fair one; left unchanged per the "change code only
  to fix a bug with a failing test" rule.
- **file_bounds**: same `order`/`cols`/`edges` (seed 2); asserts every
  file's min and max against `reduceat` before timing. The NumPy
  baseline re-gathers `col[order]` inside `numpy_bounds()` (extra
  gather the Mojo side avoids by indexing inside the kernel) — that
  extra gather is part of the honest NumPy cost, not a skip.
- **range_counts**: same sorted `keys`/`los`/`his` (seed 3); asserts
  the full count vector against the broadcast comparison. Both scan
  all n keys for all 128 ranges. No skip.
- **compaction_groups**: same `sizes`/`min_size` (seed 4); asserts
  exact group list against the Python sweep. The baseline is a Python
  loop, not a strawman: it is the only honest baseline because NumPy
  has no grouped accumulator for this rule, and it does the same single
  pass as the kernel.
- **prefix_rank**: same sorted `keys` (seed 6); asserts exact equality
  against the run-length encode. Both walk all 4M keys. No skip.
- **zorder_rewrite**: same `cols` (seed 5); asserts all five outputs
  (keys, order, split, mins, maxs) against the five-step NumPy pipeline
  before timing. The earlier duplicated timing block after the first
  `return` (lines ~201–217) is dead code — it does not affect any
  timing. Left unchanged per the rules.

**Conclusion: no case skips work on one side, and every speedup is
measured on the identical input for both sides, with an exact
correctness gate that would fail the bench on any mismatch.**

## Test parity audit (`git log -p -- tests/`)

Diff `dc76d52..HEAD` over `tests/` was inspected for removed assertions
and loosened tolerances:

- **No assertion was removed or loosened.** The only deletion in the
  whole diff is one line: `delta_tables = pytest.importorskip(...)`
  replaced by `find_spec`-based detection plus explicit
  `@requires_delta` marks. `importorskip` skips the whole module when
  the package is absent, which silently suppresses every test in the
  file; the replacement skips only the tests that need the real
  package, so more tests run, not fewer. The verbatim-upstream parity
  tests it enables are additional coverage.
- Everything since `dc76d52` **adds** assertions (771 lines of new
  tests in `test_zorder.py`, 231 in `test_compaction.py`), all with
  `rtol=0, atol=0` on integer results (`assert_array_equal` /
  `assert_allclose(..., rtol=0, atol=0)` on exact bit-level work) and
  exact `==` on grouping decisions. The stated non-zero
  `1e-12/1e-9` tolerances appear only in test names/docstrings and in
  one self-referential check that pins `float_min/float_max` to
  themselves — they do not weaken any integer parity check.
- `tests/test_compaction.py` still requires the same exception class
  and the same message character-for-character against upstream, both
  against the installed package (when present) and against the verbatim
  copied validation block.

**Nothing was restored because nothing was loosened.** The commit
history for `tests/` is additive-only on exact assertions.

## What changed in this phase

- `README.md`: benchmark rows refreshed to the measured 2026-11-09
  best-of-three figures above (all within 20% of the prior figures;
  refreshed so that only this-session numbers are in the README).
  `radix_sort` kept as a slower row, no rows removed, no row reworded
  to look better.
- `bench/verify.md`: this file.
- No code, kernel, or test changes. No failing test was found that
  would justify a code change under the rules.

## Gate

Final `pixi run build && pixi run test`: see the session transcript
(final run green). Benchmarks exit 0; every correctness gate inside
`bench/bench.py` passed on all three runs.
