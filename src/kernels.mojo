"""Z-order curve, Z-order file bounds, data-skip counts and compaction grouping.

These are the numeric kernels behind `OPTIMIZE Z-ORDER BY` and `OPTIMIZE`
compaction. The Python package `delta-encoding` names those operations in
`DeltaOptimizeBuilder.executeZOrderBy` and `executeCompaction` but implements
neither in Python -- it forwards the column list to a JVM builder. The algorithms
themselves live in `delta-core`, so this port puts a compiled version of them
behind a C ABI and says plainly, in the README, where it came from.

Every exported symbol takes buffer addresses as plain `Int` values and rebuilds
the pointer inside the body, because `@export` rejects parametric functions and
an inferred pointer origin would make the symbol parametric. No kernel here
allocates: the radix-sort scratch buffers are owned by the Python shim.
"""

comptime I64Ptr = Pointer[Int64, AnyOrigin[mut=True]]
comptime I32Ptr = Pointer[Int32, AnyOrigin[mut=True]]
comptime F64Ptr = Pointer[Float64, AnyOrigin[mut=True]]


def p64(addr: Int) -> I64Ptr:
    return I64Ptr(unsafe_from_address=addr)


def p32(addr: Int) -> I32Ptr:
    return I32Ptr(unsafe_from_address=addr)


def pf(addr: Int) -> F64Ptr:
    return F64Ptr(unsafe_from_address=addr)


# ---------------------------------------------------------------------------
# Z-order curve
# ---------------------------------------------------------------------------


@export("de_zorder_keys")
def de_zorder_keys(cols_addr: Int, n: Int, kcols: Int, bits: Int, mode: Int,
                   out_addr: Int) abi("C"):
    """Interleave the top `bits` of each of `kcols` columns into one Z-order key.

    `cols` is row-major: `kcols` rows of `n` `int64` values, so column c of row
    i is at `cols[c * n + i]`. Row i gets a key with `kcols * bits <= 63` bits
    set: bit b of the truncated column c lands at bit `b * kcols + c`, which is
    the Z-order (Morton) interleave.

    `mode` 0 treats each element as a raw signed integer. `mode` 1 treats it as
    the bit pattern of a `float64` and first applies the transform
    `u ^ ((u >> 63) | 1 << 63)`, which is monotone as an *unsigned* 64-bit
    value: a negative double maps to `~u`, a non-negative one to `u | 1 << 63`,
    so -inf < ... < -0.0 < +0.0 < ... < +inf. Because the retained window is
    read as an unsigned `bits`-wide number, this is what makes a float column's
    Z-order key ascend with the data, sign boundary included.

    Integer min and max want a different variant, because they compare signed:
    the Python shim's `sortable_float_bits` uses `u ^ ((u >> 63) & 0x7FFF_...)`
    for the statistics, and both transforms preserve the same total order, so
    the key and the per-file bounds never disagree about which row is smaller.
    """
    var cols = p64(cols_addr)
    var dst = p64(out_addr)
    if n <= 0 or kcols <= 0 or bits <= 0:
        return
    var sh = 64 - bits
    for i in range(n):
        var z = Int64(0)
        for c in range(kcols):
            var x = cols[unsafe_offset=c * n + i]
            if mode == 1:
                x = x ^ ((x >> 63) | Int64(0x8000000000000000))
            var q = x >> Int64(sh)
            var pos = c
            for b in range(bits):
                if ((q >> Int64(b)) & Int64(1)) == 1:
                    z = z | (Int64(1) << Int64(pos))
                pos += kcols
        dst[unsafe_offset=i] = z


# ---------------------------------------------------------------------------
# Sorting and Z-order file assignment
# ---------------------------------------------------------------------------
def radix_pass(src_k: I64Ptr, src_v: I32Ptr, dst_k: I64Ptr, dst_v: I32Ptr,
               cnt: I64Ptr, n: Int, shift: Int, flip_sign: Bool):
    """One stable counting pass on the byte at `shift`.

    `flip_sign` is set for the most significant byte, where the digit has its
    top bit inverted so that negative keys -- whose arithmetic shift puts every
    1 above the sign -- sort below positive ones. Without it the whole sort is
    an unsigned sort and INT64_MIN lands at the top.
    """
    for b in range(256):
        cnt[unsafe_offset=b] = 0
    for i in range(n):
        var d = (src_k[unsafe_offset=i] >> Int64(shift)) & 255
        if flip_sign:
            d = d ^ 128
        cnt[unsafe_offset=d] = cnt[unsafe_offset=d] + 1
    var total = Int64(0)
    for b in range(256):
        var c = cnt[unsafe_offset=b]
        cnt[unsafe_offset=b] = total
        total += c
    for i in range(n):
        var k = src_k[unsafe_offset=i]
        var d = (k >> Int64(shift)) & 255
        if flip_sign:
            d = d ^ 128
        var pos = cnt[unsafe_offset=d]
        cnt[unsafe_offset=d] = pos + 1
        dst_k[unsafe_offset=pos] = k
        dst_v[unsafe_offset=pos] = src_v[unsafe_offset=i]


@export("de_radix_sort_u64")
def de_radix_sort_u64(keys_addr: Int, vals_addr: Int, skey_addr: Int,
                      sval_addr: Int, cnt_addr: Int, n: Int) abi("C"):
    """Stable LSD radix sort of `n` signed 64-bit keys, permuting `vals` with them.

    Eight 8-bit passes, an even number, so the sorted result lands back in the
    caller's own buffers. `skey`/`sval` are scratch of length `n` and `cnt` is
    scratch of length 256. The digit is always masked with 255 so a negative key
    yields the same byte as its bit pattern, the top byte is sign-flipped to get
    signed order, and the passes are stable, so equal keys keep their input
    order.
    """
    var keys = p64(keys_addr)
    var vals = p32(vals_addr)
    var skey = p64(skey_addr)
    var sval = p32(sval_addr)
    var cnt = p64(cnt_addr)
    for p in range(8):
        var shift = p * 8
        var flip = p == 7
        if p % 2 == 0:
            radix_pass(keys, vals, skey, sval, cnt, n, shift, flip)
        else:
            radix_pass(skey, sval, keys, vals, cnt, n, shift, flip)


@export("de_split_points")
def de_split_points(keys_addr: Int, n: Int, nfiles: Int,
                    out_addr: Int) abi("C"):
    """Write `nfiles + 1` monotone split keys that cut sorted keys into
    `nfiles` nearly equal row ranges.

    `keys` must already be sorted. The boundary of file f is the key at index
    `f * n // nfiles`, so every file holds either `n // nfiles` or one more row.
    Duplicated keys can collapse a range to zero rows, which is correct: a bucket
    that would be empty stays empty. For `n == 0` every split is 0.
    """
    var keys = p64(keys_addr)
    var dst = p64(out_addr)
    if n <= 0:
        for f in range(nfiles + 1):
            dst[unsafe_offset=f] = 0
        return
    for f in range(nfiles + 1):
        if f == 0:
            dst[unsafe_offset=0] = keys[unsafe_offset=0]
        elif f == nfiles:
            dst[unsafe_offset=nfiles] = keys[unsafe_offset=n - 1]
        else:
            var idx = (f * n) // nfiles
            if idx >= n:
                idx = n - 1
            dst[unsafe_offset=f] = keys[unsafe_offset=idx]


@export("de_row_files")
def de_row_files(keys_addr: Int, n: Int, split_addr: Int, nfiles: Int,
                 out_addr: Int) abi("C"):
    """Assign each sorted key to a file, given the `nfiles + 1` split keys.

    File f owns the half-open key range `[split[f], split[f + 1])`, except the
    last file, which also takes the maximum key, so every row lands in exactly
    one file. The walk is linear because `keys` is sorted.
    """
    var keys = p64(keys_addr)
    var split = p64(split_addr)
    var dst = p32(out_addr)
    if nfiles <= 0:
        return
    var f = 0
    for i in range(n):
        while f < nfiles - 1 and keys[unsafe_offset=i] >= split[unsafe_offset=f + 1]:
            f += 1
        dst[unsafe_offset=i] = Int32(f)


# ---------------------------------------------------------------------------
# Per-file statistics: the data-skipping payload
# ---------------------------------------------------------------------------


@export("de_file_bounds")
def de_file_bounds(ord_addr: Int, cols_addr: Int, bounds_addr: Int, n: Int,
                   kcols: Int, nfiles: Int, omin_addr: Int,
                   omax_addr: Int) abi("C"):
    """Per-file, per-column min and max over the rows in each file.

    `ord` is the row permutation produced by the sort, so file f owns the rows
    `ord[lo] .. ord[hi)` where `bounds` holds the `nfiles + 1` row-index
    boundaries. `cols` is the row-major `kcols` by `n` source array. Output is
    `nfiles * kcols` `int64` values each for the min and the max.

    A file with no rows has no statistics; its bounds are written as 0 so a
    reader can tell it apart from a file whose columns are genuinely zero.
    """
    var ord = p32(ord_addr)
    var cols = p64(cols_addr)
    var bnd = p32(bounds_addr)
    var omin = p64(omin_addr)
    var omax = p64(omax_addr)
    for f in range(nfiles):
        var lo = bnd[unsafe_offset=f]
        var hi = bnd[unsafe_offset=f + 1]
        if hi <= lo:
            for c in range(kcols):
                omin[unsafe_offset=f * kcols + c] = 0
                omax[unsafe_offset=f * kcols + c] = 0
        else:
            var first = Int(ord[unsafe_offset=lo])
            for c in range(kcols):
                var v = cols[unsafe_offset=c * n + first]
                omin[unsafe_offset=f * kcols + c] = v
                omax[unsafe_offset=f * kcols + c] = v
            for i in range(lo + 1, hi):
                var r = Int(ord[unsafe_offset=i])
                for c in range(kcols):
                    var v = cols[unsafe_offset=c * n + r]
                    if v < omin[unsafe_offset=f * kcols + c]:
                        omin[unsafe_offset=f * kcols + c] = v
                    if v > omax[unsafe_offset=f * kcols + c]:
                        omax[unsafe_offset=f * kcols + c] = v


@export("de_range_count")
def de_range_count(keys_addr: Int, n: Int, lo: Int, hi: Int,
                   out_addr: Int) abi("C"):
    """Count rows whose Z-order key lies in the closed range `[lo, hi]`.

    This is the row-level data-skipping primitive: a file whose per-column bounds
    cannot overlap a predicate's range is skipped without being read. Inclusive
    bounds avoid the overflow an exclusive upper bound would hit at INT64_MAX.
    """
    var keys = p64(keys_addr)
    var dst = p64(out_addr)
    var count = Int64(0)
    for i in range(n):
        var k = keys[unsafe_offset=i]
        if k >= Int64(lo) and k <= Int64(hi):
            count += 1
    dst[unsafe_offset=0] = count


@export("de_range_counts")
def de_range_counts(keys_addr: Int, n: Int, lo_addr: Int, hi_addr: Int,
                    nfiles: Int, out_addr: Int) abi("C"):
    """Vector form of `de_range_count`: `nfiles` closed key ranges, one count
    each. `lo` and `hi` are `int64` arrays of length `nfiles`."""
    var keys = p64(keys_addr)
    var los = p64(lo_addr)
    var his = p64(hi_addr)
    var dst = p64(out_addr)
    for f in range(nfiles):
        var lo = los[unsafe_offset=f]
        var hi = his[unsafe_offset=f]
        var count = Int64(0)
        for i in range(n):
            var k = keys[unsafe_offset=i]
            if k >= lo and k <= hi:
                count += 1
        dst[unsafe_offset=f] = count


# ---------------------------------------------------------------------------
# Compaction
# ---------------------------------------------------------------------------


@export("de_compaction_groups")
def de_compaction_groups(sizes_addr: Int, n: Int, min_size: Float64,
                         start_addr: Int, len_addr: Int,
                         ngroups_addr: Int) abi("C"):
    """Group files for `OPTIMIZE` compaction from their sizes alone.

    A file already at or above `min_size` is left alone and becomes its own
    group. A run of smaller files accumulates until its total reaches
    `min_size`, then closes as a group, which is the rewrite unit. A trailing
    run of small files becomes one final group. Returns the group count through
    `ngroups`; `start` and `len` must hold `n` entries each.
    """
    var sizes = pf(sizes_addr)
    var start = p32(start_addr)
    var lens = p32(len_addr)
    var ngroups = p64(ngroups_addr)
    var g = 0
    var first = 0
    var acc = Float64(0.0)
    for i in range(n):
        var sz = sizes[unsafe_offset=i]
        if sz >= min_size:
            if i > first:
                start[unsafe_offset=g] = Int32(first)
                lens[unsafe_offset=g] = Int32(i - first)
                g += 1
            start[unsafe_offset=g] = Int32(i)
            lens[unsafe_offset=g] = 1
            g += 1
            first = i + 1
            acc = Float64(0.0)
        else:
            acc += sz
            if acc >= min_size:
                start[unsafe_offset=g] = Int32(first)
                lens[unsafe_offset=g] = Int32(i - first + 1)
                g += 1
                first = i + 1
                acc = Float64(0.0)
    if first < n:
        start[unsafe_offset=g] = Int32(first)
        lens[unsafe_offset=g] = Int32(n - first)
        g += 1
    ngroups[unsafe_offset=0] = Int64(g)


# ---------------------------------------------------------------------------
# Bucket occupancy of a Z-order key range
# ---------------------------------------------------------------------------


@export("de_prefix_rank")
def de_prefix_rank(keys_addr: Int, n: Int, prefix_bits: Int,
                   out_addr: Int) abi("C"):
    """For each key, write how many earlier keys share its top `prefix_bits`.

    `keys` must be sorted. Equal `prefix_bits` prefixes occupy a contiguous run,
    so one backward-looking scan assigns each run a start index. This is the
    bucket occupancy the Z-order writer consults when it decides whether a file
    range is dense enough to be worth pruning.
    """
    var keys = p64(keys_addr)
    var dst = p64(out_addr)
    if n <= 0 or prefix_bits <= 0:
        return
    var sh = Int64(64 - prefix_bits)
    var run_start = 0
    var prev = keys[unsafe_offset=0] >> sh
    for i in range(n):
        var p = keys[unsafe_offset=i] >> sh
        if i > 0 and p != prev:
            run_start = i
        prev = p
        dst[unsafe_offset=i] = Int64(i - run_start)
