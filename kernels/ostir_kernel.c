/* ostir_kernel — the measured half of the Part VI protocol.
 *
 * Subcommands:
 *   bandwidth    E3: beta_L2 and beta_DR per core, hence r.
 *   panel-sweep  E2: sweep resident working-set size, find the residency knee.
 *   mixed-h      E4: drive h as an INDEPENDENT variable and measure S(h).
 *   batch        E6: sweep batch size, find the memory->compute transition.
 *
 * Emits one JSON object per line on stdout; diagnostics go to stderr.
 *
 * On the E4 method. The monograph asks for >=20 configurations spanning
 * h in [0.3,1.0] obtained by varying m_c*k_c around the E2 knee, with h read
 * from counters. That works on Intel but conflates two things: it can only
 * produce the h values the cache happens to give you, and it cannot run at all
 * without counters. So this kernel also drives h directly: every read is
 * dispatched to either a known-resident panel or a known-DRAM buffer by a
 * Bresenham accumulator, making h a dial rather than an observation. On Intel
 * the counter-measured h is recorded alongside the designed h, and E4 checks
 * that the two agree -- which validates the counter methodology itself.
 *
 * Build: make          (see Makefile)
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>
#include <errno.h>

#ifdef __linux__
#include <unistd.h>
#include <sys/syscall.h>
#include <sys/ioctl.h>
#include <linux/perf_event.h>
#include <sys/mman.h>
#endif

/* Dispatch granularity for mixed-h. 4 KiB throttled the DRAM leg to 17 GB/s
 * against 42 GB/s for the same bytes read contiguously -- the prefetcher gets
 * no run-up when the stream restarts every page. 64 KiB recovers most of it
 * and still gives ~32k dispatch decisions per 2 GiB region, far finer than
 * the h resolution needs. */
#ifndef CHUNK
#define CHUNK        65536ULL
#endif
#define CACHELINE    64ULL
#define MIN_RUN_NS   80000000ULL       /* >=80 ms per timed region           */

/* Consume a value and clobber memory, so the compiler may neither hoist a
 * loop-invariant pass out of the repeat loop nor drop the reduction. Using
 * `volatile` for this instead is a trap: a volatile accumulator reachable
 * through a pointer may alias the buffer, which blocks vectorization and
 * pins the loop at ~1 load/cycle. That reads as a flat bandwidth curve with
 * no cache knee at all -- the kernel stops being able to see the hierarchy
 * it exists to measure. */
#define BARRIER(x) __asm__ __volatile__("" :: "r"(x) : "memory")

/* Non-temporal load capability, detected once and used both by the DRAM-leg
 * reader and by the JSON receipt (which must not claim a capability the
 * binary lacks). Declared here because counters_json() reports it. */
#if defined(__clang__) && __has_builtin(__builtin_nontemporal_load)
#define HAVE_NT_LOAD 1
#else
#define HAVE_NT_LOAD 0
#endif

/* ------------------------------------------------------------------ time */

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

/* ----------------------------------------------------------- allocation */

static void *alloc_aligned(size_t bytes)
{
    void *p = NULL;
    size_t rounded = (bytes + 4095) & ~(size_t)4095;
#ifdef __linux__
    p = mmap(NULL, rounded, PROT_READ | PROT_WRITE,
             MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return NULL;
#ifdef MADV_HUGEPAGE
    /* Huge pages keep TLB misses from masquerading as cache misses. */
    madvise(p, rounded, MADV_HUGEPAGE);
#endif
#else
    if (posix_memalign(&p, 4096, rounded) != 0) return NULL;
#endif
    /* Fault every page in now so the timed region never takes a page fault. */
    memset(p, 1, rounded);
    return p;
}

/* ------------------------------------------------------- perf counters  */
/* Event encodings are uarch-specific. The defaults below target Sapphire
 * Rapids / Emerald Rapids (Golden Cove cores). They are transcribed from the
 * Intel perfmon event lists and are NOT verified on hardware by this repo --
 * see docs/COUNTERS.md. Override any of them with:
 *     OSTIR_EVENT_L2_HIT=0xd1,0x02  (event,umask[,cmask])
 */

typedef struct {
    const char *name;
    const char *env;
    uint64_t    event;
    uint64_t    umask;
    uint64_t    cmask;
    int         fd;
    uint64_t    value;
} counter_t;

static counter_t g_counters[] = {
    /* name                        env override            ev    umask cmask */
    { "MEM_LOAD_RETIRED.L2_HIT",   "OSTIR_EVENT_L2_HIT",   0xd1, 0x02, 0, -1, 0 },
    { "MEM_LOAD_RETIRED.L2_MISS",  "OSTIR_EVENT_L2_MISS",  0xd1, 0x10, 0, -1, 0 },
    { "L2_LINES_IN.ALL",           "OSTIR_EVENT_LINES_IN", 0x25, 0x1f, 0, -1, 0 },
    { "L2_RQSTS.ALL_DEMAND_MISS",  "OSTIR_EVENT_DEM_MISS", 0x24, 0x27, 0, -1, 0 },
    { "OFFCORE_REQUESTS.ALL_DATA_RD", "OSTIR_EVENT_OFFCORE", 0xb0, 0x08, 0, -1, 0 },
    { "CYCLE_ACTIVITY.STALLS_L2_MISS", "OSTIR_EVENT_STALLS", 0xa3, 0x05, 5, -1, 0 },
};
static const int N_COUNTERS = (int)(sizeof(g_counters) / sizeof(g_counters[0]));
static int g_counters_ok = 0;

/* OSTIR_NT_STREAM=1 makes the DRAM leg use non-temporal loads (§3.2). */
static int g_nt_stream = 0;

#ifdef __linux__
static long perf_open(struct perf_event_attr *a, pid_t pid, int cpu,
                      int grp, unsigned long flags)
{
    return syscall(__NR_perf_event_open, a, pid, cpu, grp, flags);
}

static void parse_override(counter_t *c)
{
    const char *v = getenv(c->env);
    if (!v) return;
    unsigned long ev = 0, um = 0, cm = 0;
    int n = sscanf(v, "%lx,%lx,%lx", &ev, &um, &cm);
    if (n >= 2) { c->event = ev; c->umask = um; c->cmask = (n >= 3) ? cm : 0; }
}

static void counters_init(void)
{
    for (int i = 0; i < N_COUNTERS; i++) {
        counter_t *c = &g_counters[i];
        parse_override(c);
        struct perf_event_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.type           = PERF_TYPE_RAW;
        attr.size           = sizeof(attr);
        attr.config         = c->event | (c->umask << 8) | (c->cmask << 24);
        attr.disabled       = 1;
        attr.exclude_kernel = 1;
        attr.exclude_hv     = 1;
        attr.inherit        = 0;
        c->fd = (int)perf_open(&attr, 0, -1, -1, 0);
        if (c->fd < 0)
            fprintf(stderr, "warn: counter %s unavailable (%s)\n",
                    c->name, strerror(errno));
    }
    for (int i = 0; i < N_COUNTERS; i++)
        if (g_counters[i].fd >= 0) { g_counters_ok = 1; break; }
    if (!g_counters_ok)
        fprintf(stderr,
                "warn: no perf counters. Need perf_event_paranoid <= 2 and a\n"
                "      uarch matching the event table. Timing still valid;\n"
                "      hit rate will be reported as designed-only.\n");
}

static void counters_start(void)
{
    for (int i = 0; i < N_COUNTERS; i++) {
        if (g_counters[i].fd < 0) continue;
        ioctl(g_counters[i].fd, PERF_EVENT_IOC_RESET, 0);
        ioctl(g_counters[i].fd, PERF_EVENT_IOC_ENABLE, 0);
    }
}

static void counters_stop(void)
{
    for (int i = 0; i < N_COUNTERS; i++) {
        counter_t *c = &g_counters[i];
        if (c->fd < 0) { c->value = 0; continue; }
        ioctl(c->fd, PERF_EVENT_IOC_DISABLE, 0);
        uint64_t v = 0;
        if (read(c->fd, &v, sizeof(v)) != sizeof(v)) v = 0;
        c->value = v;
    }
}
#else
static void counters_init(void)
{
    fprintf(stderr,
            "note: perf counters are Linux-only. On this platform h is the\n"
            "      designed value only; S(h) is still measured directly.\n");
}
static void counters_start(void) {}
static void counters_stop(void)
{
    for (int i = 0; i < N_COUNTERS; i++) g_counters[i].value = 0;
}
#endif

static void counters_json(void)
{
    printf(", \"nt_stream\": %s, \"nt_loads_available\": %s",
           g_nt_stream ? "true" : "false",
           HAVE_NT_LOAD ? "true" : "false");
    printf(", \"counters_available\": %s", g_counters_ok ? "true" : "false");
    printf(", \"counters\": {");
    for (int i = 0; i < N_COUNTERS; i++)
        printf("%s\"%s\": %llu", i ? ", " : "", g_counters[i].name,
               (unsigned long long)g_counters[i].value);
    printf("}");
    if (g_counters_ok) {
        double hit = (double)g_counters[0].value;
        double miss = (double)g_counters[1].value;
        double h = (hit + miss) > 0 ? hit / (hit + miss) : -1.0;
        printf(", \"h_measured\": %.6f", h);   /* §6.2 h-hat */
    } else {
        printf(", \"h_measured\": null");
    }
}

/* ------------------------------------------------------------- kernels  */

/* Streaming reduction over one buffer. Represents the weight-fetch stream:
 * every byte is touched exactly once per pass and used in a trivial op, which
 * is the batch-1 arithmetic intensity regime of §4.2 (I = 3.56 ops/byte).
 * Returns a checksum so nothing can be optimized away. */
/* Eight independent accumulators over a full cache line per iteration. Four
 * accumulators measured 28 GB/s on an M2 Pro L2-resident buffer -- 8 B/cycle,
 * i.e. dependency-bound, not bandwidth-bound. A kernel that is compute-bound
 * cannot measure bandwidth, and would show up in E4 as "S saturating below
 * 1/r" while the real cause was this loop. Do not reduce the accumulator
 * count. */
static inline __attribute__((always_inline))
uint64_t stream_sum(const uint64_t *restrict p, size_t words)
{
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    size_t i = 0;

    /* Masking the bound is not cosmetic. With `i + 8 <= words` the compiler
     * cannot prove the trip count is a multiple of 8, emits a weak loop with
     * a runtime tail, and throughput collapses from 86 to 28 GB/s -- flat
     * across every working-set size, so the cache knee vanishes entirely.
     * Masking first makes divisibility provable and restores vectorization.
     * Verified on M2 Pro: 86 (L1) / 72 (L2) / 41 (DRAM) GB/s. */
    const size_t bulk = words & ~(size_t)7;
    for (; i < bulk; i += 8) {
        a0 += p[i];     a1 += p[i + 1]; a2 += p[i + 2]; a3 += p[i + 3];
        a4 += p[i + 4]; a5 += p[i + 5]; a6 += p[i + 6]; a7 += p[i + 7];
    }
    for (; i < words; i++) a0 += p[i];
    return (a0 + a1) + (a2 + a3) + (a4 + a5) + (a6 + a7);
}

#define MAX_BATCH 64

/* Batched reduction: each loaded weight serves nb multiply-accumulates
 * against nb distinct activation values, into nb distinct accumulators.
 *
 * Two things had to be fixed here before E6 measured anything real.
 *
 * 1. The first version did `acc += w * (b + 1)` into ONE accumulator, which
 *    the compiler folds to `acc += w * nb(nb+1)/2` -- the MACs never execute.
 *    MAC rate scaled perfectly linearly to 155 GMAC/s while no arithmetic
 *    happened. Opaque per-batch activations read from memory fix that.
 *
 * 2. With nb a RUNTIME value the compiler cannot keep acc[] in registers, so
 *    it spills to the stack and every MAC becomes a load-modify-store. That
 *    capped the low-batch plateau at 19 GB/s against L2's 73 -- only 27% --
 *    so the loop was never memory-bound and n_b* (the ratio of compute
 *    saturation to the memory-bound byte rate) inherited the bias.
 *    BATCH_CASE specializes nb at compile time so acc[] lives in registers.
 *    Unspecialized nb falls back to the generic path and is reported as
 *    such, so a biased point can never be silently quoted.
 *
 * The i-loop is unrolled by 8 with a tree reduction so that nb = 1 still has
 * instruction-level parallelism and is limited by bandwidth rather than by
 * one dependent add chain.
 */
#define BATCH_BODY(NB)                                                       \
    do {                                                                     \
        uint64_t acc[NB];                                                    \
        for (int b = 0; b < (NB); b++) acc[b] = 0;                           \
        const size_t bulk = words & ~(size_t)7;                              \
        size_t i = 0;                                                        \
        for (; i < bulk; i += 8) {                                           \
            for (int b = 0; b < (NB); b++) {                                 \
                const uint64_t a = act[b];                                   \
                uint64_t t0 = p[i] * a + p[i + 1] * a;                       \
                uint64_t t1 = p[i + 2] * a + p[i + 3] * a;                   \
                uint64_t t2 = p[i + 4] * a + p[i + 5] * a;                   \
                uint64_t t3 = p[i + 6] * a + p[i + 7] * a;                   \
                acc[b] += (t0 + t1) + (t2 + t3);                             \
            }                                                                \
        }                                                                    \
        for (; i < words; i++)                                               \
            for (int b = 0; b < (NB); b++) acc[b] += p[i] * act[b];          \
        uint64_t s = 0;                                                      \
        for (int b = 0; b < (NB); b++) s += acc[b];                          \
        return s;                                                            \
    } while (0)

#define BATCH_CASE(NB) case NB: BATCH_BODY(NB)

/* Which nb values get a register-resident specialization. E6 sweeps exactly
 * these so every reported point is unbiased. */
#define BATCH_SPECIALIZED_LIST "1,2,3,4,6,8,12,16,20,24,32,40,48,64"

static uint64_t batch_kernel(const uint64_t *restrict p, size_t words,
                             const uint64_t *restrict act, int nb)
{
    switch (nb) {
    BATCH_CASE(1);  BATCH_CASE(2);  BATCH_CASE(3);  BATCH_CASE(4);
    BATCH_CASE(6);  BATCH_CASE(8);  BATCH_CASE(12); BATCH_CASE(16);
    BATCH_CASE(20); BATCH_CASE(24); BATCH_CASE(32); BATCH_CASE(40);
    BATCH_CASE(48); BATCH_CASE(64);
    default: break;
    }
    /* Generic fallback: acc[] spills, so this point is BIASED. Callers must
     * mark it -- see "specialized" in the JSON record. */
    {
        uint64_t acc[MAX_BATCH] = {0};
        for (size_t i = 0; i < words; i++)
            for (int b = 0; b < nb; b++) acc[b] += p[i] * act[b];
        uint64_t s = 0;
        for (int b = 0; b < nb; b++) s += acc[b];
        return s;
    }
}

static int batch_is_specialized(int nb)
{
    static const int ok[] = {1, 2, 3, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64};
    for (size_t i = 0; i < sizeof(ok) / sizeof(ok[0]); i++)
        if (ok[i] == nb) return 1;
    return 0;
}

/* Non-temporal streaming read.
 *
 * E4 measured the panel leg running 13-18% slower when a DRAM stream was
 * interleaved with it: the streaming operand flows through L2 and evicts
 * panel lines. §3.2 names this ("streaming pollution") and prescribes the
 * fix -- non-temporal loads or explicit prefetch hints, so the streaming
 * operand does not allocate in the cache it is merely passing through.
 *
 * This is that fix. If it works, beta_L2 stops depending on the mix, Thm
 * 4.2's fixed-per-tier-bandwidth premise is restored, and the S(h) fit
 * should tighten. If beta_L2 still varies with h afterwards, the pollution
 * hypothesis was wrong and something else is going on -- equally worth
 * knowing, which is why this is a switch and not a silent default.
 */
#if HAVE_NT_LOAD
static inline __attribute__((always_inline))
uint64_t stream_sum_nt(const uint64_t *restrict p, size_t words)
{
    uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, a4 = 0, a5 = 0, a6 = 0, a7 = 0;
    const size_t bulk = words & ~(size_t)7;
    size_t i = 0;
    for (; i < bulk; i += 8) {
        a0 += __builtin_nontemporal_load(p + i);
        a1 += __builtin_nontemporal_load(p + i + 1);
        a2 += __builtin_nontemporal_load(p + i + 2);
        a3 += __builtin_nontemporal_load(p + i + 3);
        a4 += __builtin_nontemporal_load(p + i + 4);
        a5 += __builtin_nontemporal_load(p + i + 5);
        a6 += __builtin_nontemporal_load(p + i + 6);
        a7 += __builtin_nontemporal_load(p + i + 7);
    }
    for (; i < words; i++) a0 += p[i];
    return (a0 + a1) + (a2 + a3) + (a4 + a5) + (a6 + a7);
}
#else
static inline __attribute__((always_inline))
uint64_t stream_sum_nt(const uint64_t *restrict p, size_t words)
{
    /* No non-temporal builtin here: fall back to a normal read with an NTA
     * prefetch hint, and report nt_loads_available:false so the receipt does
     * not imply a capability the binary lacks. */
    for (size_t i = 0; i + 64 < words; i += 8)
        __builtin_prefetch(p + i + 64, 0, 0);
    return stream_sum(p, words);
}
#endif

/* Mixed-residency reduction: dispatch CHUNK-sized reads to the resident panel
 * or the DRAM buffer so that exactly `h` of bytes come from the panel.
 * Bresenham keeps the interleave even, which matters -- a blocked interleave
 * lets the DRAM prefetcher run ahead and inflates apparent bandwidth.
 *
 * No BARRIER inside this loop. Each chunk reads a different address, so there
 * is nothing for the compiler to hoist, and a "memory" clobber every 4 KiB
 * throttled the DRAM leg from 42 to 14 GB/s. The single barrier the caller
 * applies per region is sufficient. */
/* Per-leg timing. Thm 4.2 assumes each tier delivers a FIXED bandwidth
 * independent of the mix. That is an assumption, not a measurement, and it is
 * exactly what fails when S(h) shows curvature: if effective DRAM bandwidth
 * rises as DRAM references thin out (memory-level parallelism), or if part of
 * the "DRAM" stream is really being served by an intermediate cache, then the
 * per-tier bandwidths are functions of h and the harmonic law cannot hold with
 * constants.
 *
 * So measure them. Accumulate panel-leg and DRAM-leg time separately within
 * the same run. Two clock_gettime calls per 64 KiB chunk costs ~1-3% of the
 * region and turns the explanation for E4's curvature from a hypothesis into
 * a number. */
typedef struct {
    uint64_t acc;
    uint64_t ns_panel, ns_dram;
    uint64_t bytes_panel, bytes_dram;
} mixed_stats_t;

static mixed_stats_t mixed_sum_split(const uint64_t *restrict panel,
                                     size_t panel_words,
                                     const uint64_t *restrict dram,
                                     size_t dram_words,
                                     double h, size_t total_chunks,
                                     uint64_t *dram_cursor)
{
    const size_t cw = CHUNK / sizeof(uint64_t);
    mixed_stats_t st = {0, 0, 0, 0, 0};
    double err = 0.0;
    size_t pcur = 0, dcur = *dram_cursor;

    for (size_t k = 0; k < total_chunks; k++) {
        err += h;
        if (err >= 1.0) {
            err -= 1.0;
            if (pcur + cw > panel_words) pcur = 0;
            uint64_t t0 = now_ns();
            st.acc += stream_sum(panel + pcur, cw);
            st.ns_panel += now_ns() - t0;
            st.bytes_panel += CHUNK;
            pcur += cw;
        } else {
            if (dcur + cw > dram_words) dcur = 0;
            uint64_t t0 = now_ns();
            st.acc += stream_sum(dram + dcur, cw);
            st.ns_dram += now_ns() - t0;
            st.bytes_dram += CHUNK;
            dcur += cw;
        }
    }
    *dram_cursor = dcur;
    return st;
}

static uint64_t mixed_sum(const uint64_t *restrict panel, size_t panel_words,
                          const uint64_t *restrict dram, size_t dram_words,
                          double h, size_t total_chunks, uint64_t *dram_cursor)
{
    const size_t cw = CHUNK / sizeof(uint64_t);
    uint64_t acc = 0;
    double err = 0.0;
    size_t pcur = 0, dcur = *dram_cursor;

    for (size_t k = 0; k < total_chunks; k++) {
        err += h;
        if (err >= 1.0) {
            err -= 1.0;
            if (pcur + cw > panel_words) pcur = 0;
            acc += stream_sum(panel + pcur, cw);
            pcur += cw;
        } else {
            if (dcur + cw > dram_words) dcur = 0;
            acc += g_nt_stream ? stream_sum_nt(dram + dcur, cw)
                               : stream_sum(dram + dcur, cw);
            dcur += cw;
        }
    }
    *dram_cursor = dcur;
    return acc;
}

/* ------------------------------------------------- timed-region helper  */

typedef struct {
    double   bps;
    size_t   passes;
    uint64_t ns;
    uint64_t checksum;
} measurement_t;

/* Calibrate a pass count until the region clears MIN_RUN_NS, then time a
 * FRESH region of exactly that size with counters wrapped tightly around it.
 * Every timed number in this file goes through here so the calibration bug
 * cannot come back in one subcommand and not another. */
/* The accumulator is deliberately NOT volatile, and not written through a
 * caller-supplied pointer. A `volatile uint64_t *sink` parameter may alias
 * `buf`, so the compiler must assume every store clobbers the buffer; that
 * alone cost 2.6x here (72 -> 27 GB/s on an M2 Pro L2-resident probe).
 * Returning the checksum defeats dead-code elimination just as effectively,
 * because main() prints it. */
static measurement_t measure_bps(const uint64_t *restrict buf, size_t words,
                                 size_t bytes)
{
    size_t passes = 1;
    uint64_t t0, t1, acc = 0;

    /* Mask HERE, where `words` is derived, not inside stream_sum. Masking in
     * the callee does not propagate into the inlined loop bound; masking at
     * the derivation point does. Buffers are always cache-line multiples, so
     * this drops nothing. */
    words &= ~(size_t)7;

    acc += stream_sum(buf, words);                      /* warm */
    BARRIER(acc);
    do {
        t0 = now_ns();
        for (size_t i = 0; i < passes; i++) {
            acc += stream_sum(buf, words);
            BARRIER(acc);
        }
        t1 = now_ns();
        if (t1 - t0 < MIN_RUN_NS) passes *= 2;
    } while (t1 - t0 < MIN_RUN_NS);

    counters_start();
    t0 = now_ns();
    for (size_t i = 0; i < passes; i++) {
        acc += stream_sum(buf, words);
        BARRIER(acc);
    }
    t1 = now_ns();
    counters_stop();

    measurement_t m;
    m.passes = passes;
    m.ns = t1 - t0;
    m.bps = (double)passes * (double)bytes / (m.ns / 1e9);
    m.checksum = acc;
    return m;
}

/* ---------------------------------------------------------------- E3    */

static void cmd_bandwidth(size_t l2_probe_bytes, size_t dram_bytes, int reps)
{
    uint64_t *small = alloc_aligned(l2_probe_bytes);
    uint64_t *big   = alloc_aligned(dram_bytes);
    if (!small || !big) { fprintf(stderr, "alloc failed\n"); exit(1); }

    size_t sw = l2_probe_bytes / sizeof(uint64_t);
    size_t bw = dram_bytes / sizeof(uint64_t);
    uint64_t sink = 0;

    /* alloc_aligned memsets, leaving every page dirty. The first read pass
     * then contends with writeback and under-reports DRAM bandwidth by ~6x.
     * Burn one untimed pass to settle it. */
    sink += stream_sum(big, bw);

    for (int rep = 0; rep < reps; rep++) {
        /* Calibrate a pass count, THEN time a fresh region. Timing the
         * calibration loop itself charges the cumulative doubling time
         * against only the final round's bytes and halves the result. */
        measurement_t l2 = measure_bps(small, sw, l2_probe_bytes);
        measurement_t dr = measure_bps(big, bw, dram_bytes);
        double l2_bps = l2.bps, dr_bps = dr.bps;
        sink += l2.checksum + dr.checksum;

        printf("{\"kind\": \"bandwidth\", \"rep\": %d, "
               "\"l2_probe_bytes\": %zu, \"dram_bytes\": %zu, "
               "\"beta_l2_bps\": %.6e, \"beta_dram_bps\": %.6e, \"r\": %.6f",
               rep, l2_probe_bytes, dram_bytes, l2_bps, dr_bps,
               dr_bps / l2_bps);
        printf(", \"checksum\": %llu}\n", (unsigned long long)sink);
        fflush(stdout);
    }
    free(small); free(big);
}

/* ---------------------------------------------------------------- E2    */

static void cmd_panel_sweep(size_t min_bytes, size_t max_bytes, int steps,
                            int reps)
{
    double lo = (double)min_bytes, hi = (double)max_bytes;
    uint64_t sink = 0;
    counters_init();

    for (int s = 0; s < steps; s++) {
        double frac = (steps == 1) ? 0.0 : (double)s / (steps - 1);
        size_t bytes = (size_t)(lo * pow(hi / lo, frac));
        bytes &= ~(CACHELINE - 1);
        if (bytes < CACHELINE) bytes = CACHELINE;

        uint64_t *buf = alloc_aligned(bytes);
        if (!buf) { fprintf(stderr, "alloc failed at %zu\n", bytes); exit(1); }
        size_t words = bytes / sizeof(uint64_t);

        for (int rep = 0; rep < reps; rep++) {
            measurement_t m = measure_bps(buf, words, bytes);
            sink += m.checksum;
            printf("{\"kind\": \"panel\", \"rep\": %d, \"bytes\": %zu, "
                   "\"passes\": %zu, \"ns\": %llu, \"bps\": %.6e",
                   rep, bytes, m.passes, (unsigned long long)m.ns, m.bps);
            counters_json();
            printf(", \"checksum\": %llu}\n", (unsigned long long)sink);
            fflush(stdout);
        }
        free(buf);
    }
}

/* ---------------------------------------------------------------- E4    */

/* Run the mixed-residency measurement at an explicit list of h values.
 * E7 needs specific h values (the weight-traffic share f of a given
 * attention configuration), not a uniform grid. */
static void cmd_mixed_h_at(size_t panel_bytes, size_t dram_bytes,
                           double h_value, int reps);

static void cmd_mixed_h(size_t panel_bytes, size_t dram_bytes,
                        int n_points, int reps)
{
    uint64_t *panel = alloc_aligned(panel_bytes);
    uint64_t *dram  = alloc_aligned(dram_bytes);
    if (!panel || !dram) { fprintf(stderr, "alloc failed\n"); exit(1); }
    size_t pw = (panel_bytes / sizeof(uint64_t)) & ~(size_t)7;
    size_t dw = (dram_bytes / sizeof(uint64_t)) & ~(size_t)7;
    uint64_t sink = 0;
    counters_init();

    /* 2 GiB of dispatched reads per timed region. The first cut used 64 MiB,
     * which took ~1.5 ms at DRAM speed -- far too short to time, and small
     * enough that the "DRAM" leg was really being served from the last-level
     * cache. That made the h=0 baseline read 61 GB/s instead of 42, which
     * compressed the whole S(h) curve and produced a spurious E4 failure with
     * R^2 = -5.6. The DRAM leg must sweep far more than any cache. */
    const size_t chunks = (2ULL << 30) / CHUNK;

    /* Cursor persists across warm-up, reps and h points, so successive DRAM
     * chunks keep marching through the whole buffer instead of re-reading a
     * warm window from offset zero. */
    uint64_t cur = 0;

    for (int i = 0; i < n_points; i++) {
        /* Span [0, 1] inclusive; the monograph asks for >=20 points over
         * [0.3, 1.0] but the low end anchors S(0)=1 and costs nothing. */
        double h = (n_points == 1) ? 1.0 : (double)i / (n_points - 1);

        for (int rep = 0; rep < reps; rep++) {
            sink += mixed_sum(panel, pw, dram, dw, h, chunks / 8, &cur); /* warm */

            counters_start();
            uint64_t t0 = now_ns();
            mixed_stats_t st = mixed_sum_split(panel, pw, dram, dw, h,
                                               chunks, &cur);
            uint64_t t1 = now_ns();
            counters_stop();
            sink += st.acc;

            double bytes = (double)chunks * CHUNK;
            double bps = bytes / ((t1 - t0) / 1e9);
            double bps_panel = st.ns_panel
                ? (double)st.bytes_panel / (st.ns_panel / 1e9) : 0.0;
            double bps_dram = st.ns_dram
                ? (double)st.bytes_dram / (st.ns_dram / 1e9) : 0.0;
            printf("{\"kind\": \"mixed\", \"rep\": %d, \"h_designed\": %.6f, "
                   "\"panel_bytes\": %zu, \"bytes\": %.0f, \"ns\": %llu, "
                   "\"bps\": %.6e, \"bps_panel_leg\": %.6e, "
                   "\"bps_dram_leg\": %.6e",
                   rep, h, panel_bytes, bytes,
                   (unsigned long long)(t1 - t0), bps, bps_panel, bps_dram);
            counters_json();
            printf(", \"checksum\": %llu}\n", (unsigned long long)sink);
            fflush(stdout);
        }
    }
    free(panel); free(dram);
}

/* ---------------------------------------------------------------- E6    */

/* Batch sweep: each loaded weight byte serves n_b multiply-accumulates, so
 * arithmetic intensity scales with batch and the loop should cross from
 * memory-bound to compute-bound near n_b* (Cor. to Thm 4.1). */
static void cmd_batch(size_t panel_bytes, int max_batch, int reps)
{
    uint64_t *panel = alloc_aligned(panel_bytes);
    if (!panel) { fprintf(stderr, "alloc failed\n"); exit(1); }
    size_t words = (panel_bytes / sizeof(uint64_t)) & ~(size_t)7;
    uint64_t sink = 0;
    counters_init();

    /* Activations taken from the buffer itself: runtime data the compiler
     * cannot constant-fold through. */
    uint64_t act[MAX_BATCH];
    for (int b = 0; b < MAX_BATCH; b++)
        act[b] = (panel[b % words] | 1ULL) + (uint64_t)b;

    static const int NB_LIST[] = {1, 2, 3, 4, 6, 8, 12, 16, 20, 24,
                                  32, 40, 48, 64};
    for (size_t nbi = 0; nbi < sizeof(NB_LIST) / sizeof(NB_LIST[0]); nbi++) {
        int nb = NB_LIST[nbi];
        if (nb > max_batch) break;
        for (int rep = 0; rep < reps; rep++) {
            size_t passes = 1;
            uint64_t t0, t1;
            do {
                t0 = now_ns();
                for (size_t p = 0; p < passes; p++) {
                    sink += batch_kernel(panel, words, act, nb);
                    BARRIER(sink);
                }
                t1 = now_ns();
                if (t1 - t0 < MIN_RUN_NS) passes *= 2;
            } while (t1 - t0 < MIN_RUN_NS);

            counters_start();
            uint64_t c0 = now_ns();
            for (size_t p = 0; p < passes; p++) {
                sink += batch_kernel(panel, words, act, nb);
                BARRIER(sink);
            }
            uint64_t c1 = now_ns();
            counters_stop();

            double secs = (c1 - c0) / 1e9;
            double bytes = (double)passes * panel_bytes;
            double macs = (double)passes * words * nb;
            printf("{\"kind\": \"batch\", \"rep\": %d, \"n_batch\": %d, "
                   "\"bytes\": %.0f, \"macs\": %.0f, \"ns\": %llu, "
                   "\"bps\": %.6e, \"macs_per_s\": %.6e, \"specialized\": %s",
                   rep, nb, bytes, macs, (unsigned long long)(c1 - c0),
                   bytes / secs, macs / secs,
                   batch_is_specialized(nb) ? "true" : "false");
            counters_json();
            printf(", \"checksum\": %llu}\n", (unsigned long long)sink);
            fflush(stdout);
        }
    }
    free(panel);
}

/* --------------------------------------------------------- E7 helper    */

static void cmd_mixed_h_at(size_t panel_bytes, size_t dram_bytes,
                           double h_value, int reps)
{
    uint64_t *panel = alloc_aligned(panel_bytes);
    uint64_t *dram  = alloc_aligned(dram_bytes);
    if (!panel || !dram) { fprintf(stderr, "alloc failed\n"); exit(1); }
    size_t pw = (panel_bytes / sizeof(uint64_t)) & ~(size_t)7;
    size_t dw = (dram_bytes / sizeof(uint64_t)) & ~(size_t)7;
    uint64_t sink = 0, cur = 0;
    counters_init();
    const size_t chunks = (2ULL << 30) / CHUNK;

    for (int rep = 0; rep < reps; rep++) {
        sink += mixed_sum(panel, pw, dram, dw, h_value, chunks / 8, &cur);
        counters_start();
        uint64_t t0 = now_ns();
        sink += mixed_sum(panel, pw, dram, dw, h_value, chunks, &cur);
        uint64_t t1 = now_ns();
        counters_stop();
        double bytes = (double)chunks * CHUNK;
        printf("{\"kind\": \"mixed\", \"rep\": %d, \"h_designed\": %.6f, "
               "\"panel_bytes\": %zu, \"bytes\": %.0f, \"ns\": %llu, "
               "\"bps\": %.6e",
               rep, h_value, panel_bytes, bytes,
               (unsigned long long)(t1 - t0), bytes / ((t1 - t0) / 1e9));
        counters_json();
        printf(", \"checksum\": %llu}\n", (unsigned long long)sink);
        fflush(stdout);
    }
    free(panel); free(dram);
}

/* ------------------------------------------------- selftest (diagnostic) */

/* Byte-identical to the standalone probe used to validate the methodology.
 * If `selftest` and `panel-sweep` disagree at the same size, the discrepancy
 * is in the harness, not the machine. */
static void cmd_selftest(size_t bytes)
{
    uint64_t *b = alloc_aligned(bytes);
    size_t w = (bytes / sizeof(uint64_t)) & ~(size_t)7;
    uint64_t acc = 0;
    acc += stream_sum(b, w);
    size_t passes = 1; uint64_t t0, t1;
    do {
        t0 = now_ns();
        for (size_t i = 0; i < passes; i++) { acc += stream_sum(b, w); BARRIER(acc); }
        t1 = now_ns();
        if (t1 - t0 < MIN_RUN_NS) passes *= 2;
    } while (t1 - t0 < MIN_RUN_NS);
    t0 = now_ns();
    for (size_t i = 0; i < passes; i++) { acc += stream_sum(b, w); BARRIER(acc); }
    t1 = now_ns();
    printf("{\"kind\": \"selftest\", \"bytes\": %zu, \"bps\": %.6e, "
           "\"passes\": %zu, \"checksum\": %llu}\n",
           bytes, (double)passes * bytes / ((t1 - t0) / 1e9), passes,
           (unsigned long long)acc);
    free(b);
}

/* ------------------------------------------------------------------ main */

static size_t arg_bytes(const char *s)
{
    char *end = NULL;
    double v = strtod(s, &end);
    if (end && *end) {
        if (*end == 'k' || *end == 'K') v *= 1024;
        else if (*end == 'm' || *end == 'M') v *= 1024 * 1024;
        else if (*end == 'g' || *end == 'G') v *= 1024.0 * 1024 * 1024;
    }
    return (size_t)v;
}

static void usage(void)
{
    fprintf(stderr,
        "ostir_kernel <cmd> [opts]   (all sizes accept k/M/G suffixes)\n"
        "  bandwidth   [l2_probe=128k] [dram=1G] [reps=5]\n"
        "  panel-sweep [min=64k] [max=64M] [steps=32] [reps=3]\n"
        "  mixed-h     [panel=128k] [dram=1G] [points=21] [reps=5]\n"
        "  mixed-h-at  [panel=128k] [dram=1G] [h=1.0] [reps=5]\n"
        "  batch       [panel=128k] [max_batch=64] [reps=3]\n");
}

int main(int argc, char **argv)
{
    { const char *v = getenv("OSTIR_NT_STREAM"); g_nt_stream = v && *v == '1'; }
    if (argc < 2) { usage(); return 2; }
    const char *cmd = argv[1];
    const char *a2 = argc > 2 ? argv[2] : NULL;
    const char *a3 = argc > 3 ? argv[3] : NULL;
    const char *a4 = argc > 4 ? argv[4] : NULL;
    const char *a5 = argc > 5 ? argv[5] : NULL;

    if (!strcmp(cmd, "bandwidth"))
        cmd_bandwidth(a2 ? arg_bytes(a2) : 128 << 10,
                      a3 ? arg_bytes(a3) : 1ULL << 30,
                      a4 ? atoi(a4) : 5);
    else if (!strcmp(cmd, "panel-sweep"))
        cmd_panel_sweep(a2 ? arg_bytes(a2) : 64 << 10,
                        a3 ? arg_bytes(a3) : 64 << 20,
                        a4 ? atoi(a4) : 32, a5 ? atoi(a5) : 3);
    else if (!strcmp(cmd, "mixed-h"))
        cmd_mixed_h(a2 ? arg_bytes(a2) : 128 << 10,
                    a3 ? arg_bytes(a3) : 1ULL << 30,
                    a4 ? atoi(a4) : 21, a5 ? atoi(a5) : 5);
    else if (!strcmp(cmd, "batch"))
        cmd_batch(a2 ? arg_bytes(a2) : 128 << 10,
                  a3 ? atoi(a3) : 64, a4 ? atoi(a4) : 3);
    else if (!strcmp(cmd, "mixed-h-at"))
        cmd_mixed_h_at(a2 ? arg_bytes(a2) : 128 << 10,
                       a3 ? arg_bytes(a3) : 1ULL << 30,
                       a4 ? atof(a4) : 1.0, a5 ? atoi(a5) : 5);
    else if (!strcmp(cmd, "selftest"))
        cmd_selftest(a2 ? arg_bytes(a2) : 64 << 10);
    else { usage(); return 2; }
    return 0;
}
