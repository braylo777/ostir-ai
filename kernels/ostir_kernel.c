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

#define CHUNK        4096ULL           /* dispatch granularity for mixed-h   */
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

/* Mixed-residency reduction: dispatch CHUNK-sized reads to the resident panel
 * or the DRAM buffer so that exactly `h` of bytes come from the panel.
 * Bresenham keeps the interleave even, which matters -- a blocked interleave
 * lets the DRAM prefetcher run ahead and inflates apparent bandwidth. */
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
            BARRIER(acc);
            pcur += cw;
        } else {
            if (dcur + cw > dram_words) dcur = 0;
            acc += stream_sum(dram + dcur, cw);
            BARRIER(acc);
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

static void cmd_mixed_h(size_t panel_bytes, size_t dram_bytes,
                        int n_points, int reps)
{
    uint64_t *panel = alloc_aligned(panel_bytes);
    uint64_t *dram  = alloc_aligned(dram_bytes);
    if (!panel || !dram) { fprintf(stderr, "alloc failed\n"); exit(1); }
    size_t pw = panel_bytes / sizeof(uint64_t);
    size_t dw = dram_bytes / sizeof(uint64_t);
    uint64_t sink = 0;
    counters_init();

    /* Enough chunks that each timed region clears MIN_RUN_NS comfortably. */
    const size_t chunks = (64ULL << 20) / CHUNK;

    for (int i = 0; i < n_points; i++) {
        /* Span [0, 1] inclusive; the monograph asks for >=20 points over
         * [0.3, 1.0] but the low end anchors S(0)=1 and costs nothing. */
        double h = (n_points == 1) ? 1.0 : (double)i / (n_points - 1);

        for (int rep = 0; rep < reps; rep++) {
            uint64_t cur = 0;
            sink += mixed_sum(panel, pw, dram, dw, h, chunks / 8, &cur); /* warm */

            cur = 0;
            counters_start();
            uint64_t t0 = now_ns();
            sink += mixed_sum(panel, pw, dram, dw, h, chunks, &cur);
            uint64_t t1 = now_ns();
            counters_stop();

            double bytes = (double)chunks * CHUNK;
            double bps = bytes / ((t1 - t0) / 1e9);
            printf("{\"kind\": \"mixed\", \"rep\": %d, \"h_designed\": %.6f, "
                   "\"panel_bytes\": %zu, \"bytes\": %.0f, \"ns\": %llu, "
                   "\"bps\": %.6e",
                   rep, h, panel_bytes, bytes,
                   (unsigned long long)(t1 - t0), bps);
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

    for (int nb = 1; nb <= max_batch; nb = (nb < 4) ? nb + 1 : (nb * 3) / 2) {
        for (int rep = 0; rep < reps; rep++) {
            size_t passes = 1;
            uint64_t t0, t1;
            do {
                t0 = now_ns();
                for (size_t p = 0; p < passes; p++) {
                    uint64_t acc = 0;
                    for (size_t i = 0; i < words; i++) {
                        uint64_t w = panel[i];          /* one load ...      */
                        for (int b = 0; b < nb; b++)    /* ... nb MACs on it */
                            acc += w * (uint64_t)(b + 1);
                    }
                    sink += acc;
                    BARRIER(sink);
                }
                t1 = now_ns();
                if (t1 - t0 < MIN_RUN_NS) passes *= 2;
            } while (t1 - t0 < MIN_RUN_NS);

            counters_start();
            uint64_t c0 = now_ns();
            for (size_t p = 0; p < passes; p++) {
                uint64_t acc = 0;
                for (size_t i = 0; i < words; i++) {
                    uint64_t w = panel[i];
                    for (int b = 0; b < nb; b++) acc += w * (uint64_t)(b + 1);
                }
                sink += acc;
                BARRIER(sink);
            }
            uint64_t c1 = now_ns();
            counters_stop();

            double secs = (c1 - c0) / 1e9;
            double bytes = (double)passes * panel_bytes;
            double macs = (double)passes * words * nb;
            printf("{\"kind\": \"batch\", \"rep\": %d, \"n_batch\": %d, "
                   "\"bytes\": %.0f, \"macs\": %.0f, \"ns\": %llu, "
                   "\"bps\": %.6e, \"macs_per_s\": %.6e",
                   rep, nb, bytes, macs, (unsigned long long)(c1 - c0),
                   bytes / secs, macs / secs);
            counters_json();
            printf(", \"checksum\": %llu}\n", (unsigned long long)sink);
            fflush(stdout);
        }
    }
    free(panel);
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
        "  batch       [panel=128k] [max_batch=64] [reps=3]\n");
}

int main(int argc, char **argv)
{
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
    else if (!strcmp(cmd, "selftest"))
        cmd_selftest(a2 ? arg_bytes(a2) : 64 << 10);
    else { usage(); return 2; }
    return 0;
}
