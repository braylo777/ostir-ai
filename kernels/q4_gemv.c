/* q4_gemv — the measurement the rest of the harness could not make.
 *
 * Everything in E1-E7 validates the MODEL. None of it demonstrates the
 * IMPROVEMENT, because a reduced bit rate only becomes reduced time if there
 * is a kernel that keeps weights packed and dequantizes inside the inner
 * loop. Dequantizing to fp32 up front (which is what experiments/e5 does, on
 * purpose, to measure accuracy) moves exactly as many bytes as fp32 did.
 *
 * This file is that kernel, in the batch-1 decode regime the monograph cares
 * about: y[m] = sum_k W[m][k] * x[k], each weight loaded once and used once.
 *
 * Three weight formats over identical logical shapes:
 *   fp32   32.000 bits/weight   baseline
 *   fp16   16.000 bits/weight   the honest comparison point for llama.cpp-era work
 *   q4     4.500 bits/weight    b=4, G=64, fp16 scale + fp16 zero (Def 1.1)
 *
 * The question it answers: does the 3.56x byte reduction from fp16 -> q4
 * show up as a 3.56x time reduction, or does the unpack eat it?
 *
 * Key optimization, and the reason a naive q4 kernel looks bad: with s and z
 * constant across a group,
 *     sum_i (s*q_i + z) * x_i  =  s * sum_i q_i*x_i  +  z * sum_i x_i
 * so the per-group work is one integer-ish dot product plus a precomputed
 * activation sum. sum_i x_i depends only on the group's k-range, so it is
 * hoisted out of the row loop entirely. Without this the kernel does two
 * FMAs per weight instead of one and is compute-bound on nonsense.
 *
 * Build: make q4     (see Makefile)
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <time.h>
#include <math.h>

#if defined(__aarch64__)
#include <arm_neon.h>
#define HAVE_NEON 1
#else
#define HAVE_NEON 0
#endif

#define GROUP        64          /* weights per quantization group (Def 1.1) */
#define MIN_RUN_NS   300000000ULL
#define BARRIER(x) __asm__ __volatile__("" :: "r"(x) : "memory")

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void *xalloc(size_t bytes)
{
    void *p = NULL;
    size_t r = (bytes + 4095) & ~(size_t)4095;
    if (posix_memalign(&p, 4096, r) != 0) { fprintf(stderr, "oom\n"); exit(1); }
    memset(p, 0, r);
    return p;
}

/* ------------------------------------------------------------ fp32 GEMV */

/* NEON-vectorized. The first version of this file left fp32 and fp16 as
 * scalar C while q4 used intrinsics; both baselines came out at ~2.7
 * Gweights/s (0.77 FMA/cycle), and the resulting "3x speedup for q4" was
 * mostly measuring hand-vectorization against compiler-scalar code. A
 * bit-rate comparison is only meaningful when every format gets the same
 * quality of kernel. */
static void gemv_fp32(const float *restrict W, const float *restrict x,
                      float *restrict y, size_t M, size_t K)
{
    for (size_t m = 0; m < M; m++) {
        const float *w = W + m * K;
#if HAVE_NEON
        float32x4_t v0 = vdupq_n_f32(0), v1 = vdupq_n_f32(0);
        float32x4_t v2 = vdupq_n_f32(0), v3 = vdupq_n_f32(0);
        size_t k = 0;
        for (; k + 16 <= K; k += 16) {
            v0 = vfmaq_f32(v0, vld1q_f32(w + k),      vld1q_f32(x + k));
            v1 = vfmaq_f32(v1, vld1q_f32(w + k + 4),  vld1q_f32(x + k + 4));
            v2 = vfmaq_f32(v2, vld1q_f32(w + k + 8),  vld1q_f32(x + k + 8));
            v3 = vfmaq_f32(v3, vld1q_f32(w + k + 12), vld1q_f32(x + k + 12));
        }
        float acc = vaddvq_f32(vaddq_f32(vaddq_f32(v0, v1), vaddq_f32(v2, v3)));
        for (; k < K; k++) acc += w[k] * x[k];
        y[m] = acc;
#else
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        size_t k = 0;
        for (; k + 4 <= K; k += 4) {
            a0 += w[k] * x[k];         a1 += w[k + 1] * x[k + 1];
            a2 += w[k + 2] * x[k + 2]; a3 += w[k + 3] * x[k + 3];
        }
        for (; k < K; k++) a0 += w[k] * x[k];
        y[m] = (a0 + a1) + (a2 + a3);
#endif
    }
}

/* ------------------------------------------------------------ fp16 GEMV */

static void gemv_fp16(const __fp16 *restrict W, const float *restrict x,
                      float *restrict y, size_t M, size_t K)
{
    for (size_t m = 0; m < M; m++) {
        const __fp16 *w = W + m * K;
#if HAVE_NEON
        float32x4_t v0 = vdupq_n_f32(0), v1 = vdupq_n_f32(0);
        float32x4_t v2 = vdupq_n_f32(0), v3 = vdupq_n_f32(0);
        size_t k = 0;
        for (; k + 16 <= K; k += 16) {
            float16x8_t h0 = vld1q_f16(w + k);
            float16x8_t h1 = vld1q_f16(w + k + 8);
            v0 = vfmaq_f32(v0, vcvt_f32_f16(vget_low_f16(h0)),  vld1q_f32(x + k));
            v1 = vfmaq_f32(v1, vcvt_f32_f16(vget_high_f16(h0)), vld1q_f32(x + k + 4));
            v2 = vfmaq_f32(v2, vcvt_f32_f16(vget_low_f16(h1)),  vld1q_f32(x + k + 8));
            v3 = vfmaq_f32(v3, vcvt_f32_f16(vget_high_f16(h1)), vld1q_f32(x + k + 12));
        }
        float acc = vaddvq_f32(vaddq_f32(vaddq_f32(v0, v1), vaddq_f32(v2, v3)));
        for (; k < K; k++) acc += (float)w[k] * x[k];
        y[m] = acc;
#else
        float a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        size_t k = 0;
        for (; k + 4 <= K; k += 4) {
            a0 += (float)w[k] * x[k];         a1 += (float)w[k + 1] * x[k + 1];
            a2 += (float)w[k + 2] * x[k + 2]; a3 += (float)w[k + 3] * x[k + 3];
        }
        for (; k < K; k++) a0 += (float)w[k] * x[k];
        y[m] = (a0 + a1) + (a2 + a3);
#endif
    }
}

/* -------------------------------------------------------------- q4 GEMV */
/* Layout per group of 64 weights: 32 bytes of packed nibbles (low nibble =
 * even index, high nibble = odd index), then fp16 scale, then fp16 zero.
 * That is 64*4 + 32 = 288 bits / 64 weights = 4.500 bpw exactly (Def 1.1). */

typedef struct { uint8_t q[GROUP / 2]; __fp16 s; __fp16 z; } q4_group_t;

static void gemv_q4(const q4_group_t *restrict W, const float *restrict x,
                    const float *restrict xsum, float *restrict y,
                    size_t M, size_t K)
{
    const size_t G = K / GROUP;
    for (size_t m = 0; m < M; m++) {
        const q4_group_t *g = W + m * G;
        float acc = 0.0f;
        for (size_t gi = 0; gi < G; gi++, g++) {
            const uint8_t *q = g->q;
            const float *xx = x + gi * GROUP;
            const float s = (float)g->s;
            const float z = (float)g->z;
#if HAVE_NEON
            float32x4_t v0 = vdupq_n_f32(0.0f), v1 = vdupq_n_f32(0.0f);
            for (int i = 0; i < GROUP / 2; i += 8) {
                /* 8 packed bytes -> 16 nibbles -> 16 weights */
                uint8x8_t b = vld1_u8(q + i);
                uint8x8_t lo = vand_u8(b, vdup_n_u8(0x0F));
                uint8x8_t hi = vshr_n_u8(b, 4);
                /* interleave so lane order matches weight order */
                uint8x8x2_t zip = vzip_u8(lo, hi);
                uint16x8_t w0 = vmovl_u8(zip.val[0]);
                uint16x8_t w1 = vmovl_u8(zip.val[1]);
                float32x4_t f0 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w0)));
                float32x4_t f1 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w0)));
                float32x4_t f2 = vcvtq_f32_u32(vmovl_u16(vget_low_u16(w1)));
                float32x4_t f3 = vcvtq_f32_u32(vmovl_u16(vget_high_u16(w1)));
                const float *xp = xx + i * 2;
                v0 = vfmaq_f32(v0, f0, vld1q_f32(xp));
                v1 = vfmaq_f32(v1, f1, vld1q_f32(xp + 4));
                v0 = vfmaq_f32(v0, f2, vld1q_f32(xp + 8));
                v1 = vfmaq_f32(v1, f3, vld1q_f32(xp + 12));
            }
            float qdot = vaddvq_f32(vaddq_f32(v0, v1));
#else
            float qdot = 0.0f;
            for (int i = 0; i < GROUP / 2; i++) {
                qdot += (float)(q[i] & 0x0F) * xx[2 * i];
                qdot += (float)(q[i] >> 4) * xx[2 * i + 1];
            }
#endif
            /* sum_i (s*q_i + z)*x_i = s*qdot + z*sum_i x_i */
            acc += s * qdot + z * xsum[gi];
        }
        y[m] = acc;
    }
}

/* ----------------------------------------------------------------- main */

static double timed(void (*run)(void *), void *ctx, double bytes,
                    size_t *passes_out)
{
    size_t passes = 1;
    uint64_t t0, t1;
    run(ctx);
    do {
        t0 = now_ns();
        for (size_t i = 0; i < passes; i++) run(ctx);
        t1 = now_ns();
        if (t1 - t0 < MIN_RUN_NS) passes *= 2;
    } while (t1 - t0 < MIN_RUN_NS);

    t0 = now_ns();
    for (size_t i = 0; i < passes; i++) run(ctx);
    t1 = now_ns();
    *passes_out = passes;
    return (double)passes * bytes / ((t1 - t0) / 1e9);
}

static size_t g_M, g_K;
static float *g_x, *g_xsum, *g_y;
static float *g_w32;
static __fp16 *g_w16;
static q4_group_t *g_wq4;

static void run32(void *c) { (void)c; gemv_fp32(g_w32, g_x, g_y, g_M, g_K); BARRIER(g_y[0]); }
static void run16(void *c) { (void)c; gemv_fp16(g_w16, g_x, g_y, g_M, g_K); BARRIER(g_y[0]); }
static void runq4(void *c) { (void)c; gemv_q4(g_wq4, g_x, g_xsum, g_y, g_M, g_K); BARRIER(g_y[0]); }

int main(int argc, char **argv)
{
    g_M = argc > 1 ? (size_t)atol(argv[1]) : 4096;
    g_K = argc > 2 ? (size_t)atol(argv[2]) : 4096;
    if (g_K % GROUP) { fprintf(stderr, "K must be a multiple of %d\n", GROUP); return 2; }

    const size_t N = g_M * g_K;
    const size_t G = g_K / GROUP;

    g_x    = xalloc(g_K * sizeof(float));
    g_xsum = xalloc(G * sizeof(float));
    g_y    = xalloc(g_M * sizeof(float));
    g_w32  = xalloc(N * sizeof(float));
    g_w16  = xalloc(N * sizeof(__fp16));
    g_wq4  = xalloc(g_M * G * sizeof(q4_group_t));

    /* Deterministic pseudo-random content; values are irrelevant to timing
     * but must not be constant, or the compiler and the DRAM page policy
     * both get to cheat. */
    uint64_t st = 88172645463325252ULL;
    #define NEXT() (st ^= st << 13, st ^= st >> 7, st ^= st << 17, st)
    for (size_t k = 0; k < g_K; k++) g_x[k] = (float)((NEXT() >> 40) % 1000) / 500.0f - 1.0f;
    for (size_t gi = 0; gi < G; gi++) {
        float s = 0; for (int i = 0; i < GROUP; i++) s += g_x[gi * GROUP + i];
        g_xsum[gi] = s;
    }
    for (size_t i = 0; i < N; i++) {
        float v = (float)((NEXT() >> 40) % 1000) / 500.0f - 1.0f;
        g_w32[i] = v; g_w16[i] = (__fp16)v;
    }
    for (size_t m = 0; m < g_M; m++)
        for (size_t gi = 0; gi < G; gi++) {
            q4_group_t *g = &g_wq4[m * G + gi];
            const float *w = g_w32 + m * g_K + gi * GROUP;
            float lo = w[0], hi = w[0];
            for (int i = 1; i < GROUP; i++) { if (w[i] < lo) lo = w[i]; if (w[i] > hi) hi = w[i]; }
            float s = (hi - lo) / 15.0f; if (s <= 0) s = 1.0f;
            g->s = (__fp16)s; g->z = (__fp16)lo;
            for (int i = 0; i < GROUP / 2; i++) {
                int a = (int)lrintf((w[2 * i] - lo) / s);      if (a < 0) a = 0; if (a > 15) a = 15;
                int b = (int)lrintf((w[2 * i + 1] - lo) / s);  if (b < 0) b = 0; if (b > 15) b = 15;
                g->q[i] = (uint8_t)(a | (b << 4));
            }
        }

    const double B32 = (double)N * 4.0;
    const double B16 = (double)N * 2.0;
    const double BQ4 = (double)(g_M * G) * sizeof(q4_group_t);

    size_t p;
    double bps32 = timed(run32, NULL, B32, &p);
    double bps16 = timed(run16, NULL, B16, &p);
    double bpsq4 = timed(runq4, NULL, BQ4, &p);

    /* Weights processed per second is the format-independent figure of merit:
     * how fast the machine can push through the model, regardless of how many
     * bytes each weight costs. */
    double w32 = bps32 / 4.0, w16 = bps16 / 2.0;
    double wq4 = bpsq4 / (sizeof(q4_group_t) / (double)GROUP);

    printf("{\"kind\": \"q4_gemv\", \"M\": %zu, \"K\": %zu, \"weights\": %zu, "
           "\"neon\": %s, \"bpw_q4\": %.4f,\n",
           g_M, g_K, N, HAVE_NEON ? "true" : "false",
           sizeof(q4_group_t) * 8.0 / GROUP);
    printf("  \"fp32\": {\"bytes\": %.0f, \"GBps\": %.3f, \"Gweights_s\": %.3f},\n",
           B32, bps32 / 1e9, w32 / 1e9);
    printf("  \"fp16\": {\"bytes\": %.0f, \"GBps\": %.3f, \"Gweights_s\": %.3f},\n",
           B16, bps16 / 1e9, w16 / 1e9);
    printf("  \"q4\":   {\"bytes\": %.0f, \"GBps\": %.3f, \"Gweights_s\": %.3f},\n",
           BQ4, bpsq4 / 1e9, wq4 / 1e9);
    printf("  \"speedup_q4_vs_fp16\": %.3f, \"speedup_q4_vs_fp32\": %.3f,\n",
           wq4 / w16, wq4 / w32);
    printf("  \"byte_ratio_fp16_over_q4\": %.3f, \"efficiency_vs_byte_bound\": %.3f,\n",
           B16 / BQ4, (wq4 / w16) / (B16 / BQ4));
    printf("  \"tok_s_7B_q4\": %.2f, \"tok_s_7B_fp16\": %.2f,\n",
           wq4 / 7e9, w16 / 7e9);

    /* Roofline crossover. Each format runs at
     *     weights/s = min( beta_DRAM / bytes_per_weight , pi_format )
     * where pi_format is the compute-limited rate measured above (valid when
     * the format is NOT memory-bound on this host). q4 beats fp16 only where
     * fp16 has gone memory-bound below q4's compute ceiling, i.e. where
     *     beta_DRAM / 2  <  pi_q4     =>     beta_DRAM < 2 * pi_q4
     * Below that threshold the byte saving is real time; above it, the extra
     * unpack work is pure cost. This is the number that decides whether the
     * whole thesis applies to a given machine. */
    const double bpw_q4 = sizeof(q4_group_t) / (double)GROUP;   /* bytes */
    double beta_crossover = 2.0 * wq4;                 /* vs fp16 */
    double beta_crossover32 = 4.0 * wq4;               /* vs fp32 */
    printf("  \"pi_fp32_Gw_s\": %.3f, \"pi_fp16_Gw_s\": %.3f, \"pi_q4_Gw_s\": %.3f,\n",
           w32 / 1e9, w16 / 1e9, wq4 / 1e9);
    printf("  \"beta_dram_needed_to_saturate_fp16_GBps\": %.2f,\n", w16 * 2.0 / 1e9);
    printf("  \"beta_dram_needed_to_saturate_q4_GBps\": %.2f,\n", wq4 * bpw_q4 / 1e9);
    printf("  \"crossover_beta_vs_fp16_GBps\": %.2f, \"crossover_beta_vs_fp32_GBps\": %.2f,\n",
           beta_crossover / 1e9, beta_crossover32 / 1e9);
    printf("  \"verdict\": \"q4 beats fp16 only on hosts with per-core DRAM "
           "bandwidth below %.1f GB/s; this host measures ~41 GB/s\",\n",
           beta_crossover / 1e9);
    printf("  \"projected_at_sapphire_rapids_8.5GBps\": {"
           "\"fp16_Gw_s\": %.3f, \"q4_Gw_s\": %.3f, \"speedup\": %.2f},\n",
           fmin(8.5e9 / 2.0, w16) / 1e9,
           fmin(8.5e9 / bpw_q4, wq4) / 1e9,
           fmin(8.5e9 / bpw_q4, wq4) / fmin(8.5e9 / 2.0, w16));
    printf("  \"checksum\": %.4f}\n", (double)g_y[0]);
    return 0;
}
