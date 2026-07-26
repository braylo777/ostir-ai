# OSTIR validation harness

Executable form of Part VI of *Cache-Resident Quantized Inference on
General-Purpose CPUs* (`docs/monograph.md`): the quantizers, the perf-counter
instrumentation, and the `S(h)` fit, plus experiments E1–E7 with the
monograph's own pass/fail thresholds.

Every experiment writes a JSON receipt to `results/` carrying its checks, the
machine it ran on, and every caveat that limits how the number should be read.
The gates are meant to be believed, so some of them currently fail — see
[Status](#status).

---

## Quick start

```bash
python3 tests/test_theory.py          # 24 closed-form invariants (numpy + scipy only)
python3 run_all.py                    # E1-E7 in §6.4 dependency order
python3 run_all.py --only E3 E4       # a subset
```

The C kernel builds automatically on first use (`make -C kernels`).

E5's real-model path needs a **native** interpreter and the pinned stack:

```bash
/usr/bin/python3 -m venv .venv        # macOS: NOT a Rosetta Anaconda python
.venv/bin/pip install -r requirements-e5.txt
.venv/bin/python run_all.py --model Qwen/Qwen2.5-0.5B --e5-device mps
```

On Apple Silicon, confirm `platform.machine()` returns `arm64` — a Rosetta
x86_64 interpreter reports the wrong ISA and runs torch emulated. If
HuggingFace downloads stall, pre-fetch once, then set `HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1`.

On a Linux target:

```bash
scripts/linux-target.sh               # checks §6.1 discipline, pins a core, runs the protocol
```

---

## Status

Last full run: Apple M2 Pro, 12 cores, 16 MiB cluster-shared L2, no AMX, no
PMU, machine quiet.

| Experiment | Result | Note |
|---|---|---|
| **E1** Bit-rate verification | **PASS** 10/10 | algebra exact to the last digit |
| **E2** Residency knee | **PASS** 3/3 | knee reproducible across reps |
| **E3** Bandwidth constants | **PASS** 4/4 | `r` measured two ways |
| **E4** Residency law | **FAIL** 3/6 | premise violated — [see below](#4-thm-42s-premise-is-measurably-false--on-the-dram-side) |
| **E5** Accuracy frontier | **FAIL** 10/11 | Δ-PPL gate; all 10 theory checks pass |
| **E6** Batch threshold | **PASS** 3/3 | plateau reaches 100% of `beta_L2` |
| **E7** End-to-end Amdahl | **PASS** 2/2 | composition within 20% |

Plus 24/24 closed-form invariants.

> **Read `docs/PLATFORM.md` before quoting any number from this host.**
> `r = 0.61` here against the monograph's 0.085, so the residency ceiling is
> 1.6×, not 11.8×. No magnitude in Parts III–IV is reproducible on Apple
> Silicon — only functional forms and premises are.

---

## What this harness does *not* show

Read this before anyone quotes a speedup.

**There is no end-to-end tokens/sec measurement in this repo, on any
hardware.** E7's "composite" is synthetic memory streams standing in for
weight and KV traffic — it validates the *composition law*, not a running
model. E5 quantizes real weights but dequantizes back to fp32 before
evaluating, so it measures *accuracy only*.

The reason: **no 4-bit inference kernel exists yet.** Demonstrating the thesis
requires a GEMV that keeps weights packed at 4.5 bpw and dequantizes on the
fly in the inner loop, so a reduced bit rate actually becomes reduced memory
traffic. That kernel is the product. Building it and benchmarking against an
fp16 baseline is the largest remaining gap; until it exists this harness
validates the *model* but cannot demonstrate the *improvement*.

Also not established here:

- **The §6.2 counter methodology.** The `perf_event_open` backend is written
  but has never executed — macOS has no PMU. Event encodings in
  `kernels/ostir_kernel.c` are transcribed Intel values, unverified on
  hardware. See `docs/COUNTERS.md`.
- **AMX predictions.** `n_b* ≈ 18` follows from AMX INT8 peak; this host has
  no AMX, so its measured `n_b* = 4.5` is not comparable.

---

## Layout

```
ostir/
  rate.py        Part I        B_eff, phi, N_max, hierarchical rates
  quantize.py    §1.1, Alg 2/3 flat / hierarchical / outlier extraction
  distortion.py  Part II       Table 2.1, Bennett, Panter-Dite, optimal clipping
  residency.py   Parts III-IV  S(h), Amdahl, KV footprint, Thm 3.4 traffic
  solver.py      Alg 1         SolveResidentConfig
  fit.py         E4            the no-free-parameter S(h) fit and its diagnostics
  kernel.py                    driver for the C kernel; cache topology detection
  evalsuite.py   E5            WikiText-2 / MMLU / GSM8K
  report.py                    receipts, gates, tables
kernels/
  ostir_kernel.c               bandwidth / panel-sweep / mixed-h / batch + perf counters
experiments/                   e1_bitrate .. e7_amdahl
scripts/linux-target.sh        Linux bring-up and pinned run
docs/                          monograph.md, PLATFORM.md, COUNTERS.md
results/                       JSON receipts (gitignored; regenerate with run_all.py)
```

---

## The one design departure from §6.3

The monograph obtains E4's `h` values by varying `m_c·k_c` around the E2 knee
and reading `h` from hardware counters. This harness *also* drives `h`
directly: each dispatch goes to a known-resident panel or a known-DRAM buffer
under a Bresenham accumulator, making `h` an independent variable rather than
an observation.

Two reasons. It makes the fit a real test of the functional form on machines
with no PMU. And where counters do exist, comparing designed `h` against
measured `h` validates the §6.2 methodology instead of assuming it — a wrong
event encoding surfaces as a disagreement rather than as a plausible, wrong
`h`.

---

## Findings

All reproducible from `results/*.json`.

**Confirmed.** The Part I algebra is exact — every operating point, Q4_K at
exactly 4.500 bpw, Prop 1.7's scatter collinear at `N = 3,613,812` weights,
Thm 2.2's elasticity identity, the 12.5% exhaustion bound. Prop 3.1's charge
that the deck implies `eta = 0.97` is arithmetically correct. Prop 2.11's
`32p` rate penalty is exact. Cor 4.4's `f` reproduces to four decimals.

### Four corrections to the monograph

#### 1. Appendix B is wrong at two entries, and they hit the headline operating point

Exact integration of `E[max]` matches 9 of 11 tabulated values to <0.001, but
`n=64` (table 2.320, exact **2.3437**) and `n=128` (table 2.580, exact
**2.5946**) are wrong. Those are the values Table 2.1 uses for G=64 — the
deck's Path B. True `D(4,64) = 0.008138`, not 0.007982: **Path B is ~2% more
lossy than stated.**

#### 2. Cor 4.4 does not reproduce from its own inputs

With `S_w = 11.7647` and the monograph's own `f` values, composition gives
**3.83× (MHA) and 7.33× (GQA-8)** — not 3.86× and 7.15×. The printed figures
imply `f = 0.8098` and `f = 0.9400`, neither of which is tabulated. The
headline should be **3.8–7.3×**: slightly worse for MHA, slightly *better* for
GQA-8.

#### 3. Thm 2.9 is false as stated

Per-group Lloyd–Max beat grouped min–max on clean Gaussian draws, where (A2)
holds, by 2.9 dB at G=32 and 1.8 dB at G=64. This is expected — Lloyd–Max is
MSE-optimal for a given distribution and level count, so a per-group Lloyd–Max
codebook is *both* adaptive and optimally shaped. The monograph's proof
compares grouping against a **global** companded codebook, a different object,
so it does not support the stated claim.

The conclusion survives in **rate-aware** form and there it is decisive: a
per-group codebook costs `2^b·16/G` extra, i.e. 12.0 bpw at G=32 against
min–max's 5.0. §2.6 should assert the rate-aware version. The practical advice
— spend effort on outliers, not codebooks — is unaffected.

#### 4. Thm 4.2's premise is measurably false — on the DRAM side

E4 times each leg separately, so per-tier bandwidth is measured, not assumed:

| tier | spread across `h` | verdict |
|---|---|---|
| `beta_L2` (resident panel) | 5.5–7.5% | constant — premise holds |
| `beta_DRAM` (streaming) | **27–34%**, 42 → 58 GB/s | **not constant** |

`beta_DRAM` rises monotonically with `h`: as DRAM references thin out they
overlap better and each costs less — memory-level parallelism. Thm 4.2
composes two tiers at *constant* per-tier bandwidth, so no single `r` can
describe this system and E4's R² failure follows necessarily. **The premise
fails, not the harmonic form.**

The harmonic law itself cannot be validated from this decomposition: given
per-leg times, `1/beta_eff = h/beta_L2 + (1-h)/beta_DR` is an algebraic
identity. The premise is the only part these measurements independently test,
and the harness says so rather than reporting a tautology as a result.

> **Retraction.** An earlier revision attributed this to §3.2 streaming
> pollution, having measured the panel leg 13–18% slower when interleaved.
> That was wrong: the cause was contention from unrelated processes (an IDE
> extension host at 134% CPU). On a quiet machine the panel leg is flat, and
> implementing §3.2's prescribed non-temporal-load fix changes nothing. Check
> `uptime` before believing any bandwidth result on a desktop OS.

### E5 on real weights

Qwen2.5-0.5B, WikiText-2, `nn.Linear` weights only (the embedding is a gather,
not a GEMM, and is tied to `lm_head` in this model):

| config | bpw | PPL |
|---|---|---|
| fp16 baseline | 16 | 12.6495 |
| flat b4 G64 — the deck's **Path B** | 4.500 | 15.0333 |
| hier b4 G32 K8 — **Q4_K** | 4.500 | **14.4276** |
| hier + 0.1% outliers | 4.532 | **14.3027** |
| flat b3 G64 | 3.500 | 37.5493 |

**Thm 2.8 confirmed end-to-end:** at an identical 4.500 bpw, hierarchical
G=32 beats flat G=64 by **0.61 PPL** on real weights. Outlier extraction adds
a further 0.12 PPL for 0.032 bits/weight. b=3 is not viable at this scale.

The Δ-PPL < 0.15 gate **fails at +1.65**, bounded by two things that are not
the residency argument: a sub-1B model is far more quantization-sensitive than
the 7B class the gate targets, and Alg 2/3 are round-to-nearest with no
calibration — precisely what GPTQ and AWQ exist to recover. Re-run on a 7B
with a calibrated quantizer before treating anything here as the operating
point.

---

## Five bugs the kernel had before it could measure anything

Each produced plausible but wrong numbers rather than an obvious failure. All
are documented at their sites because all are easy to reintroduce.

1. **Four accumulators** in the reduction — dependency-bound at 8 B/cycle.
2. **`t0` outside the calibration loop** — charged cumulative doubling time
   against one round's bytes, halving every result.
3. **`volatile uint64_t *sink`** may alias the buffer, so the compiler assumed
   every store clobbered it and stopped vectorizing — 2.6×.
4. **An unprovable trip count** (`i + 8 <= words` instead of masking first) —
   3×, and it flattened the curve to 27 GB/s from 32 KiB to 256 MiB. That is
   *a machine with no cache hierarchy at all*, which reads as "residency does
   nothing" rather than as a broken benchmark.
5. **E6's batch loop** folded `Σ_b w·(b+1)` into `w·nb(nb+1)/2`, so the MACs
   never ran while MAC rate scaled linearly to 155 GMAC/s; and with `nb` a
   runtime value the accumulators spilled to the stack, capping the plateau at
   27% of `beta_L2`. Compile-time specialization put them back in registers.

Number 4 is the cautionary one: it would have refuted the thesis on the
strength of a compiler heuristic.

---

## Roadmap

| Priority | Work | Unblocks |
|---|---|---|
| 1 | 4-bit GEMV with on-the-fly dequant; tokens/sec vs fp16 | the only real speedup claim |
| 2 | Run on Sapphire Rapids / Xeon 6 with `isolcpus` | E4, E6, and every Part III–IV magnitude |
| 3 | Verify the §6.2 event encodings against `perf list` | every counter-derived number |
| 4 | E5 on a 7B model with a calibrated quantizer (GPTQ/AWQ) | the Δ-PPL operating point |
| 5 | AMX inner kernel | `n_b*` comparable to the monograph's 18 |
