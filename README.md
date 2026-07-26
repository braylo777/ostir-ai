# OSTIR validation harness

Executable form of Part VI of *Cache-Resident Quantized Inference on
General-Purpose CPUs* (`docs/monograph.md`): the quantizers, the perf-counter
instrumentation, and the `S(h)` fit, plus E1–E7 with the monograph's own
pass/fail thresholds.

Every experiment writes a JSON receipt to `results/` carrying the checks, the
machine it ran on, and any caveat that limits how the number should be read.

```bash
python3 tests/test_theory.py         # 23 closed-form invariants
python3 run_all.py                   # E1-E7 in §6.4 dependency order
python3 run_all.py --only E3 E4      # a subset
python3 run_all.py --model Qwen/Qwen2.5-0.5B   # full E5 accuracy pipeline
```

The C kernel builds automatically on first use (`make -C kernels`).

For E5 `--model`, install the pinned stack into a **native** interpreter:

```bash
/usr/bin/python3 -m venv .venv          # macOS: NOT a Rosetta Anaconda python
.venv/bin/pip install -r requirements-e5.txt
```

On Apple Silicon check `python -c "import platform;print(platform.machine())"`
returns `arm64`. A Rosetta x86_64 interpreter reports the wrong ISA and runs
torch emulated. If HuggingFace downloads stall (they did here, repeatedly, at
a few MB/s), pre-fetch once and then run with `HF_HUB_OFFLINE=1
HF_DATASETS_OFFLINE=1`.

## Layout

```
ostir/
  rate.py        Part I    B_eff, phi, N_max, hierarchical rates
  quantize.py    §1.1, Alg 2, Alg 3   flat / hierarchical / outlier extraction
  distortion.py  Part II   Table 2.1, Bennett, Panter-Dite, optimal clipping
  residency.py   Parts III-IV  S(h), Amdahl, KV footprint, Thm 3.4 traffic
  solver.py      Alg 1     SolveResidentConfig
  fit.py         E4        the no-free-parameter S(h) fit and its diagnostics
  kernel.py      driver for the C kernel; cache topology detection
  evalsuite.py   E5        WikiText-2 / MMLU / GSM8K
  report.py      receipts, gates, tables
kernels/
  ostir_kernel.c bandwidth / panel-sweep / mixed-h / batch, + perf counters
experiments/     e1..e7
docs/            monograph.md, PLATFORM.md, COUNTERS.md
```

## The one design departure from §6.3

The monograph obtains E4's `h` values by varying `m_c*k_c` around the E2 knee
and reading `h` from hardware counters. This harness *also* drives `h`
directly: every dispatch goes to either a known-resident panel or a
known-DRAM buffer, under a Bresenham accumulator, so `h` is an independent
variable rather than an observation.

Two reasons. It makes the fit a real test of the functional form on machines
with no PMU. And where counters do exist, comparing designed `h` against
measured `h` validates the §6.2 counter methodology instead of assuming it —
if an event encoding is wrong for the host uarch, that shows up as a
disagreement rather than as a plausible, wrong `h`.

## Findings so far

Reproducible from the receipts in `results/`. See `docs/PLATFORM.md` for what
does and does not transfer off this host.

**Confirmed.** The Part I algebra is exact — every operating point, Q4_K at
exactly 4.500 bpw, Prop 1.7's scatter collinear at `N = 3,613,812` weights,
Thm 2.2's elasticity identity, the 12.5% exhaustion bound. Prop 3.1's charge
that the deck implies `eta = 0.97` is arithmetically right. Thm 2.8's
hierarchical advantage measured +1.16 dB against a predicted +1.00. Prop
2.11's `32p` rate penalty is exact. Cor 4.4's `f` reproduces to 4 decimals.

**Three corrections to the monograph.**

1. *Appendix B has two wrong entries, and they hit the headline operating
   point.* Exact integration of `E[max]` matches 9 of 11 tabulated values to
   <0.001, but `n=64` (2.320 vs 2.3437) and `n=128` (2.580 vs 2.5946) are
   wrong. Those are the values Table 2.1 uses for G=64 — the deck's Path B.
   The true `D(4,64)` is 0.008138, not 0.007982: Path B is ~2% more lossy
   than stated.

2. *Cor 4.4 does not reproduce from its own inputs.* With `S_w = 11.7647` and
   the monograph's own `f`, composition gives **3.83x (MHA) and 7.33x
   (GQA-8)**, not 3.86x and 7.15x. The printed figures imply `f = 0.8098` and
   `f = 0.9400`. The headline should be **3.8–7.3x** — slightly worse for MHA,
   slightly better for GQA-8.

3. *Thm 2.9 is false as stated.* Per-group Lloyd–Max beat grouped min–max on
   clean Gaussian draws, where (A2) holds, by 2.9 dB at G=32 and 1.8 dB at
   G=64. That is expected: Lloyd–Max is MSE-optimal for a given distribution
   and level count, so a per-group Lloyd–Max codebook is *both* adaptive and
   optimally shaped. The monograph's proof compares grouping against a
   **global** companded codebook, a different object, so it does not support
   the stated claim. The conclusion survives in rate-aware form — a per-group
   codebook costs `2^b * 16 / G` bits/weight, i.e. 12.0 bpw at G=32 against
   min–max's 5.0 — and in that form it is decisive. §2.6 should assert the
   rate-aware version.

**A fourth finding: Thm 4.2's premise is measurably false — but on the DRAM
side, not the cache side.**

An earlier revision of this file attributed the effect to §3.2 streaming
pollution, measuring the panel leg 13–18% slower when interleaved. **That was
wrong** — it was contention from unrelated processes (an IDE extension host at
134% CPU, plus a browser) competing for memory bandwidth. On a quiet machine
the panel leg is flat to 5.5%, and a non-temporal-load implementation of
§3.2's prescribed fix changes nothing, because there was nothing to fix.

The real effect is on the other tier. E4 times both legs separately:

| | spread across h | verdict |
|---|---|---|
| `beta_L2` (resident panel) | 5.5–7.5% | constant, premise holds |
| `beta_DRAM` (streaming) | **27–34%**, 42 → 58 GB/s | **not constant** |

`beta_DRAM` rises monotonically with `h`: as DRAM references thin out they
overlap better and each one costs less — memory-level parallelism. Thm 4.2
composes two tiers at *constant* per-tier bandwidth, so no single `r` can
describe this system and E4's R² failure follows necessarily. **It is the
premise that fails, not the harmonic form.**

Note the harmonic law itself cannot be validated from this decomposition —
given per-leg times, `1/beta_eff = h/beta_L2 + (1-h)/beta_DR` is an algebraic
identity. The premise is the only part these measurements can independently
test, and the harness says so rather than reporting a tautology as a result.

Two consequences. On this platform the S(h) gate as written is untestable and
needs the Sapphire Rapids target. And wherever the effect exists,
counter-measured `h` alone cannot predict speedup — the value of a hit depends
on how much miss traffic runs beside it.

**E5 on real weights.** Qwen2.5-0.5B, WikiText-2, Linear weights only:

| config | bpw | PPL |
|---|---|---|
| fp16 baseline | 16 | 12.6495 |
| flat b4 G64 — the deck's **Path B** | 4.500 | 15.0333 |
| hier b4 G32 K8 — **Q4_K** | 4.500 | **14.4276** |
| hier + 0.1% outliers | 4.532 | **14.3027** |
| flat b3 G64 | 3.500 | 37.5493 |

**Thm 2.8 confirmed end-to-end:** at an identical 4.500 bpw the hierarchical
G=32 scheme beats flat G=64 by **0.61 PPL** on real weights. Outlier
extraction adds 0.12 PPL for 0.032 bits/weight. b=3 is not viable at this
scale (3x worse).

The delta-PPL < 0.15 gate **fails at +1.65**, and two things bound how far
that generalizes — neither of them the residency argument. Quantization error
at fixed bits/weight falls sharply with model size, and a sub-1B model is far
more sensitive than the 7B class the gate is written for. And Alg. 2/3 are
round-to-nearest with no calibration set and no error compensation, which is
exactly what GPTQ and AWQ exist to recover. **Re-run on a 7B with a
calibrated quantizer before treating any of this as the operating point.**

**Still open.** E4's main gate fails on this host (R^2 ~ 0.6–0.8), now with a
measured mechanism rather than a hypothesis. E5's delta-PPL gate needs
`--model`. And `r` is 0.57–0.66 here against the monograph's 0.085, so no
magnitude in Parts III–IV is reproducible on this class of hardware — only
functional forms and premises.

## Four bugs the kernel had before it could measure anything

Each produced plausible, wrong numbers rather than an obvious failure. They
are documented at their sites because they are easy to reintroduce.

- Four accumulators in the reduction: dependency-bound at 8 B/cycle.
- `t0` taken outside the calibration loop: charged cumulative doubling time
  against one round's bytes, halving every result.
- A `volatile uint64_t *sink` parameter may alias the buffer, so the compiler
  assumed every store clobbered it and stopped vectorizing — 2.6x.
- An unprovable trip count (`i + 8 <= words` instead of masking first): 3x,
  and it flattened the curve to 27 GB/s from 32 KiB to 256 MiB. **A machine
  with no cache hierarchy at all** — which reads as "residency does nothing"
  rather than as a broken benchmark.
- E6's batch loop folded `sum_b w*(b+1)` into `w*nb(nb+1)/2`, so the MACs
  never ran while MAC rate scaled linearly to 155 GMAC/s; and with `nb` a
  runtime value the accumulators spilled to the stack, capping the plateau at
  27% of `beta_L2`. `n_b*` is the ratio of those two quantities, so it
  inherited both. Compile-time specialization (`BATCH_CASE`) puts the
  accumulators back in registers — `nb=1` now measures 100% of `beta_L2`.

The last one is the cautionary case: it would have refuted the thesis on the
strength of a compiler heuristic.
