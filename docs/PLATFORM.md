# Platform notes: what transfers and what does not

The monograph's Part VI targets one machine: a Sapphire Rapids / Xeon 6 class
socket, Linux, 2 MiB **private** L2 per core, AMX INT8, `perf_event` counters,
`isolcpus` + pinned frequency. Every numeric prediction in Parts III–IV is
specific to that machine.

This harness runs anywhere. That is deliberate, and it is also a trap, so the
receipts carry provenance and each experiment says explicitly which of its
gates are meaningful on the current host.

## What is machine-independent

Parts I–II are closed-form. The bit-rate algebra, the elasticity identity, the
exhaustion bound, the rate–distortion results and the quantizers are exact and
must hold identically everywhere. `tests/test_theory.py` pins them, and E1 and
E5's synthetic mode check them against real serialized bits and real tensors.

## What is machine-specific

| Quantity | Sapphire Rapids assumption | Measured on Apple M2 Pro |
|---|---|---|
| `r = beta_DR / beta_L2` | 0.085 | **0.57** |
| ceiling `S(1) = 1/r` | 11.76x | **1.74x** |
| L2 geometry | 2 MiB private/core | 16 MiB shared across 8 P-cores |
| Compute peak | AMX INT8, 2048 ops/cycle | NEON, no AMX |
| Counters | `perf_event_open` | none |
| Isolation | isolcpus, pinned freq | none available |

`r` is the consequential one. The entire residency thesis is an argument about
a bandwidth *ratio*, and Apple Silicon's unified memory makes DRAM fast
relative to cache. The M2 Pro's ceiling of 1.74x is not a refutation of the
OSTIR thesis; it is a statement that the thesis has almost no headroom on that
class of hardware, because there is not much of a cliff to fall off.

## Consequences for each experiment

- **E1** — fully valid anywhere. Pure algebra plus a serializer.
- **E2** — the knee is real, but `eta` is only comparable to Prop. 3.1 when L2
  is private per core. On a cluster-shared L2 a single-threaded sweep gets the
  whole cache, so the harness takes `eta` against the shared size and says so.
  It measured 1.25, i.e. above 1, because the M2 Pro's system-level cache
  extends residency past L2. Not an `eta` in the monograph's sense.
- **E3** — valid, but the repeatability gate (three runs within 5%) assumes
  §6.1's isolation discipline. On a loaded macOS box the DRAM leg swings 7–17%
  between runs while the L2 leg holds within 1%. Treat a marginal E3 here as
  an environment failure, not a hardware finding.
- **E4** — the functional form is testable anywhere, because the harness drives
  `h` as an independent variable rather than reading it from counters. The
  §6.2 counter methodology is **not** tested here. See below.
- **E5** — synthetic mode is fully valid. The delta-PPL gate needs `--model`.
- **E6** — `n_b* ~ 18` is an AMX number; a machine without AMX cannot confirm
  or refute it. The harness reports the measured knee, and also checks whether
  the low-batch plateau actually reached `beta_L2` — on this host it does not,
  so the knee is flagged as biased and should not be quoted.
- **E7** — the composition law is testable anywhere; the magnitude is bounded
  by the host's own `1/r`, so the monograph's 3.8–7.3x is not reproducible
  here.

## E4 on a three-tier hierarchy

Thm 4.2 assumes exactly two tiers at fixed per-tier bandwidth. On the M2 Pro
the fit lands at R^2 ~ 0.6–0.8 with residuals that arch above the model
through mid-`h` and below it at both endpoints, while a free-`r` fit recovers
`r` to within half a percent of the measured value. Scale right, shape wrong.

Two mechanisms produce that signature and both are present here: a system-level
cache between L2 and DRAM, and memory-level parallelism (as DRAM references
thin out, the remaining ones overlap better and effective DRAM bandwidth
rises). Neither is in the model. A three-tier harmonic extension is the
principled fix; until then, an E4 failure on this class of machine should not
be read as a refutation of Part IV.

## Getting a real answer

Run on the intended target:

```bash
sudo cpupower frequency-set -g performance
sudo cpupower idle-set -D 0
# kernel cmdline: isolcpus=8-15 nohz_full=8-15 rcu_nocbs=8-15
echo 1024 | sudo tee /proc/sys/vm/nr_hugepages
echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid

taskset -c 8 numactl --membind=0 python3 run_all.py
```

See `COUNTERS.md` before trusting any counter-derived number.
