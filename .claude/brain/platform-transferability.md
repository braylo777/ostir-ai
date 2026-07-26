# What of the OSTIR thesis transfers off Sapphire Rapids

The whole residency argument is about a **bandwidth ratio**
`r = beta_DRAM / beta_L2`, and `r` is not portable.

| | Sapphire Rapids (assumed) | Apple M2 Pro (measured) |
|---|---|---|
| `r` | 0.085 | **0.57** |
| ceiling `1/r` | 11.76x | **1.74x** |
| L2 | 2 MiB private/core | 16 MiB shared / 8 P-cores |

Apple's unified memory makes DRAM fast relative to cache, so there is little
cliff to fall off and cache residency buys ~1.7x at most. That is not a
refutation of the thesis; it is a statement that the thesis is about machines
with a steep memory cliff.

Two traps this created in the harness, both now guarded:

- **`eta` against the wrong denominator.** A single-threaded residency sweep
  gets the *whole* shared L2, so dividing the knee by a per-core share gave
  `eta = 9.9`. `kernel.l2_topology()` now distinguishes private from shared
  and E2 takes `eta` against `single_thread_capacity`, noting that an `eta`
  comparable to Prop 3.1 needs private L2 and one thread per core.
- **Rosetta lying about the ISA.** `platform.machine()` returns `x86_64` for
  an emulated Python on Apple Silicon, mislabelling every receipt. Use
  `sysctl -n hw.optional.arm64` (a native binary) — `kernel.true_machine()`.

**E4 on a three-tier hierarchy.** Thm 4.2 assumes exactly two tiers at fixed
per-tier bandwidth. With an SLC between L2 and DRAM the fit shows residuals
arching above the model at mid-`h` and below at both endpoints, while free-`r`
recovers `r` to <0.5%. Scale right, shape wrong — no choice of `r` fixes it.
Memory-level parallelism (effective DRAM bandwidth rising as DRAM references
thin out) produces the same signature. `fit.diagnose()` names this case
explicitly so it is not misread as refuting Part IV.

**The premise, measured.** E4 now times each leg separately, so the
fixed-per-tier-bandwidth assumption is tested rather than assumed. Result: the
panel leg runs 13% slower when interleaved with a DRAM stream (63.6 vs 73.0
GB/s). This is §3.2's streaming pollution -- the streaming operand flows
through L2 and evicts panel lines. No constant `r` can absorb a tier whose
bandwidth depends on the mix, so residual structure is guaranteed. The
composition law is fine; its premise is not. This also matters on Intel:
counter-measured `h` alone cannot predict speedup, because the cost of a hit
depends on the miss traffic running beside it.

Related: [[monograph-errata]], [[measure-what-you-think-you-measure]].
