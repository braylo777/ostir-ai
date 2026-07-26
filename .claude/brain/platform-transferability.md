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

**The premise, measured -- and one retraction.** E4 times each leg
separately, so the fixed-per-tier-bandwidth assumption is tested rather than
assumed.

I first measured the panel leg 13-18% slower when interleaved and attributed
it to §3.2 streaming pollution. That was WRONG: it was contention from an IDE
extension host at 134% CPU plus a browser. On a quiet machine the panel leg is
flat to 5.5%. **Check `uptime` before believing any bandwidth result on a
desktop OS** -- this cost a published-to-the-repo wrong conclusion.

The real violation is on the DRAM side: beta_DRAM rises 42 -> 58 GB/s (27-34%)
as h increases, monotonically, while beta_L2 stays flat. That is memory-level
parallelism -- thinner DRAM reference streams overlap better. A single r
cannot describe it, so E4's R^2 failure is a premise failure, not a refutation
of the harmonic form.

Do NOT try to "validate" the harmonic law from the per-leg decomposition:
given per-leg times it is an algebraic identity and proves nothing.

Related: [[monograph-errata]], [[measure-what-you-think-you-measure]].
