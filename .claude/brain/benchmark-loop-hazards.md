# Benchmark loop hazards that silently flatten the cache curve

Four separate bugs in `kernels/ostir_kernel.c` each produced plausible but
wrong bandwidth numbers. None crashed; all had to be caught by disbelieving
the result.

1. **Too few accumulators.** 4 independent accumulators in the reduction is
   dependency-bound at ~8 B/cycle. Use 8.
2. **Timing the calibration loop.** Taking `t0` before the pass-doubling loop
   charges cumulative time against the final round's bytes only — every
   result halves. Calibrate, then time a *fresh* region. `measure_bps()` is
   the single place this is done.
3. **`volatile` as a DCE guard.** A `volatile uint64_t *sink` parameter may
   alias the read buffer, so the compiler must assume each store clobbers it
   and stops vectorizing. Cost 2.6x (72 -> 27 GB/s). Return the checksum and
   print it instead; use an asm barrier where a per-pass guard is needed.
4. **Unprovable trip count.** `for (i + 8 <= words; i += 8)` leaves the
   compiler unable to prove divisibility; it emits a weak loop with a runtime
   tail. Cost 3x AND flattened the curve to 27 GB/s from 32 KiB to 256 MiB —
   a machine with no cache hierarchy. Mask **at the point `words` is
   derived** (`words &= ~7`), not inside the callee; masking in the callee
   does not propagate into the inlined loop bound.

**Diagnostic that works:** a flat bandwidth curve across working-set sizes is
almost never the machine. Bisect by building a 30-line standalone probe and
comparing against the same code inside the real binary; when they disagree,
the difference is the harness. See [[measure-what-you-think-you-measure]].
