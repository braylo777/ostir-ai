# Endpoint sanity checks catch the bugs that gates miss

E4's `S(h)` fit failed twice for reasons that had nothing to do with the
theory, and both were caught by checking endpoints rather than by reading
R^2.

The mixed-residency kernel must satisfy two identities by construction:
  - `h = 0` throughput  ==  E3's `beta_DRAM`
  - `h = 1` throughput  ==  E3's `beta_L2`

First failure: h=0 read 61 GB/s against a `beta_DRAM` of 42. The "DRAM" leg
was only sweeping 64 MiB per region with the cursor reset each time, so it
was being served largely from the last-level cache. R^2 came out at -5.6 and
the diagnosis pointed at Part IV, which was innocent.

Second failure: after fixing the sweep, h=0 read 14-19 GB/s — now *below*
`beta_DRAM`. Two causes: a `"memory"`-clobber barrier every 4 KiB throttled
the stream, and a 4 KiB dispatch chunk gives the prefetcher no run-up. 64 KiB
chunks with no inner barrier restored 41.3 GB/s against E3's 42.

**Rule:** before fitting anything, assert that the measurement reproduces its
own known endpoints. A fit statistic cannot distinguish "the model is wrong"
from "the instrument is wrong", but an endpoint check can.

Related: [[benchmark-loop-hazards]], [[monograph-errata]].
