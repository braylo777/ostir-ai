# Performance counters — status and caveats

**The event encodings in `kernels/ostir_kernel.c` are transcribed from the
Intel perfmon lists and have NOT been verified on hardware by this repo.** No
Sapphire Rapids machine was available during development. Verify them before
trusting any counter-derived number, especially `h`.

## Events (§6.2)

| Quantity | Event | Default encoding (ev, umask, cmask) |
|---|---|---|
| L2 hits | `MEM_LOAD_RETIRED.L2_HIT` | 0xd1, 0x02 |
| L2 misses | `MEM_LOAD_RETIRED.L2_MISS` | 0xd1, 0x10 |
| Lines filled | `L2_LINES_IN.ALL` | 0x25, 0x1f |
| Demand misses | `L2_RQSTS.ALL_DEMAND_MISS` | 0x24, 0x27 |
| Offcore reads | `OFFCORE_REQUESTS.ALL_DATA_RD` | 0xb0, 0x08 |
| Stall cycles | `CYCLE_ACTIVITY.STALLS_L2_MISS` | 0xa3, 0x05, cmask 5 |

`h_hat = L2_HIT / (L2_HIT + L2_MISS)`.

The `MEM_LOAD_RETIRED.*` encodings are stable across Skylake through Golden
Cove. `L2_LINES_IN.ALL` is the one most likely to be wrong: it is 0xf1/0x1f on
Skylake and 0x25/0x1f on later cores. `L2_RQSTS` umasks have also shifted.

## Verifying before use

Cross-check every event against `perf` on the actual target:

```bash
perf list | grep -i l2_
perf stat -e mem_load_retired.l2_hit,mem_load_retired.l2_miss \
          -e l2_lines_in.all,offcore_requests.all_data_rd \
          ./kernels/bin/ostir_kernel panel-sweep 64k 8M 8 1
```

Compare against the harness's own counter output for the same run. If they
disagree, override without recompiling:

```bash
export OSTIR_EVENT_LINES_IN=0xf1,0x1f     # event,umask[,cmask], hex
export OSTIR_EVENT_L2_HIT=0xd1,0x02
python3 run_all.py --only E2 E4
```

## The independent check that does not need counters

E4 drives `h` as an independent variable: each dispatch goes to a
known-resident panel or a known-DRAM buffer, so the *designed* `h` is known a
priori. Where counters exist, E4 compares designed against measured `h` and
fails if the mean absolute deviation exceeds 0.05.

That comparison is the real value of the design. It validates the counter
methodology of §6.2 rather than assuming it — if the event encodings are wrong
for the host uarch, the cross-check catches it instead of silently producing a
plausible, wrong `h`. Run E4 on the target **before** trusting E2's `h_hat`.

## Permissions

`perf_event_open` needs `kernel.perf_event_paranoid <= 2` for user-space
counting (`<= 0` is safest here):

```bash
echo 0 | sudo tee /proc/sys/kernel/perf_event_paranoid
```

Inside containers add `--cap-add=PERFMON` (or `CAP_SYS_ADMIN` on older
kernels). Most cloud VMs expose no PMU at all; check before renting one — on
those instances the harness runs and reports `counters_available: false`, and
`h` is designed-only.

## AMX

E6's `n_b* ~ 18` derives from AMX INT8 peak (`TDPBSSD`, 2048 ops/cycle). The
current batch kernel is a portable scalar loop, not an AMX kernel, so its knee
is not comparable — and on hosts where the low-batch plateau fails to reach
`beta_L2`, E6 flags its own result as biased. AMX also requires explicit
kernel permission before use:

```c
arch_prctl(ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA);
```

A trustworthy E6 needs that inner kernel written. It is the largest remaining
gap in the harness.
