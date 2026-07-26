# Working on this harness

## The one rule

**A gate that fails is a result, not a bug.** Do not loosen a threshold to
make a run go green. The value of this harness to a technical reviewer is
that its gates mean something; a table of passes obtained by moving goalposts
is worth less than an honest failure with a diagnosed cause.

If a gate fails, the acceptable responses are: fix the measurement, fix the
code under test, or record why the gate does not apply on this host. All
three are in use today — see `docs/PLATFORM.md`.

## Before you trust a number

1. `uptime`. A loaded desktop OS will fabricate a bandwidth result. One IDE
   extension host at 134% CPU once produced a 13–18% effect that was
   published as a physical finding and later retracted.
2. Check the receipt's `environment` block — the machine travels with the
   number.
3. Read the `notes` array. Anything limiting interpretation is recorded there,
   not just in the summary line.

## Editing the C kernel

`kernels/ostir_kernel.c` has five documented traps at their sites, every one
of which produced plausible but wrong numbers rather than an obvious failure.
Read those comments before changing the reduction, the timing, or the loop
bounds. After any change:

```bash
make -C kernels && ./kernels/bin/ostir_kernel selftest 32k
```

`selftest` should report roughly L1 bandwidth for a 32 KiB working set. If it
reports a flat number across sizes, the loop is issue-bound and the kernel
can no longer see the cache hierarchy it exists to measure.

## Adding an experiment

Use `ostir/report.py`'s `Experiment`. Every check needs a name, a boolean,
and a detail string carrying the measured value. Emit a receipt. Prefer
`exp.note()` over silence whenever a result is real but not comparable to the
monograph's prediction.

## Style

Match the surrounding code. Comments explain *why*, especially where a naive
change would silently break a measurement.
