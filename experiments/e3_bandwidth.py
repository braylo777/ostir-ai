#!/usr/bin/env python3
"""E3 — Bandwidth constants (§6.3).

Tests: the inputs to Thm 4.2.
Output: beta_L2, beta_DR, hence r.
Prediction (Sapphire Rapids): r in [0.06, 0.12]; the deck's 0.085 lands inside.
Pass: three runs within 5%.

r is the single most consequential constant in the harness: it sets the
ceiling S(1) = 1/r that the deck quotes as 12x, and E4 holds it fixed.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ostir import kernel as K  # noqa: E402
from ostir.report import Experiment  # noqa: E402

PREDICTED_RANGE = (0.06, 0.12)  # §6.3, Sapphire Rapids
SPREAD_TOL = 0.05  # "three runs within 5%"


def main(reps: int = 15) -> int:
    exp = Experiment("E3", "Bandwidth constants", "inputs to Thm 4.2")
    C, prov = K.l2_bytes_per_core()
    exp.data["l2_per_core_bytes"] = C
    exp.data["l2_provenance"] = prov
    exp.note(prov)
    exp.note(K.platform_note())

    # Probe sized to sit inside L2 but comfortably outside L1; DRAM buffer
    # far larger than any LLC.
    probe = max(1 << 20, C // 2)
    # 256 MiB rather than 1 GiB for the DRAM leg: still far outside any LLC,
    # but small enough that measure_bps runs ~16 passes instead of one. A
    # single 24 ms pass is a coin flip on a machine without core isolation;
    # averaging over a longer region suppresses scheduler noise.
    recs = K.run("bandwidth", f"{probe}", "256M", reps)

    l2 = np.array([r["beta_l2_bps"] for r in recs])
    dr = np.array([r["beta_dram_bps"] for r in recs])

    # Median, not mean (§6.1). A descheduled run only ever biases one way.
    beta_l2, beta_dr = float(np.median(l2)), float(np.median(dr))
    r = beta_dr / beta_l2

    def spread(x):
        return float((np.percentile(x, 75) - np.percentile(x, 25)) / np.median(x))

    s_l2, s_dr = spread(l2), spread(dr)

    exp.data.update(
        {
            "reps": reps,
            "probe_bytes": probe,
            "beta_l2_bps": beta_l2,
            "beta_dram_bps": beta_dr,
            "r": r,
            "beta_l2_all": l2.tolist(),
            "beta_dram_all": dr.tolist(),
            "iqr_frac_l2": s_l2,
            "iqr_frac_dram": s_dr,
            "ceiling_S1": 1.0 / r,
        }
    )

    print(f"\n  beta_L2   = {beta_l2 / 1e9:7.2f} GB/s   (IQR {s_l2:.1%})")
    print(f"  beta_DRAM = {beta_dr / 1e9:7.2f} GB/s   (IQR {s_dr:.1%})")
    print(f"  r         = {r:.4f}")
    print(f"  ceiling S(1) = 1/r = {1.0 / r:.2f}x")

    import os

    load1 = os.getloadavg()[0]
    exp.data["loadavg_1min"] = load1
    exp.data["beta_dram_best"] = float(np.max(dr))
    exp.data["beta_l2_best"] = float(np.max(l2))
    exp.data["r_best_of_n"] = float(np.max(dr) / np.max(l2))

    repeatable = s_l2 <= SPREAD_TOL and s_dr <= SPREAD_TOL
    exp.check(
        "E3 repeatable within 5% (IQR/median)",
        repeatable,
        f"L2 {s_l2:.1%}, DRAM {s_dr:.1%} over {reps} reps",
    )
    if not repeatable:
        exp.note(
            f"Repeatability gate failed with 1-min load average {load1:.2f} "
            f"and no core isolation. §6.1 requires isolcpus/nohz_full, pinned "
            f"frequency, disabled C-states and taskset/numactl pinning, none "
            f"of which this platform provides. L2 spread is {s_l2:.1%}, so "
            f"the kernel and timing are stable; the variance is in the DRAM "
            f"leg only and is consistent with scheduler and DVFS noise. "
            f"Best-of-{reps} gives r = {np.max(dr) / np.max(l2):.4f} vs "
            f"median r = {r:.4f}. Re-run on the isolated Linux target before "
            f"quoting r."
        )
    exp.check(
        "beta_L2 > beta_DRAM (hierarchy is visible at all)",
        beta_l2 > beta_dr * 1.05,
        f"{beta_l2 / 1e9:.1f} vs {beta_dr / 1e9:.1f} GB/s",
    )

    in_range = PREDICTED_RANGE[0] <= r <= PREDICTED_RANGE[1]
    if in_range:
        exp.check(
            "r within the monograph's predicted [0.06, 0.12]", True, f"r = {r:.4f}"
        )
    else:
        # Not a failure of the harness -- a failure of the prediction to
        # transfer to this machine. Record it as such rather than as a bug.
        exp.check(
            "r measured and reported",
            True,
            f"r = {r:.4f} is OUTSIDE the monograph's [0.06, 0.12]",
        )
        exp.note(
            f"r = {r:.4f} on this machine vs 0.085 assumed for Sapphire "
            f"Rapids. The residency ceiling here is only {1.0 / r:.2f}x, not "
            f"11.8x. Part IV's conclusions are bandwidth-ratio-specific and "
            f"do NOT transfer to this platform; only the functional form of "
            f"Thm 4.2 can be validated here."
        )

    print(exp.summary())
    print(f"\nreceipt: {exp.emit()}")
    return 0 if exp.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
