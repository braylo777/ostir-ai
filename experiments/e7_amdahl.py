#!/usr/bin/env python3
"""E7 — End-to-end Amdahl composition (§6.3).

Tests Cor. 4.4:  S_total = [ f/S_w + (1-f)/S_kv ]^-1
Method: instrument the per-token byte split (weights vs KV) to obtain f-hat,
        predict S_total, compare to a measured composite.
Prediction: S_total in [3.5, 7.5] depending on GQA.
Pass: predicted within 20% of measured.
Deliverable: "this is the number that goes in the deck."

How the composite is measured here. Cor. 4.4 with S_kv = 1 is exactly the
harmonic mix of a fully-resident weight stream and a never-resident KV
stream. The mixed-h kernel already realizes that mix: dispatching a fraction
f of reads to the resident panel and (1-f) to a buffer far larger than any
cache IS the two-component workload, with f playing the role of h. So E7 runs
the kernel at h = f-hat for each attention configuration and checks the
measured speedup against Cor. 4.4 evaluated with S_w measured at h = 1.

This tests the composition law on real traffic. It does NOT execute a
transformer -- the weight and KV streams are synthetic. Running Cor. 4.4
against a real decode loop requires the E5 --model path plus a decode
harness, and is the remaining gap before the number is deck-ready.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ostir import kernel as K  # noqa: E402
from ostir.report import Experiment, table  # noqa: E402
from ostir.residency import amdahl_total, kv_bytes, weight_share  # noqa: E402

TOL = 0.20

# §4.4 worked example: 7B class, L=8192, 32 layers, d_h=128, 4.5/3.5 bpw.
CASES = [
    {"name": "MHA", "h_kv": 32, "f_expected": 0.807, "S_expected": 3.86},
    {"name": "GQA-8", "h_kv": 8, "f_expected": 0.944, "S_expected": 7.15},
]
N_PARAMS = 7e9
L, N_LAYERS, D_H = 8192, 32, 128
B_EFF, B_KV = 4.5, 3.5


def main(reps: int = 11) -> int:
    exp = Experiment("E7", "End-to-end Amdahl", "Cor. 4.4")
    topo = K.l2_topology()
    exp.note(topo.provenance)

    # ---- f-hat from the traffic model -----------------------------------
    rows = []
    for c in CASES:
        w = N_PARAMS * B_EFF / 8.0
        kv = kv_bytes(L, N_LAYERS, c["h_kv"], D_H, B_KV)
        f = weight_share(N_PARAMS, B_EFF, L, N_LAYERS, c["h_kv"], D_H, B_KV)
        rows.append(
            {
                "case": c["name"],
                "weights_GB": w / 1e9,
                "KV_GB": kv / 1e9,
                "f_hat": f,
                "f_monograph": c["f_expected"],
            }
        )
    print(
        f"\nPer-token traffic split (7B, L={L}, {N_LAYERS} layers, "
        f"B_eff={B_EFF}, B_kv={B_KV}):"
    )
    print(
        table(
            rows,
            ["case", "weights_GB", "KV_GB", "f_hat", "f_monograph"],
            {"weights_GB": ".3f", "KV_GB": ".3f", "f_hat": ".4f", "f_monograph": ".3f"},
        )
    )
    exp.data["traffic_split"] = rows

    worst_f = max(abs(r["f_hat"] - r["f_monograph"]) for r in rows)
    exp.check(
        "f-hat reproduces the §4.4 worked example",
        worst_f < 0.005,
        f"max |f_hat - monograph| = {worst_f:.4f}",
    )

    # ---- measured composite at h = f ------------------------------------
    panel = max(256 << 10, topo.per_core // 4)
    hs = sorted({0.0, 1.0} | {round(r["f_hat"], 3) for r in rows})
    recs = []
    for hv in hs:
        # The kernel sweeps a uniform grid, so request a grid that contains
        # the h values we need by running each as a 1-point sweep.
        recs += K.run("mixed-h-at", panel, "1G", f"{hv:.6f}", reps)

    med = K.median_by(recs, "h_designed", "bps")
    base = med[0.0]
    S_w = med[1.0] / base
    exp.data["S_w_measured"] = S_w
    exp.data["baseline_bps"] = base
    print(f"\n  measured S_w (h=1)  = {S_w:.3f}x")
    print(f"  baseline  (h=0)     = {base / 1e9:.2f} GB/s")

    crows = []
    for r, c in zip(rows, CASES):
        f = round(r["f_hat"], 3)
        S_meas = med[f] / base
        S_pred = amdahl_total(f, S_w, 1.0)
        err = abs(S_pred - S_meas) / S_meas
        crows.append(
            {
                "case": c["name"],
                "f": f,
                "S_measured": S_meas,
                "S_amdahl": S_pred,
                "rel_err": err,
                "S_monograph_at_ceiling": c["S_expected"],
            }
        )
    print("\nCor. 4.4 — composition at the measured S_w:")
    print(
        table(
            crows,
            [
                "case",
                "f",
                "S_measured",
                "S_amdahl",
                "rel_err",
                "S_monograph_at_ceiling",
            ],
            {
                "f": ".3f",
                "S_measured": ".3f",
                "S_amdahl": ".3f",
                "rel_err": ".1%",
                "S_monograph_at_ceiling": ".2f",
            },
        )
    )
    exp.data["composition"] = crows

    worst = max(c["rel_err"] for c in crows)
    exp.check(
        "Cor 4.4 predicts measured composite within 20%",
        worst <= TOL,
        f"worst error {worst:.1%} across {len(crows)} cases",
    )

    # ---- what the deck should say ---------------------------------------
    # With a perfectly resident weight stream the monograph's own numbers are
    # recovered by evaluating Cor 4.4 at the platform's own ceiling.
    print("\n  If S_w reached this platform's ceiling:")
    for r, c in zip(rows, CASES):
        print(
            f"    {c['name']:6s} f={r['f_hat']:.3f}  "
            f"S_total = {amdahl_total(r['f_hat'], S_w, 1.0):.2f}x measured, "
            f"{c['S_expected']:.2f}x in the monograph (S_w = 11.76)"
        )
    exp.note(
        "The monograph's 3.86x / 7.15x assume S_w = 11.76 from r = 0.085. "
        "S_total is bounded by the platform's own 1/r, so those figures are "
        "reproducible only on hardware with that bandwidth ratio. What E7 "
        "validates here is the composition law, not the magnitude."
    )

    print(exp.summary())
    print(f"\nreceipt: {exp.emit()}")
    return 0 if exp.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
