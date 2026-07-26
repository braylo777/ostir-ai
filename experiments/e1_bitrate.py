#!/usr/bin/env python3
"""E1 — Bit-rate verification (§6.3).

Tests Def. 1.1, Def. 1.3, Prop. 1.7.
Method: run the real quantizers, count the bits they actually serialize, and
compare against the Part I algebra for G in {16,32,64,128,256}, b in {3,4}.
Pass: measured == predicted within 0.1%; the Q4_K config reproduces 4.500.

This experiment is cheap and non-negotiable — every number downstream inherits
its correctness.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ostir import quantize as Q  # noqa: E402
from ostir import rate as R  # noqa: E402
from ostir.report import Experiment, table  # noqa: E402

TOL = 1e-3  # 0.1% pass threshold from §6.3
GROUPS = (16, 32, 64, 128, 256)
BITS = (3, 4)


def main() -> int:
    rng = np.random.default_rng(20260725)
    exp = Experiment("E1", "Bit-rate verification", "Def. 1.1, Def. 1.3, Prop. 1.7")

    # A panel-sized tensor so the last-group padding effect is negligible and
    # the measured rate is a fair test of the algebra.
    W = rng.standard_normal((2048, 1764)).astype(np.float32)
    exp.data["tensor_shape"] = list(W.shape)
    exp.data["n_weights"] = int(W.size)

    # ---- Def 1.1: flat scheme, measured vs predicted --------------------
    rows = []
    worst_flat = 0.0
    for b in BITS:
        for G in GROUPS:
            pred = R.b_eff(b, G)
            qr = Q.quantize_flat(W, b, G)
            err = abs(qr.rate - pred) / pred
            worst_flat = max(worst_flat, err)
            rows.append(
                {
                    "b": b,
                    "G": G,
                    "predicted": pred,
                    "measured": qr.rate,
                    "rel_err": err,
                    "phi": R.phi(b, G),
                    "sqnr_db": Q.sqnr_db(W, qr.dequant),
                }
            )
    exp.data["flat"] = rows
    print("\nDef. 1.1 — flat grouped affine, B_eff = b + m/G")
    print(
        table(
            rows,
            ["b", "G", "predicted", "measured", "rel_err", "phi", "sqnr_db"],
            {
                "predicted": ".4f",
                "measured": ".4f",
                "rel_err": ".2e",
                "phi": ".4f",
                "sqnr_db": ".2f",
            },
        )
    )
    exp.check(
        "Def 1.1 flat rate matches serialized bits",
        worst_flat <= TOL,
        f"worst rel err {worst_flat:.2e} <= {TOL:g}",
    )

    # ---- §1.2: the deck's operating-point table -------------------------
    deck = {16: 6.000, 32: 5.000, 64: 4.500, 128: 4.250, 256: 4.125}
    ok = all(abs(R.b_eff(4, G) - v) < 1e-9 for G, v in deck.items())
    exp.check(
        "§1.2 deck operating points reproduce exactly",
        ok,
        "b=4: G=32 -> 5.000 (Path A), G=64 -> 4.500 (Path B)",
    )

    # ---- Def 1.3 / Ex 1.4: Q4_K must land on 4.500 exactly --------------
    q4k_pred = R.b_eff_hier(b=4, G=32, K=8, b_s=6, b_z=6)
    q4k = Q.quantize_hierarchical(W, b=4, G=32, K=8, b_s=6, b_z=6)
    exp.data["q4k"] = {
        "predicted": q4k_pred,
        "measured": q4k.rate,
        "sqnr_db": Q.sqnr_db(W, q4k.dequant),
    }
    print(f"\nEx. 1.4 — Q4_K (b=4, b_s=b_z=6, G=32, K=8)")
    print(f"  predicted {q4k_pred:.6f} bpw   measured {q4k.rate:.6f} bpw")
    exp.check(
        "Ex 1.4 Q4_K predicted rate is exactly 4.500",
        abs(q4k_pred - 4.500) < 1e-12,
        f"{q4k_pred:.6f}",
    )
    exp.check(
        "Ex 1.4 Q4_K serializer matches 4.500",
        abs(q4k.rate - 4.500) / 4.5 <= TOL,
        f"{q4k.rate:.6f} bpw",
    )

    # ---- Ex 1.5: the aggressive hierarchical point ----------------------
    ex15_pred = R.b_eff_hier(b=4, G=32, K=16, b_s=4, b_z=4)
    ex15 = Q.quantize_hierarchical(W, b=4, G=32, K=16, b_s=4, b_z=4)
    exp.check(
        "Ex 1.5 aggressive hier point = 4.3125",
        abs(ex15_pred - 4.3125) < 1e-12 and abs(ex15.rate - 4.3125) / 4.3125 <= TOL,
        f"predicted {ex15_pred:.4f}, measured {ex15.rate:.4f}",
    )

    # ---- Prop 1.7: the deck scatter is collinear ------------------------
    N = R.n_slice_from_bytes(2.154, 5.0)
    exp.data["N_slice"] = N
    deck_mib = {32: 2.154, 64: 1.939, 128: 1.831, 256: 1.777}
    srows, worst_scatter = [], 0.0
    for G, observed in deck_mib.items():
        predicted = R.panel_bytes(N, R.b_eff(4, G)) / (1 << 20)
        err = abs(predicted - observed) / observed
        worst_scatter = max(worst_scatter, err)
        srows.append(
            {"G": G, "predicted_MiB": predicted, "deck_MiB": observed, "rel_err": err}
        )
    exp.data["prop_1_7"] = srows
    print(f"\nProp. 1.7 — deck scatter, single free parameter N = {N:,.0f}")
    print(
        table(
            srows,
            ["G", "predicted_MiB", "deck_MiB", "rel_err"],
            {"predicted_MiB": ".4f", "deck_MiB": ".4f", "rel_err": ".2e"},
        )
    )
    exp.check(
        "Prop 1.7 all four deck points collinear",
        worst_scatter <= 5e-4,
        f"N_slice = {N:,.0f} weights, worst err {worst_scatter:.2e}",
    )

    # ---- Thm 2.2 / Prop 2.4 / Thm 2.5: the identities behind "the 11%" --
    e_ok = all(abs(R.elasticity(4, G) - R.phi(4, G)) < 1e-12 for G in GROUPS)
    exp.check(
        "Thm 2.2 elasticity == metadata fraction",
        e_ok,
        f"at (b,G)=(4,64): eps = phi = {R.phi(4, 64):.6f} = 1/9",
    )
    d_ok = all(
        abs(R.doubling_gain(4, G) - R.phi(4, 2 * G)) < 1e-12 for G in (16, 32, 64, 128)
    )
    exp.check(
        "Prop 2.4 doubling G buys exactly phi(2G)",
        d_ok,
        f"32->64 gives {R.doubling_gain(4, 32):.6f} = phi(64)",
    )
    exh = R.exhaustion_bound(4, 64)
    exp.check(
        "Thm 2.5 total metadata headroom = 12.5%",
        abs(exh - 0.125) < 1e-9,
        f"{exh:.4%} of capacity",
    )

    # ---- Prop 3.1: the deck's implied eta -------------------------------
    eta_implied = R.panel_bytes(N, 4.5) / (2 * (1 << 20))
    n_real = R.n_max(4.5, 0.60, 2 << 20)
    shrink = 1.0 - n_real / N
    exp.data["prop_3_1"] = {
        "eta_implied": eta_implied,
        "N_at_eta_060": n_real,
        "shrink": shrink,
    }
    exp.check(
        "Prop 3.1 deck implies eta ~ 0.97 (infeasible)",
        abs(eta_implied - 0.970) < 0.005,
        f"eta_implied = {eta_implied:.3f}; at eta=0.60 the panel is "
        f"{n_real:,.0f} weights, {shrink:.1%} smaller",
    )

    print(exp.summary())
    print(f"\nreceipt: {exp.emit()}")
    return 0 if exp.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
