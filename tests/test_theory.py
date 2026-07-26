"""Unit tests for the closed-form layer.

These are the invariants the monograph states as exact. They must hold to the
last digit, independently of any machine.

Run: python3 -m pytest tests/ -q     (or: python3 tests/test_theory.py)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ostir import distortion as D  # noqa: E402
from ostir import quantize as Q  # noqa: E402
from ostir import rate as R  # noqa: E402
from ostir import residency as RES  # noqa: E402
from ostir.residency import GemmShape  # noqa: E402
from ostir.solver import solve_resident_config  # noqa: E402


def test_def_1_1_operating_points():
    assert R.b_eff(4, 16) == 6.0
    assert R.b_eff(4, 32) == 5.0
    assert R.b_eff(4, 64) == 4.5
    assert R.b_eff(4, 128) == 4.25
    assert R.b_eff(4, 256) == 4.125


def test_ex_1_4_q4k_is_exactly_4_5():
    assert R.b_eff_hier(4, 32, K=8, b_s=6, b_z=6) == 4.5


def test_ex_1_5_aggressive_point():
    assert R.b_eff_hier(4, 32, K=16, b_s=4, b_z=4) == 4.3125


def test_thm_2_2_elasticity_equals_phi():
    for G in (16, 32, 64, 128, 256):
        assert abs(R.elasticity(4, G) - R.phi(4, G)) < 1e-15
    assert abs(R.phi(4, 64) - 1 / 9) < 1e-15


def test_prop_2_4_doubling_lemma():
    for G in (16, 32, 64, 128):
        assert abs(R.doubling_gain(4, G) - R.phi(4, 2 * G)) < 1e-15


def test_thm_2_5_exhaustion_is_12_5_percent():
    assert abs(R.exhaustion_bound(4, 64) - 0.125) < 1e-12


def test_prop_1_7_deck_scatter_collinear():
    N = R.n_slice_from_bytes(2.154, 5.0)
    for G, observed in {32: 2.154, 64: 1.939, 128: 1.831, 256: 1.777}.items():
        pred = R.panel_bytes(N, R.b_eff(4, G)) / (1 << 20)
        assert abs(pred - observed) / observed < 5e-4


def test_thm_4_2_residency_law_endpoints():
    r = 0.085
    assert abs(RES.speedup(0.0, r) - 1.0) < 1e-12
    assert abs(RES.speedup(1.0, r) - 1 / r) < 1e-12
    assert abs(RES.speedup(1.0, r) - 11.7647) < 1e-3


def test_thm_4_2_h_for_10x():
    # "Achieving 10x requires a hit rate h >= 0.984"
    assert abs(RES.hit_rate_for_speedup(10.0, 0.085) - 0.9836) < 1e-3


def test_thm_4_2_sensitivity_at_095():
    # "At h = 0.95: dS/dh = 53.5"
    assert abs(RES.dS_dh(0.95, 0.085) - 53.5) < 1.0


def test_cor_4_4_amdahl_worked_example():
    """Cor. 4.4 does not reproduce from its own inputs.

    With S_w = 1/0.085 = 11.7647 and the f values the monograph itself
    tabulates in §4.4, the composition gives 3.83x (MHA) and 7.33x (GQA-8),
    not the 3.86x and 7.15x printed there. Inverting Cor. 4.4 on the printed
    speedups implies f = 0.8098 and f = 0.9400, neither of which is the
    tabulated f. The formula and the traffic model are each self-consistent
    (E7 reproduces f to 4 decimal places), so the error is in the final
    arithmetic of the table.

    This matters because "4-7x end-to-end" is the headline the monograph
    tells the deck to adopt. Recomputed it is 3.8-7.3x: slightly worse for
    MHA and slightly BETTER for GQA-8.
    """
    assert abs(RES.amdahl_total(0.807, 11.7647, 1.0) - 3.8227) < 0.001
    assert abs(RES.amdahl_total(0.944, 11.7647, 1.0) - 7.3400) < 0.001


def test_thm_3_2_kv_critical_context():
    L = RES.kv_critical_context(0.25, 1 << 21, 1, 128, 3.5)
    assert abs(L - 4681) < 2


def test_prop_3_1_deck_eta_is_infeasible():
    N = R.n_slice_from_bytes(2.154, 5.0)
    eta = R.panel_bytes(N, 4.5) / (2 << 20)
    assert 0.96 < eta < 0.98
    assert abs(R.n_max(4.5, 0.60, 2 << 20) - 2_236_962) < 2


def test_thm_5_3_never_recompute_on_cpu():
    b = RES.recompute_crossover_bandwidth(4.1e12, 28672, 1.4e10)
    assert 8e6 < b < 9e6  # ~8.4 MB/s


def test_sec_4_2_arithmetic_intensity_and_batch():
    assert abs(RES.arithmetic_intensity(4.5) - 3.5556) < 1e-3
    assert abs(RES.batch_threshold(4.5, 2048.0, 33.0) - 17.4) < 0.5


def test_panter_dite_gaussian_constant():
    assert abs(D.distortion_panter_dite(4) - 0.010628) < 1e-6


def test_optimal_clipping_matches_reference():
    for b, ref in {2: 1.71, 3: 2.15, 4: 2.55, 8: 3.92}.items():
        assert abs(D.optimal_clip(b) - ref) < 0.06


def test_expected_max_matches_appendix_b():
    """Appendix B is right at 9 of 11 entries and wrong at two.

    Exact numerical integration of E[max] of n standard normals agrees with
    Appendix B to better than 0.001 at n = 10, 20, 30, 32, 50, 100, 200, 256
    and 500. It disagrees at:
        n = 64  : table 2.320, exact 2.3437  (+0.024)
        n = 128 : table 2.580, exact 2.5946  (+0.015)
    Those two are not incidental -- they are the values Table 2.1 uses for
    G = 64 and G = 128, and G = 64 is the deck's Path B operating point. The
    understated range propagates to an understated distortion: Table 2.1
    gives D(4,64) = 0.007982 where the correct value is 0.008138, so the
    monograph's Path B is ~2% more lossy than stated (-0.08 dB SQNR).
    """
    for n, ref in {10: 1.5388, 20: 1.8675, 30: 2.0428, 32: 2.0697,
                   50: 2.2491, 64: 2.3437, 100: 2.5076, 128: 2.5946,
                   200: 2.7460, 256: 2.8269, 500: 3.0367}.items():
        assert abs(D.expected_max_normal(n) - ref) < 0.002, f"n={n}"

    # The two Appendix B entries that are wrong, pinned so a future edit
    # cannot quietly "fix" the code to match the erroneous table.
    assert abs(D.expected_max_normal(64) - 2.320) > 0.02
    assert abs(D.expected_max_normal(128) - 2.580) > 0.01


def test_table_2_1_g64_distortion_is_understated():
    """Direct consequence of the Appendix B error at n = 64."""
    assert abs(D.distortion_grouped(4, 64) - 0.008138) < 1e-5
    assert D.distortion_grouped(4, 64) > 0.007982  # the monograph's value


def test_quantizer_rate_matches_algebra():
    rng = np.random.default_rng(0)
    W = rng.standard_normal((512, 512)).astype(np.float32)
    for b in (3, 4):
        for G in (32, 64, 128):
            qr = Q.quantize_flat(W, b, G)
            assert abs(qr.rate - R.b_eff(b, G)) < 1e-9
    qr = Q.quantize_hierarchical(W, 4, 32, K=8, b_s=6, b_z=6)
    assert abs(qr.rate - 4.5) < 1e-9


def test_outlier_rate_penalty_is_32p():
    rng = np.random.default_rng(1)
    W = rng.standard_normal(1 << 16).astype(np.float32)
    base = Q.quantize_flat(W, 4, 32).rate
    for p in (0.001, 0.005):
        qr = Q.quantize_with_outliers(W, 4, 32, p, hierarchical=False)
        assert abs((qr.rate - base) - 32 * p) < 1e-3


def test_thm_3_4_traffic_increases_with_rate():
    shape = GemmShape(4096, 4096, 4096)
    t1 = RES.modeled_traffic(shape, 1024, 1024, 4.0)
    t2 = RES.modeled_traffic(shape, 1024, 1024, 4.5)
    assert t2 > t1


def test_solver_prefers_lower_traffic():
    shape = GemmShape(4096, 4096, 4096)
    best, all_c = solve_resident_config(
        C_bytes=2 << 20, eta=0.60, D_max=0.02, shape=shape
    )
    assert best is not None and all_c
    assert all(best.traffic <= c.traffic for c in all_c)
    assert best.distortion <= 0.02


def _run_all() -> int:
    fns = [
        (n, f)
        for n, f in sorted(globals().items())
        if n.startswith("test_") and callable(f)
    ]
    failed = 0
    for name, fn in fns:
        try:
            fn()
            print(f"  PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {name}  {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {name}  {type(e).__name__}: {e}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
