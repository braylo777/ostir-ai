"""Residency and performance model — monograph Parts III and IV.

The residency law S(h) is the single most load-bearing formula in the harness;
E4 fits measured speedup against it with r fixed from E3 and no free
parameters.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MIB = 1 << 20


# ---------------------------------------------------------------- Part IV ---


def speedup(h: float | np.ndarray, r: float) -> float | np.ndarray:
    """Thm 4.2 — S(h) = 1 / (1 - h(1-r)), r = beta_DRAM / beta_L2.

    Bandwidth composes harmonically: 1/beta_eff = h/beta_L2 + (1-h)/beta_DR.
    S(0) = 1 (all DRAM); S(1) = 1/r (the ceiling the deck quotes as 12x).
    """
    return 1.0 / (1.0 - np.asarray(h, dtype=float) * (1.0 - r))


def dS_dh(h: float | np.ndarray, r: float) -> float | np.ndarray:
    """Sensitivity: (1-r) * S(h)^2. At h=0.95, r=0.085 this is ~53.5."""
    return (1.0 - r) * speedup(h, r) ** 2


def hit_rate_for_speedup(S: float, r: float) -> float:
    """Invert Thm 4.2. S=10, r=0.085 -> h = 0.9836."""
    return float((1.0 - 1.0 / S) / (1.0 - r))


def amdahl_total(f: float, S_w: float, S_kv: float = 1.0) -> float:
    """Cor 4.4 — end-to-end speedup with weight share f of decode traffic."""
    return float(1.0 / (f / S_w + (1.0 - f) / S_kv))


def arithmetic_intensity(B_eff: float, n_batch: int = 1) -> float:
    """§4.2 — 2 ops per weight per batch element, B_eff/8 bytes per weight."""
    return float(2.0 * n_batch / (B_eff / 8.0))


def ridge_point(pi_ops_cycle: float, beta_bytes_cycle: float) -> float:
    """Roofline ridge I* = pi / beta."""
    return float(pi_ops_cycle / beta_bytes_cycle)


def batch_threshold(
    B_eff: float, pi_ops_cycle: float, beta_L2_bytes_cycle: float
) -> float:
    """Cor to Thm 4.1 — smallest batch that is compute-bound from L2.

    n_b* = I*(L2) * (B_eff/8) / 2. At B_eff=4.5, pi=2048, beta_L2=33 -> ~17.4.
    """
    I_star = ridge_point(pi_ops_cycle, beta_L2_bytes_cycle)
    return float(I_star * (B_eff / 8.0) / 2.0)


# --------------------------------------------------------------- Part III ---


def kv_bytes(L: int, n_layers: int, h_kv: int, d_h: int, B_kv: float) -> float:
    """§3.3 — KV footprint. Factor 2 is K and V."""
    return 2.0 * L * n_layers * h_kv * d_h * B_kv / 8.0


def kv_critical_context(
    beta: float, C_bytes: int, h_kv: int, d_h: int, B_kv: float
) -> float:
    """Thm 3.2 — L* at which KV alone occupies fraction beta of L2 (1 layer)."""
    return float(8.0 * beta * C_bytes / (2.0 * h_kv * d_h * B_kv))


def weight_share(
    n_params: float,
    B_eff: float,
    L: int,
    n_layers: int,
    h_kv: int,
    d_h: int,
    B_kv: float,
) -> float:
    """§4.4 — f, the weight fraction of per-token decode bytes."""
    w = n_params * B_eff / 8.0
    kv = kv_bytes(L, n_layers, h_kv, d_h, B_kv)
    return float(w / (w + kv))


@dataclass(frozen=True)
class GemmShape:
    M: int
    K: int
    N_o: int
    s_B: int = 1  # bytes per streaming-operand element
    s_C: int = 4  # bytes per output element


def modeled_traffic(shape: GemmShape, m_c: int, k_c: int, B_eff: float) -> float:
    """Thm 3.4 — DRAM bytes for a blocked GEMM.

    Three terms: A read once, B re-read once per panel row, C written+read.
    The middle term is the monograph's point — bit rate governs traffic twice,
    once directly and once through how many panel rows a fixed L2 forces.
    """
    A = shape.M * shape.K * B_eff / 8.0
    n_panel_rows = np.ceil(shape.M / m_c)
    B = shape.K * shape.N_o * shape.s_B * n_panel_rows
    C = 2.0 * shape.M * shape.N_o * shape.s_C
    return float(A + B + C)


def usable_capacity(
    C_bytes: int, eta: float, s_act: float = 0.0, s_scr: float = 0.0, s_str: float = 0.0
) -> float:
    """Alg 1 line 1 — C' = eta*C - activations - scratch - streaming lines."""
    return float(eta * C_bytes - s_act - s_scr - s_str)


# ----------------------------------------------------------------- Part V ---


def recompute_crossover_bandwidth(
    pi_eff_ops_s: float, bytes_per_token: float, flops_per_token: float
) -> float:
    """Thm 5.2 — beta* below which recompute beats fetch, in bytes/s."""
    return float(pi_eff_ops_s * bytes_per_token / flops_per_token)
