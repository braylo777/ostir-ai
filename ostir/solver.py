"""Algorithm 1 — SolveResidentConfig (§5.1).

The monograph calls this "the routine that constitutes the defensible
novelty" (Part VII): it measures the machine, then minimizes *modeled DRAM
traffic* subject to an accuracy floor -- not bits, and not distortion. That
is the distinction from published quantizers, and it is the thing the patent
claim should recite.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import distortion as D
from .rate import b_eff, b_eff_hier, n_max
from .residency import GemmShape, modeled_traffic, usable_capacity


@dataclass(frozen=True)
class Candidate:
    b: int
    G: int
    hierarchical: bool
    m_c: int
    k_c: int
    rate: float
    distortion: float
    traffic: float
    p_outlier: float

    @property
    def n_panel(self) -> int:
        return self.m_c * self.k_c


def divisors(n: int, cap: int = 64) -> list[int]:
    """Divisors of n, coarsely sampled. k_c is a tiling knob, not a search
    space worth exhausting -- the traffic surface is smooth in it."""
    ds = [d for d in range(1, int(n**0.5) + 1) if n % d == 0]
    ds += [n // d for d in ds if n // d != d]
    ds = sorted(set(ds))
    if len(ds) <= cap:
        return ds
    step = len(ds) / cap
    return sorted({ds[int(i * step)] for i in range(cap)})


def solve_resident_config(
    C_bytes: int,
    eta: float,
    D_max: float,
    shape: GemmShape,
    s_act: float = 0.0,
    s_scr: float = 0.0,
    s_str: float = 0.0,
    bits=(3, 4, 5),
    groups=(16, 32, 64, 128, 256),
    n_min: int = 1 << 16,
    p_outlier: float = 0.001,
    hierarchical=(False, True),
) -> tuple[Candidate | None, list[Candidate]]:
    """Return (best, all_feasible), minimizing modeled DRAM traffic.

    C_bytes and eta are MEASURED (E2/E3), not assumed -- that is line 1 of
    Alg. 1 and the whole point of the loop.
    """
    c_prime = usable_capacity(C_bytes, eta, s_act, s_scr, s_str)
    if c_prime <= 0:
        return None, []

    feasible: list[Candidate] = []
    for b in bits:
        for G in groups:
            for hier in hierarchical:
                rate = b_eff_hier(b, G, K=8, b_s=6, b_z=6) if hier else b_eff(b, G)
                rate += 32.0 * p_outlier  # Prop 2.11
                dist = estimate_distortion(b, G, hier)
                if dist > D_max:
                    continue
                cap = n_max(rate, 1.0, int(c_prime))  # eta already applied
                for k_c in divisors(shape.K):
                    # Clamp BEFORE validating. Clamping m_c to M after the
                    # reuse-floor check lets the clamp silently violate it:
                    # at k_c = 1 the residency-limited m_c is ~2.5e6, passes
                    # the floor, then clamps to M = 4096 and yields a
                    # 4096-weight panel -- which the solver was then free to
                    # return as optimal, because a degenerate panel trivially
                    # minimizes modeled traffic.
                    m_c = min(int(cap // k_c), shape.M)
                    if m_c < 1 or m_c * k_c < n_min:
                        continue
                    traffic = modeled_traffic(shape, m_c, k_c, rate)
                    feasible.append(
                        Candidate(b, G, hier, m_c, k_c, rate, dist, traffic, p_outlier)
                    )
    if not feasible:
        return None, []
    return min(feasible, key=lambda c: c.traffic), feasible


def estimate_distortion(b: int, G: int, hierarchical: bool) -> float:
    """Alg. 1 line 6. Table 2.1 for flat; Thm 2.8's 1 dB credit for
    hierarchical, which buys the G=32 distortion at the G=64 rate."""
    d = D.distortion_grouped(b, G)
    if hierarchical:
        d *= 10 ** (-1.00 / 10.0)
    return d
