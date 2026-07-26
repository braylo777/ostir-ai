"""Bit-rate algebra — monograph Part I.

Every function here is a direct transcription of a numbered definition. Nothing
in this module measures anything; it is the closed-form layer that E1 checks a
real serializer against.
"""

from __future__ import annotations

from dataclasses import dataclass

FP16_PAIR_BITS = 32  # fp16 scale + fp16 zero, the flat-scheme metadata budget


def b_eff(b: float, G: int, m: int = FP16_PAIR_BITS) -> float:
    """Def 1.1 — effective bit rate of flat grouped affine quantization."""
    return b + m / G


def phi(b: float, G: int, m: int = FP16_PAIR_BITS) -> float:
    """Def 1.2 — metadata fraction of the bit budget, m / (bG + m)."""
    return m / (b * G + m)


def b_eff_hier(
    b: float, G: int, K: int, b_s: int, b_z: int, super_bits: int = FP16_PAIR_BITS
) -> float:
    """Def 1.3 — two-level metadata.

    `super_bits` is the per-super-block budget: fp16 super-scale + fp16
    super-min = 32 bits, matching GGUF Q4_K. See quantize.py for why the
    super-min is a *multiplier* rather than an affine offset — an affine
    offset would need a third fp16 and break Ex. 1.4.
    """
    return b + (b_s + b_z) / G + super_bits / (K * G)


def n_max(B: float, eta: float, C_bytes: int) -> float:
    """Def 1.8 — weights holdable per core at rate B given usable L2 fraction."""
    return 8.0 * eta * C_bytes / B


def elasticity(b: float, G: int, m: int = FP16_PAIR_BITS) -> float:
    """Thm 2.2 — d ln N_max / d ln G. Equals phi exactly; kept separate so E1
    can assert the identity rather than assume it."""
    return phi(b, G, m)


def doubling_gain(b: float, G: int, m: int = FP16_PAIR_BITS) -> float:
    """Prop 2.4 — capacity fraction gained by G -> 2G. Equals phi(2G)."""
    return b_eff(b, G, m) / b_eff(b, 2 * G, m) - 1.0


def exhaustion_bound(b: float, G: int, m: int = FP16_PAIR_BITS) -> float:
    """Thm 2.5 — total capacity from removing *all* metadata: phi / (1 - phi)."""
    p = phi(b, G, m)
    return p / (1.0 - p)


def panel_bytes(N: int, B: float) -> float:
    """Bytes occupied by an N-weight panel at effective rate B."""
    return N * B / 8.0


def n_slice_from_bytes(size_mib: float, B: float) -> float:
    """§1.4 inversion — recover the panel weight count from a deck datapoint."""
    return 8.0 * size_mib * (1 << 20) / B


@dataclass(frozen=True)
class RateConfig:
    """A fully specified quantization format."""

    b: int
    G: int
    hierarchical: bool = False
    K: int = 8
    b_s: int = 6
    b_z: int = 6
    m: int = FP16_PAIR_BITS

    @property
    def rate(self) -> float:
        if self.hierarchical:
            return b_eff_hier(self.b, self.G, self.K, self.b_s, self.b_z)
        return b_eff(self.b, self.G, self.m)

    @property
    def name(self) -> str:
        if self.hierarchical:
            return f"hier-b{self.b}-G{self.G}-K{self.K}-s{self.b_s}z{self.b_z}"
        return f"flat-b{self.b}-G{self.G}"
