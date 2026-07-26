"""Quantizers — monograph §1.1, Alg. 2, Alg. 3.

Three schemes, all returning both the dequantized tensor and an exact bit
accounting so E1 can compare serialized bytes against the Part I algebra.

A design decision worth stating up front, because Alg. 2 as printed is
ambiguous about it: the super-block min is stored as a *multiplier*
(w = d*s_j*q - dmin*m_j), not as an affine offset plus range. An offset+range
encoding needs three fp16 per super-block (48 bits) and would make Ex. 1.4
evaluate to 4.5625, not 4.500. The multiplier form is what GGUF Q4_K does and
is the only one consistent with Def. 1.3's 32/(K*G) term.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .rate import FP16_PAIR_BITS

EPS = 1e-12


@dataclass
class QuantResult:
    dequant: np.ndarray  # reconstructed weights, float32, original shape
    total_bits: int  # exact serialized bit count
    n_weights: int
    scheme: str

    @property
    def rate(self) -> float:
        """Measured bits/weight — the quantity E1 checks against Part I."""
        return self.total_bits / self.n_weights

    @property
    def mse(self) -> float:
        raise NotImplementedError  # computed by caller against the original


def _pad_to(x: np.ndarray, multiple: int) -> tuple[np.ndarray, int]:
    """Right-pad a flat array so it divides evenly into groups.

    Padding is with the final value rather than zeros: a zero pad would drag a
    group's min/max outward and inflate distortion for the last group only,
    which shows up as noise in the E5 sweeps.
    """
    n = x.size
    rem = (-n) % multiple
    if rem == 0:
        return x, 0
    pad = np.full(rem, x[-1] if n else 0.0, dtype=x.dtype)
    return np.concatenate([x, pad]), rem


def quantize_flat(
    W: np.ndarray, b: int, G: int, m: int = FP16_PAIR_BITS
) -> QuantResult:
    """§1.1 — per-group affine min/max quantization.

    s_g = (max - min) / (2^b - 1), z_g = min, q = clamp(round((w-z)/s)).
    """
    shape, flat = W.shape, W.astype(np.float32).ravel()
    n = flat.size
    x, npad = _pad_to(flat, G)
    g = x.reshape(-1, G)

    lo = g.min(axis=1, keepdims=True)
    hi = g.max(axis=1, keepdims=True)
    levels = (1 << b) - 1
    s = (hi - lo) / levels
    # A constant group has s == 0; any q reconstructs exactly to lo.
    s_safe = np.where(s > EPS, s, 1.0)

    # fp16 round-trip on the stored metadata — this is what actually ships.
    s_st = np.float32(np.float16(s_safe))
    z_st = np.float32(np.float16(lo))
    s_st = np.where(s_st > EPS, s_st, 1.0)

    q = np.clip(np.rint((g - z_st) / s_st), 0, levels)
    deq = (q * s_st + z_st).astype(np.float32)

    n_groups = g.shape[0]
    total_bits = n_groups * (b * G + m)
    out = deq.ravel()[:n].reshape(shape)
    return QuantResult(out, total_bits, n, f"flat-b{b}-G{G}")


def quantize_hierarchical(
    W: np.ndarray, b: int, G: int, K: int, b_s: int = 6, b_z: int = 6
) -> QuantResult:
    """Alg. 2 — GGUF-K-quant-style two-level metadata.

    Per block: b_s-bit scale index, b_z-bit min index.
    Per super-block of K blocks: fp16 super-scale d, fp16 super-min dmin.
    Reconstruction: w = (d * s_j) * q - (dmin * m_j).
    """
    shape, flat = W.shape, W.astype(np.float32).ravel()
    n = flat.size
    x, _ = _pad_to(flat, G * K)
    sb = x.reshape(-1, K, G)  # (n_super, K, G)

    levels = (1 << b) - 1
    smax_q = (1 << b_s) - 1
    zmax_q = (1 << b_z) - 1

    lo = sb.min(axis=2)  # (n_super, K)
    hi = sb.max(axis=2)
    s = (hi - lo) / levels
    # Q4_K stores the min as a non-negative magnitude; a block whose min is
    # positive clamps to zero, costing a little range. Faithful to GGUF.
    mn = np.maximum(-lo, 0.0)

    d = s.max(axis=1, keepdims=True) / smax_q  # fp16 super-scale
    dmin = mn.max(axis=1, keepdims=True) / zmax_q  # fp16 super-min
    d = np.float32(np.float16(d))
    dmin = np.float32(np.float16(dmin))
    d_safe = np.where(d > EPS, d, 1.0)
    dmin_safe = np.where(dmin > EPS, dmin, 1.0)

    s_idx = np.clip(np.rint(s / d_safe), 0, smax_q)
    z_idx = np.clip(np.rint(mn / dmin_safe), 0, zmax_q)

    s_hat = (s_idx * d)[:, :, None]  # (n_super, K, 1)
    z_hat = -(z_idx * dmin)[:, :, None]
    s_den = np.where(s_hat > EPS, s_hat, 1.0)

    q = np.clip(np.rint((sb - z_hat) / s_den), 0, levels)
    deq = (q * s_hat + z_hat).astype(np.float32)

    n_super = sb.shape[0]
    total_bits = n_super * (K * (b * G + b_s + b_z) + FP16_PAIR_BITS)
    out = deq.ravel()[:n].reshape(shape)
    return QuantResult(out, total_bits, n, f"hier-b{b}-G{G}-K{K}-s{b_s}z{b_z}")


def extract_outliers(
    W: np.ndarray, p: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Alg. 3 — pull the top-p fraction by magnitude before grouping.

    Returns (W_clean, indices, values, extra_bits). Each outlier costs 32 bits
    (fp16 value + uint16 index), so the rate penalty is 32p bits/weight
    (Prop. 2.11).

    Must run BEFORE group min/max — that is the entire point of Prop. 2.10.
    Extracted positions are replaced by the *global* mean of the surviving
    weights, which keeps them inside every group's range so they never widen
    it.
    """
    flat = W.astype(np.float32).ravel()
    n = flat.size
    if p <= 0:
        return W.astype(np.float32), np.empty(0, np.int64), np.empty(0, np.float32), 0

    k = int(np.ceil(p * n))
    if k == 0:
        return W.astype(np.float32), np.empty(0, np.int64), np.empty(0, np.float32), 0

    idx = np.argpartition(np.abs(flat), n - k)[n - k :]
    vals = np.float32(np.float16(flat[idx]))  # stored as fp16

    clean = flat.copy()
    mask = np.ones(n, bool)
    mask[idx] = False
    clean[idx] = clean[mask].mean() if mask.any() else 0.0

    return clean.reshape(W.shape), idx, vals, 32 * k


def quantize_with_outliers(
    W: np.ndarray,
    b: int,
    G: int,
    p: float,
    hierarchical: bool = True,
    K: int = 8,
    b_s: int = 6,
    b_z: int = 6,
) -> QuantResult:
    """Alg. 3 end-to-end: extract, quantize the remainder, scatter back."""
    clean, idx, vals, extra = extract_outliers(W, p)
    qr = (
        quantize_hierarchical(clean, b, G, K, b_s, b_z)
        if hierarchical
        else quantize_flat(clean, b, G)
    )

    deq = qr.dequant.ravel().copy()
    deq[idx] = vals  # branch-free scatter
    tag = f"{qr.scheme}-p{p:g}"
    return QuantResult(deq.reshape(W.shape), qr.total_bits + extra, qr.n_weights, tag)


def mse(original: np.ndarray, reconstructed: np.ndarray) -> float:
    d = original.astype(np.float64).ravel() - reconstructed.astype(np.float64).ravel()
    return float(np.mean(d * d))


def sqnr_db(original: np.ndarray, reconstructed: np.ndarray) -> float:
    o = original.astype(np.float64).ravel()
    e = mse(original, reconstructed)
    return float(10.0 * np.log10(np.mean(o * o) / e)) if e > 0 else float("inf")
