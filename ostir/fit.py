"""The S(h) fit — the statistical core of E4.

The monograph is emphatic that this fit has NO free parameters: r comes from
E3 and is held fixed. A fit with r free will almost always look good and
proves nothing, so it is computed here only as a diagnostic and is clearly
labelled as such.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats

from .residency import speedup


@dataclass
class FitResult:
    r_used: float
    r2: float
    rmse: float
    max_abs_resid: float
    residual_trend_p: float  # Spearman p of residuals vs h
    residual_runs_p: float  # Wald-Wolfowitz p on residual signs
    residual_structured: bool
    n: int
    r_free: float = float("nan")  # diagnostic only
    r2_free: float = float("nan")
    diagnosis: str = ""
    residuals: list[float] = field(default_factory=list)

    @property
    def passes(self) -> bool:
        """E4 gate: R^2 > 0.90 AND residuals unstructured in h."""
        return self.r2 > 0.90 and not self.residual_structured


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def fit_residency_law(
    h: np.ndarray, S: np.ndarray, r_fixed: float, alpha: float = 0.05
) -> FitResult:
    """Fit S(h) = 1/(1 - h(1-r)) with r FIXED. Zero free parameters.

    Residual structure is tested two ways, because R^2 alone will happily
    accept a systematically curved fit:
      - Spearman rank correlation of residuals against h (monotone drift)
      - a Wald-Wolfowitz runs test on residual signs (any sign clustering)
    """
    h = np.asarray(h, float)
    S = np.asarray(S, float)
    pred = speedup(h, r_fixed)
    resid = S - pred

    r2 = _r2(S, pred)
    rmse = float(np.sqrt(np.mean(resid**2)))

    trend_p = 1.0
    if len(h) >= 4 and np.ptp(resid) > 0:
        trend_p = float(stats.spearmanr(h, resid).pvalue)
    runs_p = _runs_test(resid)
    structured = bool(min(trend_p, runs_p) < alpha)

    # Diagnostic free-r fit. If this differs sharply from r_fixed, the
    # bandwidth constants and the kernel's real access pattern disagree.
    r_free, r2_free = float("nan"), float("nan")
    try:
        popt, _ = optimize.curve_fit(
            lambda x, rr: speedup(x, rr), h, S, p0=[r_fixed], bounds=(1e-4, 0.999)
        )
        r_free = float(popt[0])
        r2_free = _r2(S, speedup(h, r_free))
    except Exception:  # noqa: BLE001
        pass

    return FitResult(
        r_used=r_fixed,
        r2=r2,
        rmse=rmse,
        max_abs_resid=float(np.max(np.abs(resid))),
        residual_trend_p=trend_p,
        residual_runs_p=runs_p,
        residual_structured=structured,
        n=len(h),
        r_free=r_free,
        r2_free=r2_free,
        residuals=[float(x) for x in resid],
        diagnosis=diagnose(h, S, resid, r_fixed, r2, r_free, r2_free),
    )


def _runs_test(resid: np.ndarray) -> float:
    """Two-sided p for the number of sign runs in the residual sequence."""
    signs = np.sign(resid)
    signs = signs[signs != 0]
    n = len(signs)
    if n < 8:
        return 1.0
    n_pos = int(np.sum(signs > 0))
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    runs = 1 + int(np.sum(signs[1:] != signs[:-1]))
    mu = 2.0 * n_pos * n_neg / n + 1.0
    var = (2.0 * n_pos * n_neg * (2.0 * n_pos * n_neg - n)) / (n * n * (n - 1))
    if var <= 0:
        return 1.0
    z = (runs - mu) / np.sqrt(var)
    return float(2.0 * stats.norm.sf(abs(z)))


def diagnose(h, S, resid, r_fixed, r2, r_free, r2_free=float("nan")) -> str:
    """Map a failed fit onto the monograph's named E4 fail modes (§6.3)."""
    if r2 > 0.90:
        return "fit consistent with Thm 4.2"

    hi = h >= 0.8
    ceiling = 1.0 / r_fixed
    msgs = []

    if hi.any() and np.mean(resid[hi]) < -0.05 * ceiling:
        msgs.append(
            "systematic OVERPREDICTION at high h: a serialization bottleneck "
            "not in the model (dequant, packing, or the reduction itself). "
            "Add a compute term before trusting Part IV."
        )
    if np.max(S) < 0.8 * ceiling:
        msgs.append(
            f"S saturates at {np.max(S):.2f}, well below the 1/r ceiling of "
            f"{ceiling:.2f}: beta_L2 is not achievable with this access "
            f"pattern. Re-measure E3 using the real kernel's pattern."
        )
    if np.isfinite(r_free) and abs(r_free - r_fixed) > 0.3 * r_fixed:
        msgs.append(
            f"free-r fit prefers r={r_free:.3f} vs measured r={r_fixed:.3f}: "
            f"E3 and E4 disagree about the bandwidth ratio."
        )
    # Curvature: residuals arch one way through the middle and the other way
    # at both ends. The scale is right and the shape is wrong, so refitting r
    # cannot help -- and the free-r fit confirms it by landing on the same r
    # with the same R^2. This is a distinct failure from the three the
    # monograph names, and it is the one a three-tier hierarchy produces.
    mid = (h > 0.25) & (h < 0.8)
    ends = ~mid
    if mid.any() and ends.any():
        gap = float(np.mean(resid[mid]) - np.mean(resid[ends]))
        span = float(np.ptp(S)) or 1.0
        # 15%, not 5%: on a noisy host the free-r fit wanders while the
        # curvature signature stays put, and too tight a threshold
        # silently downgrades a diagnosed curvature to "no recognized
        # fail mode" -- the least useful message the tool can emit.
        r_agrees = np.isfinite(r_free) and abs(r_free - r_fixed) <= 0.15 * r_fixed
        if abs(gap) > 0.05 * span and r_agrees:
            direction = "above" if gap > 0 else "below"
            msgs.append(
                f"systematic CURVATURE: measured S sits {direction} the model "
                f"through mid-h and the other way at both endpoints, while a "
                f"free-r fit recovers r={r_free:.4f} against the measured "
                f"{r_fixed:.4f} at R^2={r2_free:.4f}. Scale right, shape "
                f"wrong -- no choice of r fixes this. Thm 4.2 assumes exactly "
                f"two tiers at fixed per-tier bandwidth. Both an intermediate "
                f"tier (a system-level cache or large shared LLC sitting "
                f"between private L2 and DRAM) and memory-level parallelism "
                f"(effective DRAM bandwidth rising as DRAM references thin "
                f"out) produce this signature. Extend the harmonic model to "
                f"three tiers before reading this as a refutation of Part IV."
            )

    if not msgs:
        msgs.append(
            "no fit and no recognized fail mode -- the bandwidth "
            "model is wrong and Part IV must be rebuilt from "
            "measurement."
        )
    return " | ".join(msgs)
