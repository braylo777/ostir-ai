"""Driver for the C kernel: build it, run it, parse its JSON lines."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
KDIR = ROOT / "kernels"
BIN = KDIR / "bin" / "ostir_kernel"


class KernelError(RuntimeError):
    pass


def build(force: bool = False) -> Path:
    if force or not BIN.exists():
        r = subprocess.run(["make", "-C", str(KDIR)], capture_output=True, text=True)
        if r.returncode != 0:
            raise KernelError(f"build failed:\n{r.stdout}\n{r.stderr}")
    return BIN


def run(*args: Any, timeout: int = 3600) -> list[dict]:
    """Run a kernel subcommand, return its parsed JSON lines.

    stderr is forwarded rather than swallowed -- the counter-availability
    warnings there are load-bearing for interpreting the results.
    """
    build()
    cmd = [str(BIN)] + [str(a) for a in args]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.stderr.strip():
        print(r.stderr.rstrip(), file=sys.stderr)
    if r.returncode != 0:
        raise KernelError(f"{' '.join(cmd)} exited {r.returncode}")
    out = []
    for line in r.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            out.append(json.loads(line))
    if not out:
        raise KernelError(f"{' '.join(cmd)} produced no records")
    return out


def median_by(records: list[dict], key: str, value: str) -> dict[Any, float]:
    """Median of `value` grouped by `key`.

    §6.1 requires median and IQR, never mean -- a single descheduled run
    skews a mean badly and these distributions are one-sided.
    """
    buckets: dict[Any, list[float]] = {}
    for rec in records:
        buckets.setdefault(rec[key], []).append(float(rec[value]))
    return {k: float(np.median(v)) for k, v in buckets.items()}


def iqr_by(records: list[dict], key: str, value: str) -> dict[Any, float]:
    buckets: dict[Any, list[float]] = {}
    for rec in records:
        buckets.setdefault(rec[key], []).append(float(rec[value]))
    return {
        k: float(np.percentile(v, 75) - np.percentile(v, 25))
        for k, v in buckets.items()
    }


# ------------------------------------------------------------------ cache --


@dataclass(frozen=True)
class L2Topology:
    """L2 geometry, distinguishing private from cluster-shared.

    The distinction is not pedantry. A single-threaded residency sweep gets
    the WHOLE shared cache, so dividing its knee by a per-core share yields a
    meaningless eta > 1 (measured 9.9 on an M2 Pro before this was fixed).
    eta is only comparable to the monograph's when L2 is private per core.
    """

    per_core: int
    shared: int
    n_sharing: int
    provenance: str

    @property
    def is_private(self) -> bool:
        return self.n_sharing == 1

    @property
    def single_thread_capacity(self) -> int:
        """What one thread can actually hold resident."""
        return self.shared


def l2_topology() -> L2Topology:
    try:
        if sys.platform == "darwin":

            def s(k):
                return int(
                    subprocess.run(
                        ["sysctl", "-n", k], capture_output=True, text=True, timeout=5
                    ).stdout.strip()
                )

            shared = s("hw.perflevel0.l2cachesize")
            ncore = s("hw.perflevel0.physicalcpu")
            return L2Topology(
                shared // ncore,
                shared,
                ncore,
                f"darwin: {shared / (1 << 20):.0f} MiB L2 shared across "
                f"{ncore} P-cores = {shared // ncore / (1 << 20):.2f} MiB/core "
                f"(SHARED, not private -- see docs/PLATFORM.md)",
            )
        idx = Path("/sys/devices/system/cpu/cpu0/cache")
        for d in sorted(idx.glob("index*")):
            if (d / "level").read_text().strip() == "2":
                size = (d / "size").read_text().strip()
                mult = {"K": 1 << 10, "M": 1 << 20}.get(size[-1], 1)
                by = int(size[:-1]) * mult
                cpus = (d / "shared_cpu_list").read_text().strip()
                n = _count_cpu_list(cpus)
                return L2Topology(
                    by // n,
                    by,
                    n,
                    f"linux sysfs: L2 {size}, shared_cpu_list={cpus}"
                    + (" (private)" if n == 1 else f" (SHARED by {n})"),
                )
    except Exception as e:  # noqa: BLE001
        return L2Topology(
            2 << 20, 2 << 20, 1, f"fallback 2 MiB private (probe failed: {e})"
        )
    return L2Topology(
        2 << 20, 2 << 20, 1, "fallback 2 MiB private (no cache info found)"
    )


def _count_cpu_list(spec: str) -> int:
    """Count CPUs in a sysfs list like '0-3,8'. SMT siblings share L2, so a
    2-entry list on a hyperthreaded core still means one physical core."""
    total = 0
    for part in spec.split(","):
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-")
            total += int(b) - int(a) + 1
        else:
            total += 1
    return max(1, total)


def l2_bytes_per_core() -> tuple[int, str]:
    """Back-compat shim: per-core share plus provenance."""
    t = l2_topology()
    return t.per_core, t.provenance


def true_machine() -> str:
    """Real hardware arch, not the interpreter's.

    platform.machine() reports x86_64 for a Rosetta-emulated Python on Apple
    Silicon, which silently mislabels every receipt with the wrong ISA. sysctl
    is a native binary and reports the truth.
    """
    if sys.platform == "darwin":
        try:
            r = subprocess.run(
                ["sysctl", "-n", "hw.optional.arm64"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.stdout.strip() == "1":
                return "arm64"
        except Exception:  # noqa: BLE001
            pass
    return platform.machine()


def interpreter_emulated() -> bool:
    return true_machine() == "arm64" and platform.machine() == "x86_64"


def platform_note() -> str:
    m = true_machine()
    suffix = (
        " NOTE: this Python is x86_64 under Rosetta, so "
        "platform.machine() misreports the ISA; the C kernel is native "
        "arm64."
        if interpreter_emulated()
        else ""
    )
    if m == "x86_64":
        return "x86_64" + suffix
    return (
        f"{m}: no Intel AMX and no perf_event counters. Bandwidth ratios "
        f"and S(h) are measurable; AMX-specific predictions are not." + suffix
    )
