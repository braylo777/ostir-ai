"""Shared result plumbing: JSON receipts, pass/fail gates, table printing."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

RESULTS = Path(__file__).resolve().parent.parent / "results"


def _cpu_name() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def environment() -> dict[str, Any]:
    """Captured into every receipt — a result without its machine is noise."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": sys.platform,
        "machine": platform.machine(),
        "cpu": _cpu_name(),
        "python": sys.version.split()[0],
        "cpu_count": os.cpu_count(),
    }


@dataclass
class Check:
    name: str
    passed: bool
    detail: str = ""


@dataclass
class Experiment:
    """One E-numbered experiment: metadata, checks, and a JSON receipt."""

    eid: str
    title: str
    tests: str = ""
    checks: list[Check] = field(default_factory=list)
    data: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def check(self, name: str, passed: bool, detail: str = "") -> bool:
        self.checks.append(Check(name, bool(passed), detail))
        return bool(passed)

    def close(self, tol: float, actual: float, name: str, detail: str = "") -> bool:
        return self.check(name, abs(actual) <= tol, detail)

    def note(self, text: str) -> None:
        self.notes.append(text)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks) and bool(self.checks)

    def emit(self) -> Path:
        RESULTS.mkdir(exist_ok=True)
        payload = {
            "experiment": self.eid,
            "title": self.title,
            "tests": self.tests,
            "passed": self.passed,
            "environment": environment(),
            "checks": [c.__dict__ for c in self.checks],
            "notes": self.notes,
            "data": self.data,
        }
        path = RESULTS / f"{self.eid.lower()}.json"
        path.write_text(json.dumps(payload, indent=2, default=str))
        return path

    def summary(self) -> str:
        lines = [f"\n{'=' * 72}", f"{self.eid} — {self.title}"]
        if self.tests:
            lines.append(f"Tests: {self.tests}")
        lines.append("=" * 72)
        for c in self.checks:
            mark = "PASS" if c.passed else "FAIL"
            lines.append(
                f"  [{mark}] {c.name}" + (f"  — {c.detail}" if c.detail else "")
            )
        for n in self.notes:
            lines.append(f"  note: {n}")
        lines.append(f"  => {self.eid} {'PASSED' if self.passed else 'FAILED'}")
        return "\n".join(lines)


def table(rows: list[dict], cols: list[str], fmt: dict[str, str] | None = None) -> str:
    fmt = fmt or {}
    hdr = [c for c in cols]
    widths = {c: len(c) for c in cols}
    cells = []
    for r in rows:
        row = {}
        for c in cols:
            v = r.get(c, "")
            row[c] = format(v, fmt[c]) if c in fmt and v != "" else str(v)
            widths[c] = max(widths[c], len(row[c]))
        cells.append(row)
    out = [
        "  " + "  ".join(h.rjust(widths[h]) for h in hdr),
        "  " + "  ".join("-" * widths[h] for h in hdr),
    ]
    for row in cells:
        out.append("  " + "  ".join(row[c].rjust(widths[c]) for c in cols))
    return "\n".join(out)
