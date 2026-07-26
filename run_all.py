#!/usr/bin/env python3
"""Run the Part VI protocol in dependency order and print the gate table.

Sequencing follows §6.4:
    E1, E3  -> algebra + constants locked
    E2      -> GATE: is eta >= 0.5?
    E4      -> GATE: does R^2 > 0.90?   If not, STOP and rebuild Part IV.
    E6, E7  -> GATE: is S_total >= 3x?
    E5      -> accuracy operating point

Usage:
    python3 run_all.py                 # everything except E5 --model
    python3 run_all.py --only E1 E3
    python3 run_all.py --model Qwen/Qwen2.5-0.5B    # full E5 pipeline
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EXP = ROOT / "experiments"

ORDER = [
    ("E1", "e1_bitrate.py", []),
    ("E3", "e3_bandwidth.py", []),
    ("E2", "e2_residency_knee.py", []),
    ("E4", "e4_residency_law.py", []),
    ("E6", "e6_batch.py", []),
    ("E7", "e7_amdahl.py", []),
    ("E5", "e5_accuracy.py", []),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument(
        "--model", default=None, help="pass to E5 for the full " "accuracy pipeline"
    )
    ap.add_argument(
        "--settle",
        type=float,
        default=20.0,
        help="seconds to idle between experiments (default 20)",
    )
    args = ap.parse_args()

    rc_total = 0
    ran = []
    for eid, script, extra in ORDER:
        if args.only and eid not in args.only:
            continue
        cmd = [sys.executable, str(EXP / script)] + extra
        if eid == "E5" and args.model:
            cmd += ["--model", args.model]

        # Idle between experiments. Run back-to-back on a host without core
        # isolation, each experiment leaves the next one hotter and noisier:
        # E4's R^2 fell from 0.78 standalone to -0.41 in sequence, and E2's
        # knee spread went from 2% to 52%. Settling does not make the host
        # meet §6.1's discipline -- it only stops the harness from being the
        # largest single source of its own noise.
        if ran and args.settle > 0:
            print(f"\n  ... settling {args.settle:.0f}s before {eid}")
            time.sleep(args.settle)

        print(f"\n{'#' * 72}\n# {eid}  ({' '.join(cmd[1:])})\n{'#' * 72}")
        rc = subprocess.run(cmd).returncode
        ran.append(eid)
        rc_total |= rc

    print(f"\n{'=' * 72}\nGATE SUMMARY\n{'=' * 72}")
    for eid, _, _ in ORDER:
        if eid not in ran:
            continue
        p = ROOT / "results" / f"{eid.lower()}.json"
        if not p.exists():
            print(f"  {eid}  no receipt")
            continue
        d = json.loads(p.read_text())
        n_pass = sum(1 for c in d["checks"] if c["passed"])
        n = len(d["checks"])
        mark = "PASS" if d["passed"] else "FAIL"
        print(f"  {eid}  [{mark}]  {n_pass}/{n} checks   {d['title']}")
        for c in d["checks"]:
            if not c["passed"]:
                print(f"          - {c['name']}: {c['detail']}")

    print("\nReceipts in results/*.json")
    return rc_total


if __name__ == "__main__":
    raise SystemExit(main())
