"""Replay the 15 constructed checks in the proposed economic specification.

This is a stdlib-only arithmetic checker, not a portfolio or model implementation.
The historical receipt is an oracle for the hand-worked examples in Section 6.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from decimal import Decimal, DecimalException, localcontext
import hashlib
import json
from pathlib import Path
import platform
import sys


def constructed_values() -> dict[str, Decimal]:
    """Evaluate only the six synthetic examples already specified in Section 6."""
    with localcontext() as context:
        context.prec = 50
        d = Decimal
        entry_turnover = abs(d("0.5") - d("0.4"))
        entry_cost = d("0.001") * entry_turnover
        entry_cash = d("1") - d("0.5") - entry_cost
        entry_net = d("0.5") * d("0.02") - entry_cost
        two_leg_turnover = sum(
            abs(after - before)
            for before, after in zip(
                (d("0.4"), d("0.6")), (d("0.5"), d("0.5"))
            )
        )
        losses = (d("0"), d("0"), d("0"), d("0.04"))
        alpha = d("0.75")
        # For this finite distribution, the convex piecewise-linear objective
        # attains its minimum at a distinct loss breakpoint (possibly an interval).
        tail_loss = min(
            z
            + sum(max(loss - z, d("0")) for loss in losses)
            / d(len(losses))
            / (d("1") - alpha)
            for z in sorted(set(losses))
        )
        return {
            "entry_turnover": entry_turnover,
            "entry_cost_fraction": entry_cost,
            "entry_cost_currency": d("1000") * entry_cost,
            "entry_cash_fraction": entry_cash,
            "entry_net_return": entry_net,
            "entry_ending_wealth": d("1000") * (d("1") + entry_net),
            "two_leg_gross_turnover": two_leg_turnover,
            "two_leg_cost_fraction": d("0.001") * two_leg_turnover,
            "unfunded_fully_invested_cash": (
                d("1") - d("0.5") - d("0.5") - d("0.001") * two_leg_turnover
            ),
            "half_full_spread_cost": d("0.002") / d("2") * d("0.1"),
            "base_currency_return": d("1.02") * d("1.01") - d("1"),
            "illustrative_mean_loss": sum(losses) / d(len(losses)),
            "illustrative_variational_tail_loss": tail_loss,
            "risk_penalty_sign_initial": d("0.01") - d("0.5") * d("0.04"),
            "risk_penalty_sign_lower_tail": d("0.01") - d("0.5") * d("0.02"),
        }


def replay(reference_path: Path) -> dict:
    raw = reference_path.read_bytes()
    reference = json.loads(raw)
    observed = constructed_values()
    rows = reference["checks"]
    names = [row["check"] for row in rows]
    if len(names) != len(set(names)) or set(names) != set(observed):
        raise ValueError("Reference must contain each of the 15 named checks exactly once")
    if reference["checks_passed"] != len(observed):
        raise ValueError("Reference check count does not match the 15 constructed checks")

    checks = []
    for row in rows:
        if not all(isinstance(row[key], str) for key in ("expected", "observed")):
            raise ValueError("Reference decimal values must be strings")
        expected = Decimal(row["expected"])
        recorded = Decimal(row["observed"])
        if not expected.is_finite() or not recorded.is_finite():
            raise ValueError("Reference decimal values must be finite")
        value = observed[row["check"]]
        checks.append({
            "check": row["check"],
            "observed": str(value),
            "expected": row["expected"],
            "recorded_observed": row["observed"],
            "passed": value == expected == recorded and row["passed"] is True,
        })

    passed = sum(check["passed"] for check in checks)
    return {
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Constructed Section 6 arithmetic only; no market data, portfolio, models or outcomes",
        "python": platform.python_version(),
        "checker_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(raw).hexdigest(),
        "checks_passed": passed,
        "checks_failed": len(checks) - passed,
        "checks": checks,
        "scientific_status": "NOT_EVALUATED",
        "protocol_status": "PROPOSED_NOT_FROZEN",
        "parameters_adopted": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", type=Path,
        default=Path(__file__).with_name("arithmetic.json"),
        help="Constructed-check receipt to verify; defaults to the adjacent historical receipt",
    )
    args = parser.parse_args()
    try:
        result = replay(args.reference)
    except (OSError, ValueError, KeyError, TypeError, DecimalException) as error:
        print(json.dumps({"error": str(error), "checks_completed": False}), file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result["checks_failed"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
