# Reproduce the proposed economic specification's arithmetic checks

The [proposed specification](../../ECONOMIC_SPEC_PROPOSED_20261011.md), Section 6,
contains six constructed examples and 15 recorded Decimal arithmetic checks.
The original [arithmetic receipt](arithmetic.json) and
[documentation receipt](documentation_receipt.json) are retained unchanged.

From the repository root, run:

```bash
python research/verification/economic_spec_20261011/check_arithmetic.py
```

This standalone checker uses only the Python standard library. It recomputes
all 15 named values from the written examples, compares each with both the
expected and observed values in the historical receipt, prints a JSON report,
and exits with status 0 only when every check passes. Status 1 means an
arithmetic mismatch; status 2 means an invalid or unreadable reference. The
default reference is resolved beside the checker, independently of the current
working directory. Nothing is written by the checker.

The optional `--reference PATH` accepts another constructed-check receipt for
testing failure detection. It does not admit market data. The retained
[replay receipt](replay_20261011.json) records a successful default replay and
failure controls for a wrong expected value, wrong historical observed value,
missing check, and duplicate check. These controls used temporary copies;
the historical files and every inherited repository file were preserved.

The checker supplies repeatable analytic unit checks for the existing design.
It supplies no portfolio implementation, model run, backtest, calibrated risk
estimate, market-data result or empirical protocol freeze. Status remains
`PROPOSED_NOT_FROZEN / NOT_EVALUATED`; no illustrative parameter is adopted.
Scientific and investor review requirements, chronological cost-adjusted
holdout requirements and negative-result retention rules are unchanged.
