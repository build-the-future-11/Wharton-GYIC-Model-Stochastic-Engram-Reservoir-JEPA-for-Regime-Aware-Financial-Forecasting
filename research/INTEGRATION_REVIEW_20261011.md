# FREM baseline integration, 11 October 2026

The CSV parser repair in PR #4 and the CPU random-state repair in PR #3 were
sibling changes on PR #2. This continuation combines those already authored
changes in PR #4 so that adoption does not lose either fix or its tests.

The combined source has 21 inherited files. Before appending this note and the
new verification receipt, every file matched one of the two parent commits
exactly. `frem/benchmark.py` and the CSV tests come from PR #4.
`frem/component_audit.py`, its random-state tests and the two-job workflow come
from PR #3. No forecasting equations, fitting defaults or evaluation boundaries
were changed. All previous verification and failed CSV-admission evidence is
retained. The repository-local state records the new engineering receipt as an
additive update while preserving the previous receipt and scientific status.

The first combined run without PyTorch passed 33 tests and skipped two optional
cases plus the optional random-state module. After installing the workflow's
CPU PyTorch version in an isolated test environment, the complete configured
suite passed 40 tests with no skips. This verifies the combination on artificial
software fixtures. It does not establish predictive value or market alpha.

The workflow continues to check all configured tests and separately requires a
working CPU PyTorch import before executing the prototype-specific tests.
Hosted execution is reported against the published commit in the PR, rather
than inferred from these local results.

The larger private FREM package remains a separate artifact. Licensed
point-in-time data, the research contribution, matched controls, a declared
budget, an approved frozen protocol and protected-evaluation admission remain
open requirements. No market-data study, trade, submission, main-branch merge
or deployment was performed by this integration.
