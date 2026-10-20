# Decisions — 2026-10-20
- Scope exclusively this Wharton repository on the named development branch; do not access or mutate Worldcastr.
- Keep development and confirmatory evidence separate and retain failures.
- Uploaded local prototype tested with `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 pytest -q --disable-warnings`: **11 passed in 4.40s**. Not a clean checkout of remote branch.
- The trainable JEPA remains separate from the reservoir/engram/XGBoost baseline. Do not claim a complete integrated predictor.
- Earlier synthetic-memory negative results were user reported; raw logs are unavailable. Retain report as unverified, not as established statistics.
- Critical experimental blockers: label availability, survivorship, actual turnover and impact, data license/hash, baseline parity, GPU runs and protected split.
- No preregistered effect threshold yet. Never use protected outcomes to set it.
