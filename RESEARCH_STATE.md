# Wharton research state | 2026-10-20

## Question
Does a causal episodic-memory retrieval mechanism improve out-of-sample cross-sectional equity return forecasts beyond a reservoir/latent-state/gradient-boosting baseline after costs, using identical information and comparable compute?

## Scientific disposition
INCONCLUSIVE. No verified real-market performance or Kaggle experiment artifacts have been produced in this session. Do not claim positive returns, JEPA completion, or paper readiness.

## Verified in this session
- Identified this writable GitHub repository; root README describes the architecture.
- Developed a locally executable causal reservoir/PCA/engram/XGBoost prototype and four pytest tests; 4/4 passed in the execution environment.
- PCA is NOT JEPA. A dedicated predictive-latent neural module must be separately specified and implemented before the full architectural claim is valid.
- The tests used deterministic synthetic data only. No real market performance is implied.

## Baseline specification
Information at time t: features known before execution, with asset and timestamp. Supervision: next-period excess return. Fixed echo-state reservoir; train-only PCA latent features; top-k nearest earlier episodes (strict gap) carrying historically available labels; XGBoost squared-error regressor. Main control: same configuration with memory disabled.

## Falsifier and decision
Primary null: memory does not improve preregistered net cross-sectional information coefficient or net long/short performance versus memory-off and simple XGBoost under walk-forward purging and realistic costs. If effect is negative or confidence intervals span zero, retain that result rather than tuning to win.

## Critical flaws and required evidence
| Flaw | Severity | Remediation | Evidence |
|---|---|---|---|
| No verified licensed equity data | Critical | Immutable OHLCV/adjustment snapshots, provenance, corporate-action policy | Hashes, source licenses and sample counts |
| Survivorship / point-in-time membership bias | Critical | Point-in-time universe and delisting returns | Audit of additions/removals |
| No JEPA learning in current prototype | Critical | Train predictive latent representation with stop-gradient EMA target and no future leakage | Gradient, collapse, and frozen-encoder ablations |
| In-sample training-memory reuse | High | Cross-fit historical retrieval; validate episode outcome availability as of forecast time | Unit tests of information timestamps |
| Reservoir resets at prediction boundary | High | Carry prior asset state across split while fitting scalers only on train | Streaming vs batch parity checks |
| Validation strategy absent | Critical | Purged walk-forward folds and untouched confirmatory holdout | Frozen manifest |
| Trading costs and turnover missing | Critical | Fees, spreads, market impact, execution lag, capacity | Net returns and turnover |
| No compute-matched baseline | High | Match fit/predict resource budgets and tuning | Logged CPU/GPU seconds |
| No final arXiv.sty in repository | Medium | Add user-supplied style and compile | pdflatex/latexmk logs |
| No verified real-world win | Critical | Prespecified evaluation; report all outcomes | Raw predictions, metrics, bootstrap CIs |

## Experiment plan (not yet executed)
1. Establish point-in-time licensed universe, adjusted daily data, price and volume features, sector exposures, and as-of publication timestamps.
2. Freeze train/validation/test and embargo of at least maximum label horizon, with nonoverlapping observations.
3. Compare zero forecast, ridge, XGBoost, reservoir+XGBoost, latent+reservoir+XGBoost, and memory on/off.
4. Ablate k, time exclusion gap, latent representation, reservoir, and memory weighting.
5. Report pooled and per-regime rank IC, ICIR, rank portfolio returns after costs, Sharpe with uncertainty, maximum drawdown, turnover, coverage and runtime.
6. Conduct statistical testing with temporal block bootstrap, multiple-comparison adjustment, and replication across periods.
7. Never redesign using confirmatory results. Publish mixed or negative results faithfully.

## Next verified gate
Get a real point-in-time dataset and an actual train/validation/test manifest, implement trained JEPA, and execute a reproducible walk-forward comparison. This branch is DEVELOPMENT, not a release or acceptance of scientific claims.
