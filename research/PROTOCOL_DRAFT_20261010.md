# FREM prospective benchmark — pre-outcome draft

**Status:** PRE-OUTCOME / NOT FROZEN / BASELINE IMPLEMENTATION ONLY.

Hypothesis to test (not yet authorized): a versioned JEPA-plus-reservoir-plus-event-memory mechanism improves a declared primary prediction endpoint over strong chronological and matched-budget controls on a licensed point-in-time financial dataset. All enhancements must be evaluated against a baseline and a genuinely incremental ablation, not only a historical mean.

## What is implemented now

The code provides a bounded, data-agnostic, expanding-origin *exploratory* control suite: historical train mean, lag persistence, train-only standardized ridge, and a fixed random reservoir + Fourier-magnitude + sign-bigram context-feature ridge readout. It does **not** implement or train JEPA, episodic engram memory, stochastic prediction heads, XGBoost, or a tradeable backtest. Do not mislabel this as a combined FREM model or a proven result.

## Hard prerequisites before science

- [ ] Named investigator, independent reviewer and exact research question.
- [ ] Point-in-time dataset URL, license, release identity and SHA-256. No hindsight-adjusted or survivor-only universe.
- [ ] Prespecified timestamp/target/unit, context and horizon; decide return prediction vs price prediction.
- [ ] Prespecified training/validation/testing and regime-transfer cohorts; no protected-outcome-dependent redesign.
- [ ] Predeclared primary metric, uncertainty/paired-seed procedure, failure criteria and practical effect threshold.
- [ ] Match model capacity, training budget, feature availability and evaluation cadence against strong conventional/JEPA alternatives.
- [ ] Declare commissions, spreads, slippage, liquidity and financing if any investment/economic result will be claimed.
- [ ] Record complete training and inference code, environment, exact data hashes and no-leak audit.
- [ ] Require external review before outcome-bearing protected evaluation.

## Interpretation

The current runner uses online rolling-origin forecasts: at each test timestamp it may use *earlier observed* test-period values, never the target/future at that timestamp. It is **not** a closed-loop multi-step forecast. Overlapping rolling folds can share earlier test labels as training data only when those labels have become historical, exactly as declared by the pre-outcome design. Keep all reported folds; do not select a profitable market subset post hoc.

The generated JSON outputs are intentionally marked `EXPLORATORY_BASELINE_ONLY_NOT_JEPA_OR_ALPHA`. No data or outcome is bundled, and no protected analysis is authorized by this implementation.
