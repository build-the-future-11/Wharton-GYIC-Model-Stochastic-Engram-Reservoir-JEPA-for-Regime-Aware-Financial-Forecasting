# Wharton-GYIC-Model-Stochastic-Engram-Reservoir-JEPA-for-Regime-Aware-Financial-Forecasting
Stochastic Engram Reservoir JEPA learns market states with JEPA, tracks dynamics with reservoir computing, retrieves useful historical episodes through engram memory, models uncertainty stochastically, and uses XGBoost for final forecasts.


## Current engineering status: exploratory baselines, not FREM efficacy

The original JEPA + reservoir + episodic-memory + stochastic + XGBoost
architecture above is a **research proposal**. The repository now contains a
**pre-outcome control suite**, not a trained JEPA or evidence of market alpha.

The following are implemented and tested: strict chronological CSV ingestion,
a forecast-origin-aware horizon embargo, expanding train/test origins, a
historical-mean control, lag persistence, train-only scaled Ridge, and a
train-only fitted readout over fixed Fourier + reservoir + sign-bigram features.
Inputs must be a legitimately acquired point-in-time CSV with a strictly
increasing ISO timestamp and a numeric target. Neither the project nor the
example below bundles real market data.

### Install and run engineering tests

```bash
python -m pip install -e ".[dev]"
python -m pytest -q tests
```

### Run an exploratory comparison on your own licensed series

```bash
python -m frem.benchmark \
  --csv /path/to/chronological_data.csv \
  --timestamp-col timestamp \
  --target-col value \
  --context 12 --horizon 1 \
  --min-train 80 --test-size 20 \
  --out /path/to/new_exploratory_results.json
```

The tool keeps all chronological folds and writes an immutable new JSON output
(refusing accidental overwrite). The output is explicitly tagged
`EXPLORATORY_BASELINE_ONLY_NOT_JEPA_OR_ALPHA`.

CSV admission checks the complete header and every populated record, including
optional columns. Duplicate or blank headers, missing/extra cells, malformed
quoting, and nonfinite targets are rejected before any fit. UTF-8 BOMs, quoted
multiline fields and blank separator lines remain valid. Input reading is
bounded to 128 MiB plus one overflow-detection byte; the digest covers the exact
accepted raw bytes. This establishes file consistency, not data rights or
point-in-time market provenance.

`RESEARCH_STATE.json` records the current engineering scope and retained
verification. The complete test command includes the component-audit and CSV
admission suites. The two optional prototype tests skip when PyTorch is absent;
that skip does not establish prototype runtime compatibility.

**Interpretation:** models can use previously observed test-period inputs for
later rolling-origin predictions, but no future observations. For horizons
greater than one, training labels unavailable at the *first* forecast origin
are embargoed. There are no transaction costs, portfolio returns, independently
protected evaluation, market-data licenses, model-compute matching, trained
JEPA, or claims of statistical/economic superiority yet.

See [the prospectively specified research contract](research/PROTOCOL_DRAFT_20261010.md)
for the reviewer, dataset, falsification, and held-out gates that must precede
scientific result-bearing work.
