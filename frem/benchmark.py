"""Leakage-aware expanding-origin forecasting controls for a future FREM study.

Only observed, strictly earlier values may enter each prediction. This module
contains *baselines and engineered features*, NOT a trained JEPA or evidence of
investment alpha. No protected outcomes or external market data are accessed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class Windows:
    features: np.ndarray
    target: np.ndarray
    target_indices: np.ndarray
    latest_observed_indices: np.ndarray


def make_windows(values: np.ndarray, context: int, horizon: int = 1) -> Windows:
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 1 or not np.all(np.isfinite(v)):
        raise ValueError('values must be a finite one-dimensional series')
    if type(context) is not int or context < 2 or type(horizon) is not int or horizon < 1:
        raise ValueError('context >= 2 and horizon >= 1 are required')
    if len(v) < context + horizon + 1:
        raise ValueError('not enough observations for a trainable window set')
    t = np.arange(context + horizon - 1, len(v), dtype=int)
    # Forecast at end of [end-context, end); the target is end+horizon-1.
    ends = t - horizon + 1
    X = np.stack([v[end-context:end] for end in ends])
    latest = ends - 1
    if not np.all(latest < t):
        raise AssertionError('causal window construction broken')
    return Windows(X, v[t], t, latest)


def expanding_folds(n: int, min_train: int, test_size: int, gap: int = 0):
    if min_train < 1 or test_size < 1 or gap < 0:
        raise ValueError('invalid chronological fold settings')
    end = min_train
    while end + gap + test_size <= n:
        yield (end, end + gap, end + gap + test_size)
        end += test_size


def read_csv(path: str | Path, timestamp_col: str, value_col: str) -> tuple[np.ndarray, str]:
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) > 128 * 1024 * 1024:
        raise ValueError('CSV exceeds bounded 128 MiB input')
    rows = list(csv.DictReader(raw.decode('utf-8-sig').splitlines()))
    if not rows:
        raise ValueError('empty CSV')
    values, stamps = [], []
    for row in rows:
        try:
            time = datetime.fromisoformat(row[timestamp_col].strip().replace('Z', '+00:00'))
            value = float(row[value_col])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError('missing/malformed timestamp or target') from exc
        if not np.isfinite(value):
            raise ValueError('target contains nonfinite values')
        stamps.append(time)
        values.append(value)
    try:
        ascending = all(a < b for a, b in zip(stamps, stamps[1:]))
    except TypeError as exc:
        raise ValueError('mixed timezone-aware and naive timestamps') from exc
    if not ascending:
        raise ValueError('timestamps must be strictly increasing with no duplicates')
    return np.asarray(values, dtype=np.float64), hashlib.sha256(raw).hexdigest()


class FourierReservoirFeatures:
    """Fixed random reservoir plus Fourier and sign-bigram context features.

    A *development-only control*; not a JEPA mechanism. Fixed weights never see
    target values. Normalization is supplied by each fold's train-only mean/std.
    """
    def __init__(self, state_dim: int = 12, seed: int = 7, spectral_radius: float = 0.75):
        if state_dim < 1 or not 0 < spectral_radius < 1:
            raise ValueError('invalid reservoir geometry')
        rng = np.random.default_rng(seed)
        self.input_weights = rng.normal(size=state_dim)
        w = rng.normal(size=(state_dim, state_dim))
        radius = float(np.max(np.abs(np.linalg.eigvals(w))))
        self.recurrent = w * (spectral_radius / max(radius, 1e-12))
        self.state_dim = state_dim

    def transform(self, contexts: np.ndarray, train_mean: float, train_std: float) -> np.ndarray:
        X = np.asarray(contexts, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] < 3 or not np.all(np.isfinite(X)):
            raise ValueError('contexts must be finite rows with length >=3')
        if not np.isfinite(train_mean) or not np.isfinite(train_std) or train_std <= 0:
            raise ValueError('normalization must come from finite positive-scale training values')
        rows = []
        for raw in X:
            x = (raw - train_mean) / train_std
            state = np.zeros(self.state_dim)
            for z in x:
                state = np.tanh(self.input_weights * z + self.recurrent @ state)
            fft = np.abs(np.fft.rfft(x - np.mean(x)))[1:5]
            fft = np.pad(fft, (0, max(0, 4 - len(fft))))[:4]
            moves = np.sign(np.diff(x)).astype(int) + 1
            codes = moves[:-1] * 3 + moves[1:]
            bigrams = np.bincount(codes, minlength=9).astype(float) / max(1, len(codes))
            rows.append(np.concatenate(([x[-1], np.mean(x), np.std(x)], fft, bigrams, state)))
        return np.asarray(rows, dtype=np.float64)


def evaluate(values: np.ndarray, *, context: int = 12, horizon: int = 1,
             min_train: int = 80, test_size: int = 20, gap: int = 0,
             alpha: float = 1.0, reservoir_dim: int = 12, seed: int = 7) -> dict:
    """Evaluate expanding-origin baselines, leaving all fold outcomes visible."""
    if not np.isfinite(alpha) or alpha <= 0:
        raise ValueError('Ridge alpha must be positive and finite')
    w = make_windows(values, context, horizon)
    rows = []
    for fold_index, (train_end, start, stop) in enumerate(expanding_folds(len(values), min_train, test_size, gap)):
        # A horizon-h forecast for target t is made at time t-h. Training
        # targets later than that forecast origin are not yet observable,
        # even if their calendar index precedes the test target boundary.
        # For the first target in the fold, the last available target is
        # start-horizon. This embargo avoids multi-step label lookahead.
        available_train_end = min(train_end, start - horizon + 1)
        train = w.target_indices < available_train_end
        test = (w.target_indices >= start) & (w.target_indices < stop)
        if not train.any() or not test.any():
            continue
        if not (w.target_indices[train].max() <= int(w.latest_observed_indices[test].min())
                and np.all(w.latest_observed_indices[test] < w.target_indices[test])):
            raise AssertionError('train/test timing leak')
        Xtr, ytr, Xte, yte = w.features[train], w.target[train], w.features[test], w.target[test]
        # Fit all learned preprocessing on training contexts only.
        scaler = StandardScaler().fit(Xtr)
        ridge = Ridge(alpha=alpha).fit(scaler.transform(Xtr), ytr)
        fr = FourierReservoirFeatures(state_dim=reservoir_dim, seed=seed)
        train_mu = float(np.mean(Xtr))
        train_std = max(float(np.std(Xtr)), 1e-12)
        engineered_train = fr.transform(Xtr, train_mu, train_std)
        engineered_test = fr.transform(Xte, train_mu, train_std)
        engineered_scaler = StandardScaler().fit(engineered_train)
        engineered_ridge = Ridge(alpha=alpha).fit(engineered_scaler.transform(engineered_train), ytr)
        predictions = {
            'historical_mean': np.full(len(yte), float(np.mean(ytr))),
            'persistence': Xte[:, -1],
            'ridge_context': ridge.predict(scaler.transform(Xte)),
            'fourier_reservoir_ridge': engineered_ridge.predict(engineered_scaler.transform(engineered_test)),
        }
        fold = {'fold': fold_index, 'train_target_end_exclusive': int(available_train_end),
                'test_target_start': start, 'test_target_end_exclusive': stop,
                'n_train': int(len(ytr)), 'n_test': int(len(yte)), 'models': {}}
        for name, pred in predictions.items():
            if not np.all(np.isfinite(pred)):
                raise ValueError(f'nonfinite predictions: {name}')
            delta = yte - pred
            fold['models'][name] = {'mse': float(np.mean(delta**2)),
                                    'mae': float(np.mean(np.abs(delta)))}
        rows.append(fold)
    if not rows:
        raise ValueError('no valid chronological test folds; need more data or smaller settings')
    return {'evidence_class': 'EXPLORATORY_BASELINE_ONLY_NOT_JEPA_OR_ALPHA',
            'n_observations': int(len(values)), 'context': context, 'horizon': horizon,
            'min_train': min_train, 'test_size': test_size, 'gap': gap,
            'alpha': alpha, 'reservoir_dim': reservoir_dim, 'seed': seed,
            'folds': rows,
            'limitation': 'Rolling forecasts use past actual observations, including earlier test-period observations when available. No costs, investment simulation, independent held-out lockbox, or statistical claim.'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--csv',required=True)
    p.add_argument('--timestamp-col',default='timestamp')
    p.add_argument('--target-col',required=True)
    p.add_argument('--context',type=int,default=12)
    p.add_argument('--horizon',type=int,default=1)
    p.add_argument('--min-train',type=int,default=80)
    p.add_argument('--test-size',type=int,default=20)
    p.add_argument('--gap',type=int,default=0)
    p.add_argument('--reservoir-dim',type=int,default=12)
    p.add_argument('--alpha',type=float,default=1.)
    p.add_argument('--seed',type=int,default=7)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv)
    data, sha=read_csv(a.csv,a.timestamp_col,a.target_col)
    result=evaluate(data,context=a.context,horizon=a.horizon,min_train=a.min_train,
                    test_size=a.test_size,gap=a.gap,reservoir_dim=a.reservoir_dim,alpha=a.alpha,seed=a.seed)
    result['input_sha256']=sha
    result['timestamp_column']=a.timestamp_col
    result['target_column']=a.target_col
    a.out.parent.mkdir(parents=True,exist_ok=True)
    # Never accidentally overwrite a retained run with changed settings/data.
    with a.out.open('x',encoding='utf-8') as f:
        json.dump(result,f,indent=2,allow_nan=False)
        f.write('\n')
    print(f'EXPLORATORY only: {len(result["folds"])} folds; retained {a.out}')


if __name__=='__main__': main()
