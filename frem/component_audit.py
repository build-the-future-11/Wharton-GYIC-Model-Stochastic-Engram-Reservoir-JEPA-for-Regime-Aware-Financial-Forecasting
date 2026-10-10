"""Causal FREM component audit. Synthetic engineering results are NOT market evidence.

All learned preprocessing and latent training are refit inside each chronological
fold. The optional JEPA-style encoder is an unvalidated development prototype,
not the canonical FREM model. No dataset is fetched and no protected run is opened.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

EVIDENCE = 'SYNTHETIC_ENGINEERING_ONLY_NOT_MARKET_ALPHA_OR_CANONICAL_FREM'

def windows(values, context=12, horizon=1):
    v = np.asarray(values, dtype=float)
    if v.ndim != 1 or not np.isfinite(v).all():
        raise ValueError('Expected a finite one-dimensional series')
    if type(context) is not int or context < 3 or type(horizon) is not int or horizon < 1:
        raise ValueError('context must be integer >=3; horizon integer >=1')
    targets = np.arange(context + horizon - 1, len(v))
    if len(targets) < 2:
        raise ValueError('Insufficient observations')
    origins = targets - horizon
    X = np.stack([v[t-context+1:t+1] for t in origins])
    Ycontext = np.stack([v[t-context+1:t+1] for t in targets])
    return X, v[targets], targets, origins, Ycontext


def components(X, mean, std, seed=7, state_dim=12):
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] < 3 or not np.isfinite(X).all():
        raise ValueError('Invalid context matrix')
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError('Invalid TRAIN-ONLY normalization')
    if type(state_dim) is not int or state_dim < 1:
        raise ValueError('state_dim must be a positive integer')
    z = (X - mean) / std
    # Fourier coefficients retain phase; no FFT is taken across future rows.
    f = np.fft.rfft(z - z.mean(axis=1, keepdims=True), axis=1)[:, 1:5]
    fourier = np.concatenate([f.real, f.imag], axis=1)
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(state_dim, state_dim))
    radius = np.max(np.abs(np.linalg.eigvals(W)))
    W *= 0.75 / max(float(radius), 1e-12)
    U = rng.normal(size=state_dim)
    state = np.zeros((len(z), state_dim))
    for k in range(z.shape[1]):
        state = np.tanh(state @ W.T + z[:, k:k+1] * U)
    moves = np.sign(np.diff(z, axis=1)).astype(int) + 1
    codes = moves[:, :-1] * 3 + moves[:, 1:]
    bigram = np.stack([np.bincount(row, minlength=9) for row in codes]).astype(float)
    bigram /= codes.shape[1]
    return {'raw': z, 'fourier': fourier, 'reservoir': state, 'ngram': bigram}


def latent_features(Xtr, Ytr_context, Xte, seed=7, epochs=12, latent_dim=8):
    """Small EMA target encoder + latent predictor; fits TRAIN windows only.

    A variance penalty is an explicit design choice, not a claim that collapse
    is impossible. Diagnostics are always retained. Targets are fully observed
    at the fold's first forecast origin; there is no test-time adaptation.
    """
    import torch
    from torch import nn
    if epochs < 1 or latent_dim < 2:
        raise ValueError('Invalid latent training settings')
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    encoder = nn.Sequential(nn.Linear(Xtr.shape[1], 24), nn.GELU(), nn.Linear(24, latent_dim))
    target = copy.deepcopy(encoder).requires_grad_(False)
    predictor = nn.Sequential(nn.Linear(latent_dim, 24), nn.GELU(), nn.Linear(24, latent_dim))
    optim = torch.optim.AdamW(list(encoder.parameters()) + list(predictor.parameters()), lr=1e-3)
    x = torch.tensor(Xtr, dtype=torch.float32)
    y = torch.tensor(Ytr_context, dtype=torch.float32)
    losses = []
    for _ in range(epochs):
        z = encoder(x)
        with torch.no_grad():
            zy = target(y)
        prediction_loss = (predictor(z) - zy).square().mean()
        variance_penalty = torch.relu(1.0 - torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)).mean()
        loss = prediction_loss + 0.1 * variance_penalty
        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(list(encoder.parameters()) + list(predictor.parameters()), 1.0)
        optim.step()
        with torch.no_grad():
            for p, q in zip(encoder.parameters(), target.parameters()):
                q.mul_(0.99).add_(p, alpha=0.01)
        losses.append(float(loss.detach()))
    encoder.eval()
    with torch.no_grad():
        tr = encoder(x).numpy()
        te = encoder(torch.tensor(Xte, dtype=torch.float32)).numpy()
    diag = {'prototype': True, 'epochs': epochs, 'latent_dim': latent_dim,
            'objective': 'EMA latent prediction + 0.1 online variance floor',
            'trainable_parameters': sum(p.numel() for p in list(encoder.parameters())+list(predictor.parameters())),
            'train_latent_std_mean': float(tr.std(axis=0).mean()),
            'near_constant_dimensions_fraction': float((tr.std(axis=0)<1e-3).mean()),
            'initial_loss': losses[0], 'final_loss': losses[-1]}
    return tr, te, diag


def evaluate(values, *, context=12, horizon=1, min_train=100, test_size=40,
             seed=7, include_jepa=False, epochs=12):
    for name, value in [('min_train', min_train), ('test_size', test_size)]:
        if type(value) is not int or value < 1:
            raise ValueError(name + ' must be positive integer')
    X, y, targets, origins, Ycontext = windows(values, context, horizon)
    folds = []
    for start in range(min_train, len(values)-test_size+1, test_size):
        stop = start + test_size
        # Target start is predicted at start-horizon. No later training label exists yet.
        available = start - horizon
        tr = targets <= available
        te = (targets >= start) & (targets < stop)
        if tr.sum() < 4 or not te.any():
            continue
        if targets[tr].max() > origins[te].min():
            raise AssertionError('Training label is not observable at forecast origin')
        mu, sd = float(X[tr].mean()), max(float(X[tr].std()), 1e-12)
        started = time.perf_counter()
        ctr = components(X[tr], mu, sd, seed)
        cte = components(X[te], mu, sd, seed)
        diag = None
        if include_jepa:
            ctr['jepa'], cte['jepa'], diag = latent_features(
                ctr['raw'], (Ycontext[tr]-mu)/sd, cte['raw'], seed, epochs)
        arms = {'ridge_context': ['raw'], 'fourier_only': ['fourier'],
                'reservoir_only': ['reservoir'], 'ngram_only': ['ngram'],
                'full_fixed_components': ['raw', 'fourier', 'reservoir', 'ngram'],
                'no_fourier': ['raw', 'reservoir', 'ngram'],
                'no_reservoir': ['raw', 'fourier', 'ngram'],
                'no_ngram': ['raw', 'fourier', 'reservoir']}
        if include_jepa:
            arms.update({'jepa_only_prototype': ['jepa'],
                         'full_with_jepa_prototype': ['raw','fourier','reservoir','ngram','jepa']})
        preds = {'zero_return': np.zeros(te.sum()),
                 'historical_mean': np.full(te.sum(), y[tr].mean()),
                 'persistence': X[te, -1]}
        dimensions = {k: 0 for k in preds}
        for name, parts in arms.items():
            a = np.concatenate([ctr[k] for k in parts], axis=1)
            b = np.concatenate([cte[k] for k in parts], axis=1)
            scaler = StandardScaler().fit(a)
            model = Ridge(alpha=1.0).fit(scaler.transform(a), y[tr])
            preds[name] = model.predict(scaler.transform(b))
            dimensions[name] = a.shape[1]
        metrics = {}
        for name, pred in preds.items():
            if not np.isfinite(pred).all():
                raise ValueError('Nonfinite predictions in '+name)
            metrics[name] = {'mse': float(np.mean((y[te]-pred)**2)),
                             'mae': float(np.mean(np.abs(y[te]-pred))),
                             'sign_accuracy': float(np.mean(np.sign(y[te])==np.sign(pred))),
                             'readout_features': dimensions[name], 'predictions': pred.tolist()}
        folds.append({'first_test_target': start, 'last_test_target': stop-1,
                      'first_forecast_origin': int(origins[te].min()),
                      'last_train_target': int(targets[tr].max()),
                      'n_train': int(tr.sum()), 'n_test': int(te.sum()),
                      'train_normalization_mean': mu, 'train_normalization_std': sd,
                      'target_indices': targets[te].tolist(), 'actuals': y[te].tolist(),
                      'cpu_fold_seconds': time.perf_counter()-started,
                      'latent_diagnostics': diag, 'models': metrics})
    if not folds:
        raise ValueError('No valid evaluation folds')
    return {'evidence_class': 'EXPLORATORY_ENGINEERING_NOT_CONFIRMATORY',
            'seed': seed, 'context':context, 'horizon':horizon, 'folds':folds,
            'limitations': ['No licensed point-in-time market dataset', 'No portfolio or alpha claim',
             'Architectures are NOT parameter/compute matched', 'No independent lockbox opened',
             'JEPA arm is a development prototype, not a reproduced canonical FREM',
             'Rolling forecasts may use earlier observations within the test period once available']}


def synthetic_series(n=340, seed=107):
    """Known stochastic signal for plumbing tests; NOT financial observations."""
    if n < 50:
        raise ValueError('Synthetic development fixture too short')
    rng = np.random.default_rng(seed)
    v = np.zeros(n)
    for t in range(1, n):
        a = 0.35 if t < n//2 else -0.2
        v[t] = a*v[t-1] + 0.1*np.sin(2*np.pi*t/17) + rng.normal(0, 0.8)
    return v


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--include-jepa', action='store_true')
    p.add_argument('--seeds', nargs='+', type=int, default=[7,17,29])
    args = p.parse_args()
    data = synthetic_series()
    runs = [evaluate(data, seed=s, include_jepa=args.include_jepa) for s in args.seeds]
    obj = {'evidence_class': EVIDENCE, 'data_generator_seed':107,
           'data_sha256':hashlib.sha256(data.astype('<f8').tobytes()).hexdigest(),
           'runs':runs}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as f:
        json.dump(obj,f,indent=2,allow_nan=False)
    print(json.dumps({'evidence_class':EVIDENCE, 'seeds':len(runs),
                      'folds_per_seed':len(runs[0]['folds']),
                      'models_per_fold':len(runs[0]['folds'][0]['models'])}))

if __name__ == '__main__':
    main()
