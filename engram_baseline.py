"""Causally restricted PCA-reservoir-engram XGBoost baseline. NOT a JEPA implementation."""
from dataclasses import dataclass
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBRegressor

@dataclass(frozen=True)
class Config:
    seed: int = 42
    reservoir_size: int = 64
    spectral_radius: float = .85
    leak: float = .35
    input_scale: float = .25
    latent_dim: int = 12
    memory_k: int = 12
    memory_gap: int = 5
    n_estimators: int = 150

def chronological_splits(times, train_fraction=.6, validation_fraction=.2, embargo=5):
    t = np.asarray(times)
    u = np.unique(t)
    a, b = int(len(u)*train_fraction), int(len(u)*(train_fraction+validation_fraction))
    if a <= embargo or b-a <= 2*embargo or len(u)-b <= embargo:
        raise ValueError("insufficient unique dates")
    return tuple(np.isin(t, v) for v in (u[:a-embargo], u[a+embargo:b-embargo], u[b+embargo:]))

def metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if y.shape != prediction.shape or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("invalid labels/predictions")
    return {"mse": float(np.mean((y-prediction)**2)),
            "mae": float(np.mean(abs(y-prediction))),
            "sign_accuracy": float(np.mean(np.sign(y)==np.sign(prediction)))}

class EngramBaseline:
    def __init__(self, config=Config(), use_memory=True):
        self.config, self.use_memory = config, use_memory

    def _reservoir(self, x, asset):
        out = np.empty((len(x), self.config.reservoir_size))
        state = np.zeros(self.config.reservoir_size)
        previous = None
        for i, (row, identifier) in enumerate(zip(x, asset)):
            if identifier != previous:
                state[:] = 0
            state = (1-self.config.leak)*state + self.config.leak*np.tanh(row @ self.win + state @ self.w)
            out[i] = state
            previous = identifier
        return out

    def _memory(self, representations, times):
        out = np.zeros((len(representations), 3))
        if not self.use_memory:
            return out
        for i, (z, t) in enumerate(zip(representations, times)):
            valid = np.flatnonzero(self.train_times < t-self.config.memory_gap)
            if not len(valid):
                continue
            dist = np.sum((self.train_representations[valid]-z)**2, axis=1)
            picked = np.argsort(dist, kind="stable")[:self.config.memory_k]
            index = valid[picked]
            weights = np.exp(-(dist[picked]-dist[picked].min())/(np.median(dist[picked])+1e-8))
            weights /= weights.sum()
            mean = weights @ self.train_y[index]
            out[i] = (mean, np.sqrt(weights @ (self.train_y[index]-mean)**2), len(index)/self.config.memory_k)
        return out

    def _features(self, x, assets, times):
        scaled = self.scaler.transform(x)
        latent = self.pca.transform(scaled)
        state = self._reservoir(scaled, assets)
        representations = np.c_[latent, state]
        return np.c_[scaled, representations, self._memory(representations, times)]

    def fit(self, x, y, assets, times):
        x, y, assets, times = np.asarray(x, float), np.asarray(y, float), np.asarray(assets), np.asarray(times)
        if x.ndim != 2 or not len(x) or len(y)!=len(x) or len(assets)!=len(x) or len(times)!=len(x):
            raise ValueError("shape mismatch")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError("nonfinite data")
        if np.any((assets[1:]==assets[:-1]) & (times[1:]<=times[:-1])):
            raise ValueError("timestamps must increase within each asset block")
        rng = np.random.default_rng(self.config.seed)
        self.scaler = StandardScaler().fit(x)
        scaled = self.scaler.transform(x)
        self.pca = PCA(n_components=min(self.config.latent_dim, *scaled.shape), random_state=self.config.seed).fit(scaled)
        n = self.config.reservoir_size
        self.win = rng.normal(0, self.config.input_scale/np.sqrt(x.shape[1]), (x.shape[1], n))
        w = rng.normal(0, 1/np.sqrt(n), (n, n))
        self.w = w * (self.config.spectral_radius/max(abs(np.linalg.eigvals(w))))
        self.train_representations = np.c_[self.pca.transform(scaled), self._reservoir(scaled, assets)]
        self.train_times, self.train_y = times.copy(), y.copy()
        self.model = XGBRegressor(objective="reg:squarederror", n_estimators=self.config.n_estimators,
                                  max_depth=3, learning_rate=.04, reg_lambda=5., random_state=self.config.seed,
                                  n_jobs=1)
        self.model.fit(self._features(x, assets, times), y)
        return self

    def predict(self, x, assets, times):
        x = np.asarray(x, float)
        if not np.isfinite(x).all():
            raise ValueError("nonfinite features")
        return self.model.predict(self._features(x, np.asarray(assets), np.asarray(times)))
