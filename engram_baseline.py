"""Point-in-time PCA/echo-state/engram/XGBoost DEVELOPMENT baseline; not JEPA."""
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
    t=np.asarray(times); u=np.unique(t)
    a,b=int(len(u)*train_fraction),int(len(u)*(train_fraction+validation_fraction))
    if embargo<0 or a<=embargo or b-a<=2*embargo or len(u)-b<=embargo:
        raise ValueError("insufficient dates")
    return tuple(np.isin(t,v) for v in (u[:a-embargo],u[a+embargo:b-embargo],u[b+embargo:]))

def metrics(y,prediction):
    y,prediction=np.asarray(y),np.asarray(prediction)
    if y.shape!=prediction.shape or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("invalid labels/predictions")
    return {"mse":float(np.mean((y-prediction)**2)),
            "mae":float(np.mean(abs(y-prediction))),
            "sign_accuracy":float(np.mean(np.sign(y)==np.sign(prediction)))}

class EngramBaseline:
    def __init__(self, config=Config(), use_memory=True):
        self.config,self.use_memory=config,use_memory

    @staticmethod
    def _check_order(assets,times):
        seen={}
        for a,t in zip(assets,times):
            if a in seen and t<=seen[a]:
                raise ValueError("each asset must arrive strictly later than its preceding observation")
            seen[a]=t

    def _reservoir(self, x, assets, times, history=None):
        state_by_asset={} if history is None else {k:v[1].copy() for k,v in history.items()}
        last_time={} if history is None else {k:v[0] for k,v in history.items()}
        out=np.empty((len(x),self.config.reservoir_size))
        self._check_order(assets,times)
        for i,(row,a,t) in enumerate(zip(x,assets,times)):
            if a in last_time and t<=last_time[a]:
                raise ValueError("prediction times must follow training history per asset")
            old=state_by_asset.get(a,np.zeros(self.config.reservoir_size))
            updated=(1-self.config.leak)*old+self.config.leak*np.tanh(row@self.win+old@self.w)
            out[i]=updated
            state_by_asset[a]=updated
            last_time[a]=t
        return out,{a:(last_time[a],v.copy()) for a,v in state_by_asset.items()}

    def _memory(self, representations, times):
        out=np.zeros((len(representations),3))
        if not self.use_memory: return out
        for i,(z,t) in enumerate(zip(representations,times)):
            valid=np.flatnonzero(self.train_times<t-self.config.memory_gap)
            if len(valid)==0: continue
            distance=np.sum((self.train_representations[valid]-z)**2,axis=1)
            selected=np.argsort(distance,kind="stable")[:self.config.memory_k]
            idx=valid[selected]; d=distance[selected]
            weights=np.exp(-(d-d.min())/(np.median(d)+1e-8)); weights/=weights.sum()
            mean=weights@self.train_y[idx]
            out[i]=[mean,np.sqrt(weights@(self.train_y[idx]-mean)**2),len(idx)/self.config.memory_k]
        return out

    def _representation(self,x,assets,times,history=None):
        scaled=self.scaler.transform(x)
        latent=self.pca.transform(scaled)
        reservoir,ending=self._reservoir(scaled,assets,times,history)
        return scaled,np.c_[latent,reservoir],ending

    def fit(self,x,y,assets,times):
        x,y,assets,times=np.asarray(x,float),np.asarray(y,float),np.asarray(assets),np.asarray(times)
        if x.ndim!=2 or x.shape[1]==0 or len(x)<2 or not (len(y)==len(assets)==len(times)==len(x)):
            raise ValueError("shape mismatch")
        if not np.isfinite(x).all() or not np.isfinite(y).all() or not np.isfinite(times).all():
            raise ValueError("nonfinite observations")
        self._check_order(assets,times)
        c=self.config
        if c.reservoir_size<=0 or c.memory_k<=0 or c.memory_gap<0 or not 0<c.spectral_radius<1 or not 0<c.leak<=1:
            raise ValueError("invalid config")
        rng=np.random.default_rng(c.seed)
        self.scaler=StandardScaler().fit(x)
        transformed=self.scaler.transform(x)
        self.pca=PCA(n_components=min(c.latent_dim,*transformed.shape),random_state=c.seed).fit(transformed)
        n=c.reservoir_size
        self.win=rng.normal(0,c.input_scale/np.sqrt(x.shape[1]),(x.shape[1],n))
        raw=rng.normal(0,1/np.sqrt(n),(n,n))
        self.w=raw*(c.spectral_radius/max(abs(np.linalg.eigvals(raw))))
        scaled,reps,self.train_history=self._representation(x,assets,times)
        self.train_representations=reps.copy()
        self.train_times=times.copy(); self.train_y=y.copy()
        training_features=np.c_[scaled,reps,self._memory(reps,times)]
        self.model=XGBRegressor(objective="reg:squarederror",n_estimators=c.n_estimators,
                                max_depth=3,learning_rate=.04,reg_lambda=5.,
                                random_state=c.seed,n_jobs=1)
        self.model.fit(training_features,y)
        return self

    def predict(self,x,assets,times):
        x,assets,times=np.asarray(x,float),np.asarray(assets),np.asarray(times)
        if x.ndim!=2 or len(x)!=len(assets) or len(x)!=len(times) or not np.isfinite(x).all():
            raise ValueError("invalid inference data")
        scaled,reps,_=self._representation(x,assets,times,self.train_history)
        return self.model.predict(np.c_[scaled,reps,self._memory(reps,times)])
