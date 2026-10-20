"""Feature-space predictive JEPA, trained strictly on supplied training observations."""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler

@dataclass(frozen=True)
class JEPAConfig:
    latent_dim: int = 16
    hidden_dim: int = 64
    epochs: int = 35
    batch_size: int = 128
    learning_rate: float = .001
    ema: float = .99
    variance_weight: float = .1
    seed: int = 13

class _Encoder(nn.Module):
    def __init__(self,d,c):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,c.hidden_dim),nn.LayerNorm(c.hidden_dim),
                               nn.GELU(),nn.Linear(c.hidden_dim,c.latent_dim))
    def forward(self,x): return self.net(x)

class PredictiveJEPA:
    def __init__(self,config=JEPAConfig()): self.config=config
    @staticmethod
    def _pairs(assets,times):
        latest={}; pairs=[]
        for i,(a,t) in enumerate(zip(assets,times)):
            if a in latest:
                previous,previous_t=latest[a]
                if t<=previous_t: raise ValueError("nonincreasing per-asset time")
                pairs.append((previous,i))
            latest[a]=(i,t)
        if not pairs: raise ValueError("need sequential observations")
        return np.asarray(pairs,dtype=int)
    def fit(self,x,assets,times):
        x=np.asarray(x,dtype=np.float32)
        assets=np.asarray(assets);times=np.asarray(times)
        if x.ndim!=2 or x.shape[1]==0 or len(x)!=len(assets) or len(x)!=len(times) or not np.isfinite(x).all():
            raise ValueError("invalid inputs")
        c=self.config
        if c.epochs<1 or c.batch_size<1 or not 0<=c.ema<1: raise ValueError("invalid config")
        pairs=self._pairs(assets,times)
        self.scaler=StandardScaler().fit(x)
        scaled=torch.from_numpy(self.scaler.transform(x).astype(np.float32))
        torch.manual_seed(c.seed)
        self.encoder=_Encoder(x.shape[1],c)
        self.target=deepcopy(self.encoder)
        for p in self.target.parameters(): p.requires_grad_(False)
        self.predictor=nn.Sequential(nn.Linear(c.latent_dim,c.hidden_dim),
                                     nn.GELU(),nn.Linear(c.hidden_dim,c.latent_dim))
        parameters=list(self.encoder.parameters())+list(self.predictor.parameters())
        optimizer=torch.optim.AdamW(parameters,lr=c.learning_rate)
        rng=np.random.default_rng(c.seed)
        self.loss_history=[]
        for epoch in range(c.epochs):
            indices=rng.permutation(len(pairs)); total=0.; count=0
            for start in range(0,len(indices),c.batch_size):
                batch=pairs[indices[start:start+c.batch_size]]
                context=self.encoder(scaled[batch[:,0]])
                pred=self.predictor(context)
                with torch.no_grad(): target=self.target(scaled[batch[:,1]])
                loss=(pred-target).square().mean()+c.variance_weight*torch.relu(
                    1-torch.sqrt(context.var(dim=0,unbiased=False)+1e-4)).mean()
                optimizer.zero_grad();loss.backward()
                nn.utils.clip_grad_norm_(parameters,5.)
                optimizer.step()
                with torch.no_grad():
                    for p,q in zip(self.encoder.parameters(),self.target.parameters()):
                        q.lerp_(p,1-c.ema)
                total+=float(loss.detach())*len(batch);count+=len(batch)
            self.loss_history.append(total/count)
        self.encoder.eval();self.predictor.eval();self.target.eval()
        return self
    def transform(self,x):
        x=np.asarray(x,dtype=np.float32)
        if x.ndim!=2 or not np.isfinite(x).all():raise ValueError("invalid features")
        with torch.no_grad():
            return self.encoder(torch.from_numpy(
                self.scaler.transform(x).astype(np.float32))).numpy()
