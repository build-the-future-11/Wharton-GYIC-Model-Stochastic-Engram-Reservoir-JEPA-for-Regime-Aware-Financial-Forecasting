import numpy as np
import pytest
from engram_baseline import Config, EngramBaseline

def synthetic():
    rng=np.random.default_rng(123)
    a=np.repeat(np.array([0,1]),25)
    t=np.tile(np.arange(25),2)
    x=rng.normal(size=(50,4))
    y=.2*x[:,0]+rng.normal(0,.05,50)
    return x,y,a,t

def test_continuity_and_interleaved_assets():
    x,y,a,t=synthetic()
    c=Config(reservoir_size=8,n_estimators=8)
    model=EngramBaseline(c).fit(x[t<15],y[t<15],a[t<15],t[t<15])
    future=(t>=15)
    scaled,reps,_=model._representation(x[future],a[future],t[future],model.train_history)
    first_idx=np.flatnonzero((a[future]==0)&(t[future]==15))[0]
    previous=model.train_history[0][1]
    expected=(1-c.leak)*previous+c.leak*np.tanh(scaled[first_idx]@model.win+previous@model.w)
    np.testing.assert_allclose(reps[first_idx,-c.reservoir_size:],expected)
    assert model.predict(x[future],a[future],t[future]).shape==(future.sum(),)

def test_prediction_rejects_reordered_history():
    x,y,a,t=synthetic()
    model=EngramBaseline(Config(reservoir_size=8,n_estimators=8)).fit(x[t<15],y[t<15],a[t<15],t[t<15])
    with pytest.raises(ValueError):
        model.predict(x[:1],a[:1],t[:1])

def test_interleaved_asset_training_equivalence():
    x,y,a,t=synthetic()
    config=Config(reservoir_size=8,n_estimators=8)
    m=EngramBaseline(config).fit(x,y,a,t)
    order=np.lexsort((a,t))
    n=EngramBaseline(config).fit(x[order],y[order],a[order],t[order])
    np.testing.assert_allclose(m.train_representations[order],n.train_representations,atol=1e-12)
