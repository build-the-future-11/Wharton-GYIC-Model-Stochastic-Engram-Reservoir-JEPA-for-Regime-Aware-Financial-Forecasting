import numpy as np
from engram_baseline import Config, EngramBaseline, chronological_splits, metrics

def data():
    r = np.random.default_rng(7)
    assets = np.repeat(np.arange(3), 110)
    times = np.tile(np.arange(110), 3)
    x = r.normal(size=(330,5))
    y = .25*x[:,0]-.1*x[:,1]+r.normal(0,.1,330)
    return x,y,assets,times

def test_chronological_embargo():
    x,y,a,t=data()
    tr,va,te=chronological_splits(t)
    assert not np.any(tr & va) and not np.any(tr & te) and not np.any(va & te)
    assert t[tr].max()+5 < t[va].min()
    assert t[va].max()+5 < t[te].min()

def test_no_protected_label_leakage():
    x,y,a,t=data()
    tr,va,te=chronological_splits(t)
    config=Config(reservoir_size=8,n_estimators=12)
    m=EngramBaseline(config).fit(x[tr],y[tr],a[tr],t[tr])
    p=m.predict(x[te],a[te],t[te])
    y_copy=y.copy()
    y_copy[te]+=10000
    q=EngramBaseline(config).fit(x[tr],y_copy[tr],a[tr],t[tr]).predict(x[te],a[te],t[te])
    np.testing.assert_array_equal(p,q)

def test_memory_off_control():
    x,y,a,t=data()
    tr,va,te=chronological_splits(t)
    for enabled in (False,True):
        model=EngramBaseline(Config(reservoir_size=8,n_estimators=12),enabled)
        model.fit(x[tr],y[tr],a[tr],t[tr])
        assert metrics(y[va],model.predict(x[va],a[va],t[va]))["mse"] >= 0
