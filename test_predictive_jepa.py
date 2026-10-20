import numpy as np
import pytest
from predictive_jepa import PredictiveJEPA,JEPAConfig

def panel():
    r=np.random.default_rng(3)
    return r.normal(size=(75,5)),np.repeat(np.arange(3),25),np.tile(np.arange(25),3)

def test_train_and_determinism():
    x,a,t=panel()
    c=JEPAConfig(epochs=3,hidden_dim=12,latent_dim=6,batch_size=32)
    m=PredictiveJEPA(c).fit(x,a,t)
    n=PredictiveJEPA(c).fit(x,a,t)
    assert np.isfinite(m.loss_history).all()
    assert m.transform(x).shape==(75,6)
    np.testing.assert_allclose(m.transform(x),n.transform(x),atol=1e-6)

def test_protected_inputs_not_required_for_training():
    x,a,t=panel();mask=t<20
    m=PredictiveJEPA(JEPAConfig(epochs=2,hidden_dim=12,latent_dim=6)).fit(x[mask],a[mask],t[mask])
    assert np.isfinite(m.transform(x[~mask])).all()
    with pytest.raises(ValueError):m.transform(np.array([[np.nan]*5]))

def test_rejects_bad_time():
    x,a,t=panel();t[2]=t[1]
    with pytest.raises(ValueError):
        PredictiveJEPA(JEPAConfig(epochs=1)).fit(x,a,t)
