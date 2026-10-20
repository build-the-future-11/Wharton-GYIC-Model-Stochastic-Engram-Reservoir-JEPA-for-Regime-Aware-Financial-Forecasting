import numpy as np
import pandas as pd
import pytest
from cross_sectional_evaluation import rank_ic_and_backtest, block_bootstrap_mean, paired_comparison

def panel():
    return pd.DataFrame({'date':np.repeat(np.arange(12),10),'asset':np.tile(np.arange(10),12),
     'score':np.tile(np.arange(10),12),'forward_return':np.tile(np.arange(10)/100,12)})

def test_rank_ic_and_exact_cost():
    result=rank_ic_and_backtest(panel(),fee_bps=10,side_fraction=.2)
    np.testing.assert_allclose(result.rank_ic,1)
    np.testing.assert_allclose(result.gross_return,.04)
    np.testing.assert_allclose(result.cost_return,.002)
    np.testing.assert_allclose(result.net_return,.038)

def test_paired_comparison():
    result=rank_ic_and_backtest(panel())
    z=paired_comparison(result,result,block_size=3,repetitions=100)
    assert z['mean']==z['ci_low']==z['ci_high']==0

def test_invalid_inputs():
    x=panel(); x.loc[0,'score']=np.nan
    with pytest.raises(ValueError): rank_ic_and_backtest(x)
    with pytest.raises(ValueError): block_bootstrap_mean([1],repetitions=100)
