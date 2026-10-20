"""Cross-sectional rank IC, one-period long-short returns, and paired block bootstrap.
Input scores and forward returns MUST be point-in-time aligned by the caller.
"""
import numpy as np
import pandas as pd

def rank_ic_and_backtest(df, fee_bps=5., side_fraction=.2):
    required={'date','asset','score','forward_return'}
    if not required.issubset(df): raise ValueError(f"missing {required-set(df)}")
    if df.duplicated(['date','asset']).any(): raise ValueError("duplicate date/asset")
    if not np.isfinite(df[['score','forward_return']].to_numpy(float)).all(): raise ValueError("nonfinite")
    if not 0 < side_fraction <= .5 or fee_bps < 0: raise ValueError("invalid parameters")
    output=[]
    for date, block in df.groupby('date',sort=True):
        block=block.sort_values(['score','asset'],kind='stable')
        n=len(block); k=max(1,int(np.floor(n*side_fraction)))
        if n < max(3,2*k): raise ValueError("too few assets")
        weights=np.zeros(n); weights[:k]=-.5/k; weights[-k:]=.5/k
        gross=float(weights@block.forward_return.to_numpy(float))
        cost=float(2*fee_bps*1e-4) # full exit/reentry turnover convention
        ic=block.score.rank().corr(block.forward_return.rank()) if block.score.nunique()>1 and block.forward_return.nunique()>1 else np.nan
        output.append((date,gross,cost,gross-cost,ic))
    return pd.DataFrame(output,columns=['date','gross_return','cost_return','net_return','rank_ic'])

def block_bootstrap_mean(values, block_size=5, repetitions=2000, seed=42):
    y=np.asarray(values,float)
    if y.ndim!=1 or len(y)<2 or not np.isfinite(y).all() or not 1<=block_size<=len(y) or repetitions<100:
        raise ValueError("invalid bootstrap input")
    n=len(y); rng=np.random.default_rng(seed); means=np.empty(repetitions)
    for j in range(repetitions):
        starts=rng.integers(0,n,size=int(np.ceil(n/block_size)))
        idx=np.concatenate([(s+np.arange(block_size))%n for s in starts])[:n]
        means[j]=y[idx].mean()
    return dict(mean=float(y.mean()),ci_low=float(np.quantile(means,.025)),ci_high=float(np.quantile(means,.975)),n_dates=n)

def paired_comparison(challenger, baseline, metric='net_return', **kwargs):
    merged=challenger[['date',metric]].merge(baseline[['date',metric]],on='date',suffixes=('_a','_b'),validate='one_to_one')
    if len(merged)!=len(challenger) or len(merged)!=len(baseline): raise ValueError("date mismatch")
    return block_bootstrap_mean(merged[f'{metric}_a']-merged[f'{metric}_b'],**kwargs)
