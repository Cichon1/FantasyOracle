"""Fast grouped window functions (Oct 19, 2026).

`s.groupby(key).transform(lambda x: x.ewm(...).mean())` calls Python once per group (~10,000 players), which made
the feature build spend most of its time in Python overhead. pandas' grouped windows (`groupby(...).ewm/rolling`)
run the same kernels over all groups in one pass: results are bit-for-bit identical (checked in `check_fastops.py`)
and about 10x faster.

Each helper returns a Series aligned to `s` (same index, NaN where a key is missing), like transform() does.
"""
import numpy as np
import pandas as pd


def _keys(key, n):
    keys = key if isinstance(key, (list, tuple)) else [key]
    out = []
    for k in keys:
        out.append(pd.Series(np.asarray(k), index=pd.RangeIndex(n)))
    return out


def _grouped(s, key):
    s2 = pd.Series(np.asarray(s, dtype=float), index=pd.RangeIndex(len(s)))
    return s2, s2.groupby(_keys(key, len(s)), sort=False)


def _back(r, s, nkeys):
    r = r.reset_index(level=list(range(nkeys)), drop=True).reindex(pd.RangeIndex(len(s)))
    return pd.Series(r.values, index=s.index, name=s.name)


def _nkeys(key):
    return len(key) if isinstance(key, (list, tuple)) else 1


def gshift(s, key, n=1):
    """Grouped shift (already vectorised in pandas); returns float."""
    s2, g = _grouped(s, key)
    return pd.Series(g.shift(n).values, index=s.index, name=s.name)


def gewm(s, key, halflife, shift=0):
    """== s.groupby(key).transform(lambda x: x[.shift(shift)].ewm(halflife=halflife, ignore_na=True).mean())"""
    if shift:
        s = gshift(s, key, shift)
    s2, g = _grouped(s, key)
    return _back(g.ewm(halflife=halflife, ignore_na=True).mean(), s, _nkeys(key))


def groll_mean(s, key, window, min_periods=None):
    s2, g = _grouped(s, key)
    return _back(g.rolling(window, min_periods=min_periods).mean(), s, _nkeys(key))


def groll_std(s, key, window, min_periods=None):
    s2, g = _grouped(s, key)
    return _back(g.rolling(window, min_periods=min_periods).std(), s, _nkeys(key))


def gcumsum_prior(s, key):
    """== s.groupby(key).transform(lambda x: x.shift(1).cumsum())"""
    sh = gshift(s, key, 1)
    s2, g = _grouped(sh, key)
    return pd.Series(g.cumsum().values, index=s.index, name=s.name)


def gffill_shift(s, key):
    """== s.groupby(key).transform(lambda x: x.ffill().shift(1)) for numeric s."""
    s2, g = _grouped(s, key)
    f = pd.Series(g.ffill().values, index=s2.index)
    return pd.Series(f.groupby(_keys(key, len(s)), sort=False).shift(1).values, index=s.index, name=s.name)
