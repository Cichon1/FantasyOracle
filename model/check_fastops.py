"""Check that the optimised feature build gives exactly the same feature frame as a reference frame.

  python check_fastops.py REF_FRAME.parquet NEW_FRAME.parquet
Prints columns that differ (max absolute difference, NaN mismatches); exit code 1 if any.
"""
import sys

import numpy as np
import pandas as pd

a = pd.read_parquet(sys.argv[1])
b = pd.read_parquet(sys.argv[2])
bad = 0
if a.shape != b.shape or list(a.columns) != list(b.columns):
    print("shape / columns differ:", a.shape, b.shape, set(a.columns) ^ set(b.columns))
    bad = 1
for c in [c for c in a.columns if c in b.columns]:
    x, y = a[c], b[c]
    if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y):
        xv, yv = x.to_numpy(dtype=float), y.to_numpy(dtype=float)
        nan = np.isnan(xv) != np.isnan(yv)
        d = np.nanmax(np.abs(xv - yv)) if (~np.isnan(xv) & ~np.isnan(yv)).any() else 0.0
        if nan.any() or d > 0:
            print(f"{c}: max diff {d:.3g}, NaN mismatches {int(nan.sum())}")
            bad = 1
    elif not x.astype(object).where(x.notna(), None).equals(y.astype(object).where(y.notna(), None)):
        print(f"{c}: values differ")
        bad = 1
print("IDENTICAL" if not bad else "DIFFERENT", a.shape)
sys.exit(bad)
