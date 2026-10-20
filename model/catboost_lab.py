"""CatBoost learner for the v5 harness (a second model family to average with LightGBM)."""
import numpy as np
from catboost import CatBoostRegressor, Pool

CAT_PARAMS = dict(loss_function="RMSE", learning_rate=0.08, depth=6, l2_leaf_reg=5, border_count=64,
                  thread_count=2, verbose=False, allow_writing_files=False, bootstrap_type="Bernoulli",
                  subsample=0.8, rsm=0.5)


def cat_fit(tr, va, X, target, seed, iters=None, valid_pred=False, weight=None):
    p = dict(CAT_PARAMS, random_seed=seed)
    trp = Pool(tr[X].astype(np.float32).values, tr[target].values, weight=weight)
    if iters is None:
            m = CatBoostRegressor(iterations=3000, od_type="Iter", od_wait=100, **p)
            vap = Pool(va[X].astype(np.float32).values, va[target].values)
            m.fit(trp, eval_set=vap, use_best_model=True)
            r = m.get_best_iteration() + 1
            pv = m.predict(va[X].astype(np.float32).values) if valid_pred else None
            return m, pv, r
    m = CatBoostRegressor(iterations=iters, **p)
    m.fit(trp)
    return _Wrap(m), None, iters


class _Wrap:
    def __init__(self, m):
        self.m = m

    def predict(self, X):
        return self.m.predict(np.asarray(X))
