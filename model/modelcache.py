"""Content-addressed cache of trained models (Oct 19, 2026).

A model is reused only when everything it was trained from is byte-for-byte the same: the training matrix, the
target, the weights, the parameters, the number of rounds, the seed and the library version. The Thursday, Sunday
and Sunday-inactives runs of one week train on exactly the same games, and the ROS rate models train on past seasons
only (the same all season), so in a working folder that is kept between runs those fits are loaded instead of
redone. Projections are identical either way (a reloaded model predicts exactly what the original did).

Folder: $NFL_MODEL_CACHE (default cache/models next to this file); NFL_MODEL_CACHE=off turns it off.
Oldest files are removed once the folder passes $NFL_MODEL_CACHE_MB (default 600 MB).
"""
import hashlib
import json
import os
import time

import numpy as np

_DIR = os.environ.get("NFL_MODEL_CACHE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "models"))
ON = _DIR.lower() not in ("off", "0", "none", "")
MAX_MB = float(os.environ.get("NFL_MODEL_CACHE_MB", "600"))
stats = {"hit": 0, "miss": 0}


def _h(*parts):
    h = hashlib.sha1()
    for p in parts:
        if p is None:
            h.update(b"<none>")
        elif isinstance(p, np.ndarray):
            a = np.ascontiguousarray(p)
            h.update(str((a.dtype.str, a.shape)).encode())
            h.update(a.view(np.uint8).ravel())
        else:
            h.update(json.dumps(p, sort_keys=True, default=str).encode())
    return h.hexdigest()


def data_key(Xmat, y, weight=None, cols=None):
    """Fingerprint of one training set (computed once, shared by every seed / member fitted on it)."""
    return _h(np.asarray(Xmat), np.asarray(y, dtype=np.float64),
              None if weight is None else np.asarray(weight, dtype=np.float64), list(cols) if cols is not None else None)


def _path(kind, key):
    return os.path.join(_DIR, f"{kind}_{key}")


def lgb_train(lgb, params, ds, rounds, dkey, **kw):
    """lgb.train(params, ds, rounds, **kw) with caching (no callbacks / validation sets)."""
    if not ON:
        return lgb.train(params, ds, rounds, **kw)
    key = _h("lgb", lgb.__version__, dkey, params, int(rounds), sorted(kw))
    p = _path("lgb", key) + ".txt"
    if os.path.exists(p):
        stats["hit"] += 1
        os.utime(p)
        return lgb.Booster(model_file=p)
    stats["miss"] += 1
    b = lgb.train(params, ds, rounds, **kw)
    _save(p, lambda tmp: b.save_model(tmp))
    return b


def cached_value(name, dkey, extra, fn):
    """Small JSON-able results (e.g. an early-stopped round count)."""
    if not ON:
        return fn()
    p = _path("val", _h(name, dkey, extra)) + ".json"
    if os.path.exists(p):
        stats["hit"] += 1
        return json.load(open(p))["v"]
    stats["miss"] += 1
    v = fn()
    _save(p, lambda tmp: json.dump({"v": v}, open(tmp, "w")))
    return v


def cat_model(params, dkey, iters, fit):
    """CatBoost model: fit() -> fitted CatBoostRegressor; cached as .cbm."""
    if not ON:
        return fit()
    import catboost
    from catboost import CatBoostRegressor
    p = _path("cat", _h("cat", catboost.__version__, dkey, params, int(iters))) + ".cbm"
    if os.path.exists(p):
        stats["hit"] += 1
        os.utime(p)
        m = CatBoostRegressor()
        m.load_model(p)
        return m
    stats["miss"] += 1
    m = fit()
    _save(p, lambda tmp: m.save_model(tmp))
    return m


def _save(p, write):
    try:
        os.makedirs(_DIR, exist_ok=True)
        tmp = p + f".tmp{os.getpid()}"
        write(tmp)
        os.replace(tmp, p)
        _prune()
    except OSError:
        pass   # a cache that can't be written just means the next run trains again


_last_prune = [0.0]


def _prune():
    if time.time() - _last_prune[0] < 30:
        return
    _last_prune[0] = time.time()
    fs = [os.path.join(_DIR, f) for f in os.listdir(_DIR)]
    fs = sorted(((os.path.getmtime(f), os.path.getsize(f), f) for f in fs if os.path.isfile(f)))
    tot = sum(s for _, s, _ in fs)
    for _, s, f in fs:
        if tot <= MAX_MB * 1e6:
            break
        os.remove(f)
        tot -= s
