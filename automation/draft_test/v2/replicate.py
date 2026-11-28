import sys; sys.path.insert(0, __import__("os").path.dirname(__import__("os").path.abspath(__file__)))
import numpy as np, planner2 as P2
P2.CFGS = {k: P2.CFGS[k] for k in ["c_adp","plan_knows_league","plan_league_blend","plan_league_te2"]}
_rng = np.random.default_rng
P2.np.random.default_rng = lambda seed=None: _rng([999] + list(seed)) if isinstance(seed, list) else _rng(seed)
P2.run(sys.argv[1], list(range(2013, 2026)), "league", sims=3, tag="rep")
