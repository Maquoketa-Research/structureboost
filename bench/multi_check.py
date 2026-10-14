"""StructureBoostMulti on synthetic data (numerical features with NaNs plus a string categorical): fit, predict, save
the probabilities, so builds can be compared exactly. String categoricals iterate in hash order, so run with a fixed
PYTHONHASHSEED (and without python -I, which ignores it):

    PYTHONHASHSEED=0 python bench/multi_check.py out.npy
"""
import sys
import time

import numpy as np
import pandas as pd
import structureboost as stb

rng = np.random.default_rng(0)
n = 20000
X = pd.DataFrame({f"x{i}": rng.normal(size=n) for i in range(8)})
X.loc[rng.random(n) < 0.1, "x1"] = np.nan
X["cat"] = pd.Series(rng.choice(list("abcdef"), n)).astype(object)   # object, so get_basic_config sees a categorical
logit = X.x0.fillna(0) + 0.5 * (X.cat == "a") - 0.3 * X.x2
y = np.digitize(logit + rng.normal(size=n) * 0.5, [-1, 0, 1]).astype(np.int32)
fc = stb.get_basic_config(X, stb.default_config_dict())
m = stb.StructureBoostMulti(num_trees=40, feature_configs=fc, num_classes=4, learning_rate=0.1, max_depth=4)
t = time.time()
m.fit(X.iloc[:15000], y[:15000])
fit_s = time.time() - t
p = m.predict(X.iloc[15000:])
np.save(sys.argv[1], p)
print(f"fit {fit_s:.1f}s  cat config {fc['cat']['feature_type']}  first row {p[0].round(4).tolist()}")
