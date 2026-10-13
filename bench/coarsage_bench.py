"""Coarsage correctness and speed harness.

    python bench/coarsage_bench.py --train train.parquet --test test.parquet --target y \
        --cyclic hour_target:24,hour_last:24,hour_origin:24,dow_target:7 --trees 50 --save golden.npz
    python bench/coarsage_bench.py ... --check golden.npz      # after a change: same distributions?

Seeds Python's and NumPy's global generators (the library draws from both), fits Coarsage with library defaults
(depth 3, learning rate 0.05), predicts the test set and records the bin edges and per-row bin probabilities. --check
compares them with a saved run: speed-only changes must reproduce them to floating-point tolerance.
"""
import argparse
import random
import sys
import time

import numpy as np
import pandas as pd

import structureboost as stb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--target", default="y")
    ap.add_argument("--cyclic", default="", help="name:period,... features to treat as cycles")
    ap.add_argument("--trees", type=int, default=50)
    ap.add_argument("--save", default="")
    ap.add_argument("--check", default="")
    ap.add_argument("--rtol", type=float, default=1e-7)
    a = ap.parse_args()
    tr, te = pd.read_parquet(a.train), pd.read_parquet(a.test)
    cyc = dict((c.split(":")[0], int(c.split(":")[1])) for c in a.cyclic.split(",") if c)
    X, Xt = tr.drop(columns=[a.target]), te.drop(columns=[a.target])
    for c in cyc:
        X[c], Xt[c] = X[c].astype(int), Xt[c].astype(int)
    random.seed(0)
    np.random.seed(0)
    fc = stb.get_basic_config(X, stb.default_config_dict())
    for c, n in cyc.items():
        fc[c] = {"feature_type": "categorical_int", "graph": stb.graphs.cycle_int_graph(0, n - 1), "split_method": "span_tree", "num_span_trees": 1}
    m = stb.Coarsage(num_trees=a.trees, max_depth=3, feature_configs=fc, learning_rate=0.05)
    t0 = time.time()
    m.fit(X, tr[a.target].to_numpy(), verbose=0)
    t_fit = time.time() - t0
    t0 = time.time()
    pg = m.predict_distributions(Xt)
    t_pred = time.time() - t0
    binvec, probmat = np.asarray(pg.binvec), np.asarray(pg.probmat)
    ll = pg.log_loss(te[a.target].to_numpy())
    print(f"fit {t_fit:.1f}s ({t_fit / a.trees:.2f}s/tree, {len(X):,} rows)  predict {t_pred:.1f}s ({len(Xt):,} rows)  "
          f"bins {len(binvec) - 1}  log loss {ll:.6f}")
    if a.save:
        np.savez_compressed(a.save, binvec=binvec, probmat=probmat, log_loss=ll, t_fit=t_fit, t_pred=t_pred)
    if a.check:
        g = np.load(a.check)
        ok = g["binvec"].shape == binvec.shape and np.allclose(g["binvec"], binvec, rtol=a.rtol, atol=1e-12) \
            and np.allclose(g["probmat"], probmat, rtol=a.rtol, atol=1e-12)
        dmax = float(np.max(np.abs(g["probmat"] - probmat))) if g["probmat"].shape == probmat.shape else float("nan")
        print(f"check vs {a.check}: {'IDENTICAL' if ok else 'DIFFERENT'} (max |dp| {dmax:.3g}); speedup fit x{float(g['t_fit']) / t_fit:.2f}, "
              f"predict x{float(g['t_pred']) / t_pred:.2f}")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
