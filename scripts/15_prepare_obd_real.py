"""Prepare the real-reward Open Bandit evaluation (protocol v6.real; decision log 38).

For each campaign (men, women, all) reads the uniform-random log and the Bernoulli Thompson
Sampling (BTS) log of the same A/B test and writes data/obd_real_{campaign}.npz with
  X       (n, 11)  random-log contexts in real time order: 8 standardized PCA components of the
                   user features (as scripts/07_prepare_obd.py) + one-hot position (3)
  A, R    (n,)     real items and real clicks of the random log (propensity 1/n_items)
  block   (n,)     0..18, equal-count time blocks of the random log
  pi      (19, 3, n_items)  BTS item frequencies per position within each block's time span:
                   the evaluation policy pi_b(a | position) for target block b
  ctr     (19,)    BTS on-policy click rate within each block's time span (ground truth)
  ctr_n   (19,)    number of BTS impressions behind ctr
  t, edges         event times and block start times (microseconds since the epoch)

BTS is context-free, so pi_b depends on the position only. Nothing here is synthetic.

    .venv/bin/python scripts/15_prepare_obd_real.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/open_bandit_dataset"
D, N_BLOCKS = 8, 19
CAMPAIGNS = ("men", "women", "all")


def contexts(df: pd.DataFrame) -> np.ndarray:
    onehot = pd.get_dummies(df[[f"user_feature_{i}" for i in range(4)]].astype(str)).to_numpy(float)
    aff = df[[c for c in df.columns if c.startswith("user-item_affinity_")]].to_numpy(float)
    raw = np.c_[onehot, np.log1p(aff.sum(1)), np.log1p(aff.max(1))]
    raw = (raw - raw.mean(0)) / np.maximum(raw.std(0), 1e-9)
    Z = PCA(n_components=D, random_state=0).fit_transform(raw)
    pos = np.eye(3)[df["position"].to_numpy() - 1]
    return np.c_[Z / Z.std(0), pos]


def main():
    for camp in CAMPAIGNS:
        df = pd.read_csv(RAW / f"random/{camp}/{camp}.csv", index_col=0)
        df["t"] = pd.to_datetime(df["timestamp"], format="ISO8601").astype("int64")
        df = df.sort_values("t", kind="stable").reset_index(drop=True)
        n_items = int(df["item_id"].max()) + 1
        assert np.allclose(df["propensity_score"], 1 / n_items), "random policy should be uniform"
        X = contexts(df)
        block = np.minimum(np.arange(len(df)) * N_BLOCKS // len(df), N_BLOCKS - 1)
        t = df["t"].to_numpy()
        edges = np.array([t[block == b].min() for b in range(N_BLOCKS)] + [np.iinfo(np.int64).max])

        counts = np.zeros((N_BLOCKS, 3, n_items))
        clicks, imps = np.zeros(N_BLOCKS), np.zeros(N_BLOCKS)
        for ch in pd.read_csv(RAW / f"bts/{camp}/{camp}.csv", usecols=["timestamp", "item_id", "position", "click"],
                              chunksize=2_000_000):
            tb = pd.to_datetime(ch["timestamp"], format="ISO8601").astype("int64").to_numpy()
            b = np.searchsorted(edges, tb, side="right") - 1
            ok = b >= 0  # BTS impressions before the first random impression are dropped
            b, a, p, c = b[ok], ch["item_id"].to_numpy()[ok], ch["position"].to_numpy()[ok] - 1, ch["click"].to_numpy()[ok]
            np.add.at(counts, (b, p, a), 1)
            np.add.at(clicks, b, c)
            np.add.at(imps, b, 1)
        pi = counts / counts.sum(axis=2, keepdims=True)
        ctr = clicks / imps
        out = ROOT / f"data/obd_real_{camp}.npz"
        np.savez_compressed(out, X=X, A=df["item_id"].to_numpy(), R=df["click"].to_numpy().astype(np.int8),
                            block=block, pi=pi, ctr=ctr, ctr_n=imps, t=t, edges=edges[:-1])
        print(f"{camp}: random rows {len(df)}, items {n_items}, random CTR {df['click'].mean():.5f}; "
              f"BTS rows used {int(imps.sum())}, BTS CTR {clicks.sum() / imps.sum():.5f}; "
              f"block size {np.bincount(block).min()}-{np.bincount(block).max()}; "
              f"span {pd.Timestamp(t[0], unit='us')} .. {pd.Timestamp(t[-1], unit='us')}", flush=True)


if __name__ == "__main__":
    main()
