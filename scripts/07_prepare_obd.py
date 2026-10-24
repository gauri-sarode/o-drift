"""Prepare the Open Bandit semi-synthetic environment (protocol splits.real; decision log 23).

Reads the random-policy Men campaign log and writes data/obd_men.npz with
  X      (n, 8)  standardized PCA contexts, in real time order
  block  (n,)    0..12 equal-count time blocks (12 = target period)
  W, b, Q        base reward logits f(x, a) = x W[:, a] + b[a] + (x^2) Q[:, a], fit on real
                 clicks, intercepts shifted so that the mean CTR is 0.05

    .venv/bin/python scripts/07_prepare_obd.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import brentq
from scipy.special import expit
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data/open_bandit_dataset/random/men/men.csv"
OUT = ROOT / "data/obd_men.npz"
D, N_BLOCKS, CTR = 8, 13, 0.05


def main():
    df = pd.read_csv(CSV)
    df["t"] = pd.to_datetime(df["timestamp"], format="ISO8601")
    df = df.sort_values("t", kind="stable").reset_index(drop=True)
    onehot = pd.get_dummies(df[[f"user_feature_{i}" for i in range(4)]].astype(str)).to_numpy(float)
    aff = df[[c for c in df.columns if c.startswith("user-item_affinity_")]].to_numpy(float)
    raw = np.c_[onehot, np.log1p(aff.sum(1)), np.log1p(aff.max(1))]
    raw = (raw - raw.mean(0)) / np.maximum(raw.std(0), 1e-9)
    Z = PCA(n_components=D, random_state=0).fit_transform(raw)
    X = Z / Z.std(0)
    A, R = df["item_id"].to_numpy(), df["click"].to_numpy()
    n_actions = int(A.max()) + 1
    assert np.allclose(df["propensity_score"], 1 / n_actions), "random policy should be uniform"

    # base reward: per-action logistic on [x, s(x^2)] with s() standardization (heavy-tailed PCs of
    # one-hot features make raw x^2 extreme); L2 strength by 3-fold CV log-loss on the real clicks
    # (amendment 2026-10-24, before any OBD result: C=1 on raw x^2 overfit, logit sd ~6)
    X2 = X ** 2
    m2, s2 = X2.mean(0), X2.std(0)
    onehot_a = np.eye(n_actions)[A]
    F = np.hstack([onehot_a, (X[:, :, None] * onehot_a[:, None, :]).reshape(len(A), -1),
                   (((X2 - m2) / s2)[:, :, None] * onehot_a[:, None, :]).reshape(len(A), -1)])
    cv = LogisticRegressionCV(Cs=[1e-4, 1e-3, 1e-2, 1e-1, 1.0], cv=3, scoring="neg_log_loss", max_iter=5000,
                              fit_intercept=False, random_state=0).fit(F, R)
    print(f"CV-chosen C = {cv.C_[0]:g}; mean CV log-loss by C: "
          + ", ".join(f"{c:g}:{-v:.5f}" for c, v in zip(cv.Cs_, cv.scores_[1].mean(0))))
    coef = cv.coef_.ravel()
    b = coef[:n_actions]
    W = coef[n_actions:n_actions * (1 + D)].reshape(D, n_actions)
    Qs = coef[n_actions * (1 + D):].reshape(D, n_actions)
    Q = Qs / s2[:, None]           # back to raw x^2 scale: (x^2 - m2)/s2 . Qs = x^2 . Q - m2 . Q
    b = b - (m2 / s2) @ Qs
    logits = X @ W + b + X2 @ Q  # all actions, real contexts
    shift = brentq(lambda c: expit(logits + c).mean() - CTR, -10, 10)
    b = b + shift

    block = np.minimum(np.arange(len(X)) * N_BLOCKS // len(X), N_BLOCKS - 1)
    np.savez_compressed(OUT, X=X.astype(np.float64), block=block, W=W, b=b, Q=Q,
                        t=df["t"].astype("int64").to_numpy())
    print(f"rows {len(X)}, actions {n_actions}, real CTR {R.mean():.4f}, intercept shift {shift:+.3f}")
    print(f"block sizes {np.bincount(block).tolist()}")
    print(f"block start times {[str(df['t'].iloc[np.argmax(block == k)])[:16] for k in range(N_BLOCKS)]}")
    q = expit(X @ W + b + (X ** 2) @ Q)
    print(f"mean CTR after shift {q.mean():.4f}; best-item CTR {q.max(1).mean():.4f}; "
          f"nonlinear share of logit var {np.var((X ** 2) @ Q) / np.var(logits):.2f}; logit sd {logits.std():.2f}; "
          f"CTR pct 1/50/99 {np.percentile(q, [1, 50, 99]).round(4).tolist()}")
    # natural context drift across blocks: mean shift of each block vs the target block
    mu_t = X[block == N_BLOCKS - 1].mean(0)
    print("context mean shift vs target (L2):",
          [round(float(np.linalg.norm(X[block == k].mean(0) - mu_t)), 3) for k in range(N_BLOCKS)])


if __name__ == "__main__":
    main()
