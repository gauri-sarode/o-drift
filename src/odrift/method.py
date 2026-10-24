"""O-DRIFT: diagnostics, transported DR, anchor-calibrated bias bounds,
window selection and abstention (proposal §4).

Components are the K historical windows plus the recent anchor (index K).
Each component k yields a transported DR estimate of
    V_{T<-k}(pi) = E_{X~P_T}[sum_a pi(a|X) q_k(X, a)],
the anchor's being unbiased for V_T itself. Write alpha = (beta, alpha_anchor) and d_k = V_k - V_anchor, so the
mixture's transfer bias is beta . b and E[(beta . d)^2] = (beta . b)^2 + Var(beta . d).

Selection ("segment", default) partitions windows into reward-homogeneous
groups by comparing transported values with each other (history-scale
precision), then takes the most recent group the anchor does not contradict,
pooled with the anchor; see `segment_select`.

The "mse" variant minimizes a debiased estimate of the mixture MSE
    alpha^T Sigma alpha + lambda [(beta . d)^2 - Var(beta . d)]   over the simplex.
The "bound" ablation instead penalizes worst-case bounds,
    alpha^T Sigma alpha + lambda (sum_k alpha_k Bbar_k)^2,
which is valid for any alpha but over-penalizes every window by the anchor's
own noise and so collapses onto the anchor.

The interval adds a mixture-level bias allowance |beta . d| + z sd(beta . d),
computed on data independent of beta in the sample-split variant:
    V_alpha +/- (z_v SE_alpha + |beta . d| + z_b sd(beta . d)).
"""

from dataclasses import dataclass, field, replace

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression

from .estimators import effective_sample_size, importance_weights
from .nuisance import crossfit_rewards
from .simulator import LoggedData


# ---------------------------------------------------------------- diagnostics

@dataclass
class Diagnostics:
    n: int
    unsupported_target_mass: float  # pi mass on actions never logged in this window
    propensity_z: float             # z-stat of mean(1/(A * pscore)) - 1  (should be ~0)
    target_weight_z: float          # z-stat of mean(rho) - 1             (should be ~0)
    ess: float                      # ESS of context-ratio x importance weights
    reasons: list = field(default_factory=list)

    @property
    def eligible(self) -> bool:
        return not self.reasons


def _mean_one_z(x: np.ndarray) -> float:
    return float((x.mean() - 1) / (x.std(ddof=1) / np.sqrt(len(x)) + 1e-12))


def diagnose(data: LoggedData, pi_probs: np.ndarray, n_actions: int, weights: np.ndarray,
             z_crit: float, min_ess: float, max_unsupported: float = 0.01) -> Diagnostics:
    """Support, propensity-consistency and ESS checks for one window.

    With correctly logged event-time propensities, E[1/(A mu(A|X))] = 1 and
    E[pi(A|X)/mu(A|X)] = 1. Missing support or propensities recomputed from
    the wrong logger break these identities, which is what the z-tests detect.
    """
    counts = np.bincount(data.A, minlength=n_actions)
    unsupported = float(pi_probs[:, counts == 0].sum(axis=1).mean())
    rho = importance_weights(data, pi_probs)
    d = Diagnostics(
        n=len(data),
        unsupported_target_mass=unsupported,
        propensity_z=_mean_one_z(1.0 / (n_actions * data.pscore)),
        target_weight_z=_mean_one_z(rho),
        ess=effective_sample_size(weights * rho),
    )
    if unsupported > max_unsupported:
        d.reasons.append("target_support")
    if abs(d.propensity_z) > z_crit or abs(d.target_weight_z) > z_crit:
        d.reasons.append("propensity_inconsistent")
    if d.ess < min_ess:
        d.reasons.append("low_ess")
    return d


# ---------------------------------------------------------- density ratios

def context_shift_pvalue(X_src: np.ndarray, X_tgt: np.ndarray) -> float:
    """Likelihood-ratio test of a logistic source-vs-target classifier against
    the intercept-only model (chi-square, d dof). Small p = detectable context shift."""
    from scipy.stats import chi2
    X = np.vstack([X_src, X_tgt])
    y = np.r_[np.zeros(len(X_src)), np.ones(len(X_tgt))]
    clf = LogisticRegression(C=1e6, max_iter=1000).fit(X, y)
    p = np.clip(clf.predict_proba(X)[:, 1], 1e-12, 1 - 1e-12)
    ll = (y * np.log(p) + (1 - y) * np.log(1 - p)).sum()
    p0 = y.mean()
    ll0 = (y * np.log(p0) + (1 - y) * np.log(1 - p0)).sum()
    return float(chi2.sf(2 * (ll - ll0), X.shape[1]))


def context_ratio(X_src: np.ndarray, X_tgt: np.ndarray, n_folds: int = 2, seed: int = 0) -> np.ndarray:
    """Cross-fitted w(x) = p_T(x)/p_k(x) on the source contexts, normalized to mean 1.

    Probabilistic classification (target=1 vs source=0) with a logistic model;
    exact in the simulator's mean-shift family, approximate in general.
    """
    rng = np.random.default_rng(seed)
    fs, ft = rng.permutation(len(X_src)) % n_folds, rng.permutation(len(X_tgt)) % n_folds
    w = np.empty(len(X_src))
    for f in range(n_folds):
        X = np.vstack([X_src[fs != f], X_tgt[ft != f]])
        y = np.r_[np.zeros((fs != f).sum()), np.ones((ft != f).sum())]
        clf = LogisticRegression(max_iter=1000).fit(X, y)
        logit = clf.decision_function(X_src[fs == f])
        w[fs == f] = np.exp(logit) * (fs != f).sum() / (ft != f).sum()
    return w / w.mean()


# ------------------------------------------------------------ transported DR

@dataclass
class Component:
    index: int                # window index; K = anchor
    value: float
    dm_scores: np.ndarray     # m_hat_k(X*_j), length M (shared target contexts)
    corr_scores: np.ndarray   # w rho (R - q_hat), length n_k
    diagnostics: Diagnostics
    members: dict = field(default_factory=dict)  # pooled components: window -> share of records

    @property
    def var(self) -> float:
        return float(self.dm_scores.var(ddof=1) / len(self.dm_scores)
                     + self.corr_scores.var(ddof=1) / len(self.corr_scores))


@dataclass
class Nuisances:
    """Policy-independent fits for one window, reusable across candidate policies."""
    q_src: np.ndarray   # out-of-fold q_hat on the window's own contexts, (n, A)
    q_tgt: np.ndarray   # fold-averaged q_hat on target contexts, (M, A)
    w: np.ndarray       # context ratio p_T/p_k on the window's contexts, (n,)
    w_fn: object = None  # X -> p_T/p_k at arbitrary contexts (None = no detected shift, ratio 1)


def ratio_function(X_src: np.ndarray, X_tgt: np.ndarray):
    """Full-data probabilistic-classification estimate of x -> p_T(x)/p_k(x),
    normalized to mean 1 over the source sample (used to evaluate one window's
    ratio at other windows' records for balanced pooling)."""
    X = np.vstack([X_src, X_tgt])
    y = np.r_[np.zeros(len(X_src)), np.ones(len(X_tgt))]
    clf = LogisticRegression(max_iter=1000).fit(X, y)
    scale = len(X_src) / len(X_tgt)
    norm_ = np.exp(clf.decision_function(X_src)).mean() * scale
    return lambda Z: np.exp(clf.decision_function(Z)) * scale / norm_


def fit_nuisances(data: LoggedData, X_target: np.ndarray, n_actions: int,
                  same_distribution: bool = False, seed: int = 0, context_alpha: float = 0.0) -> Nuisances:
    """context_alpha > 0: transport only if the context-shift test rejects at that level."""
    q_src, q_tgt = crossfit_rewards(data, n_actions, seed=seed, extra_X=X_target)
    if same_distribution or (context_alpha > 0 and context_shift_pvalue(data.X, X_target) >= context_alpha):
        w = np.ones(len(data))
    else:
        w = context_ratio(data.X, X_target, seed=seed)
    return Nuisances(q_src, q_tgt, w)


def transported_dr(data: LoggedData, X_target: np.ndarray, pi, n_actions: int, index: int,
                   z_crit: float, min_ess: float, same_distribution: bool = False,
                   seed: int = 0, nuis: Nuisances | None = None,
                   self_normalize: bool = False) -> Component:
    """V_{T<-k} = mean_j m_k(X*_j) + mean_i w(X_i) rho_i (R_i - q_k(X_i, A_i)).

    self_normalize divides the correction by mean(w rho) (self-normalized DR),
    trading a small O(1/n) bias for lighter tails when rho is heavy-tailed.
    """
    if nuis is None:
        nuis = fit_nuisances(data, X_target, n_actions, same_distribution, seed)
    pi_src = pi.probs(data.X)
    rho = importance_weights(data, pi_src)
    corr = nuis.w * rho * (data.R - nuis.q_src[np.arange(len(data)), data.A])
    if self_normalize:
        corr = corr / (nuis.w * rho).mean()
    dm_scores = (pi.probs(X_target) * nuis.q_tgt).sum(axis=1)
    diag = diagnose(data, pi_src, n_actions, nuis.w, z_crit, min_ess)
    return Component(index, float(dm_scores.mean() + corr.mean()), dm_scores, corr, diag)


def covariance(components: list) -> np.ndarray:
    """Cross-component covariance of the estimates.

    DM parts share the target-context sample, so they covary; correction
    parts use disjoint logged data and are independent.
    """
    dm = np.array([c.dm_scores for c in components])
    M = dm.shape[1]
    S = np.cov(dm) / M if len(components) > 1 else np.array([[dm.var(ddof=1) / M]])
    S = np.atleast_2d(S)
    S[np.diag_indices_from(S)] += [c.corr_scores.var(ddof=1) / len(c.corr_scores) for c in components]
    return S


# ------------------------------------------------- bias bounds and selection

def bias_bounds(components: list, anchor: Component, S_full: np.ndarray, z: float) -> np.ndarray:
    """Bbar_k = |V_k - V_anchor| + z * sd(V_k - V_anchor) for each component.

    An upper confidence bound on |b_k| = |V_{T<-k} - V_T| that holds
    simultaneously over k when z is Bonferroni-adjusted. Zero for the anchor.
    Always >= z * sd(V_anchor): a bound calibrated on the anchor cannot be
    tighter than the anchor itself.
    """
    a = next(i for i, c in enumerate(components) if c is anchor)
    B = np.zeros(len(components))
    for k, c in enumerate(components):
        if c is anchor:
            continue
        var_diff = S_full[k, k] + S_full[a, a] - 2 * S_full[k, a]
        B[k] = abs(c.value - anchor.value) + z * np.sqrt(max(var_diff, 0.0))
    return B


def diff_moments(comps: list, S: np.ndarray) -> tuple:
    """d_k = V_k - V_anchor (anchor last) and Cov(d), embedded in alpha-space.

    Returns (d_tilde, C_tilde) with a trailing zero for the anchor so that
    beta . d = alpha . d_tilde and Var(beta . d) = alpha^T C_tilde alpha.
    """
    h = len(comps) - 1
    P = np.hstack([np.eye(h), -np.ones((h, 1))])
    d = np.r_[[c.value - comps[-1].value for c in comps[:-1]], 0.0]
    C = np.zeros_like(S)
    C[:h, :h] = P @ S @ P.T
    return d, C


def mse_objective(S: np.ndarray, d: np.ndarray, C: np.ndarray, lam: float) -> np.ndarray:
    """Quadratic form of the debiased MSE estimate, projected to the PSD cone."""
    Q = S + lam * (np.outer(d, d) - C)
    Q = (Q + Q.T) / 2
    vals, vecs = np.linalg.eigh(Q)
    return (vecs * np.clip(vals, 1e-12 * max(vals.max(), 1e-12), None)) @ vecs.T


def select_weights(S: np.ndarray, B: np.ndarray, lam: float, Q: np.ndarray | None = None) -> np.ndarray:
    """argmin alpha^T Q alpha over the probability simplex; Q = S + lam B B^T by default."""
    if Q is None:
        Q = S + lam * np.outer(B, B)
    Q = (Q + Q.T) / 2
    n = len(B)
    # warm start: inverse-diagonal weights
    x0 = 1 / np.diag(Q)
    x0 /= x0.sum()
    res = minimize(lambda a: a @ Q @ a, x0, jac=lambda a: 2 * Q @ a, method="SLSQP",
                   bounds=[(0, 1)] * n, constraints=[{"type": "eq", "fun": lambda a: a.sum() - 1}],
                   options={"ftol": 1e-14, "maxiter": 500})
    alpha = np.clip(res.x, 0, None)
    return alpha / alpha.sum()


def homogeneous_groups(values: np.ndarray, S: np.ndarray, z_crit: float) -> list:
    """Greedy recency-seeded partition of windows into reward-homogeneous groups.

    `values`/`S` cover historical components only, ordered oldest -> newest.
    After context transport and propensity correction, transported values of
    windows with the same conditional reward agree up to sampling noise, so a
    significant difference between windows is evidence of reward drift --
    measured at history-scale precision, without involving the anchor.
    Each group is seeded by the most recent unassigned window; remaining
    windows are visited newest-first and added when consistent with the
    group's minimum-variance mean.
    """
    unassigned = list(range(len(values)))[::-1]
    groups = []
    while unassigned:
        group = [unassigned.pop(0)]
        for j in list(unassigned):
            w = select_weights(S[np.ix_(group, group)], np.zeros(len(group)), 0.0)
            m = w @ values[group]
            var = S[j, j] + w @ S[np.ix_(group, group)] @ w - 2 * w @ S[group, j]
            if abs(values[j] - m) <= z_crit * np.sqrt(max(var, 1e-18)):
                group.append(j)
                unassigned.remove(j)
        groups.append(group)
    return groups


def _block_diff_z(values: np.ndarray, S: np.ndarray, left: list, right: list) -> float:
    """z-statistic of the difference between two blocks' minimum-variance means."""
    w1 = select_weights(S[np.ix_(left, left)], np.zeros(len(left)), 0.0)
    w2 = select_weights(S[np.ix_(right, right)], np.zeros(len(right)), 0.0)
    diff = w1 @ values[left] - w2 @ values[right]
    var = w1 @ S[np.ix_(left, left)] @ w1 + w2 @ S[np.ix_(right, right)] @ w2 - 2 * w1 @ S[np.ix_(left, right)] @ w2
    return abs(diff) / np.sqrt(max(var, 1e-18))


def changepoint_groups(values: np.ndarray, S: np.ndarray, z_crit: float) -> list:
    """Binary segmentation of the window sequence into reward-homogeneous segments.

    For a segment, every split point is scored by the z-statistic between the
    minimum-variance means on either side; the segment is split at the best
    point if it exceeds z_crit, recursively. Comparing blocks rather than
    single windows gives far more power for the same drift (e.g. a 6 | 6 split
    has sd ~ sqrt(1/6 + 1/6) of one window's). Returns segments newest-first.
    """
    segments, stack = [], [list(range(len(values)))]
    while stack:
        seg = stack.pop()
        best_z, best_s = -np.inf, None
        for s in range(1, len(seg)):
            z = _block_diff_z(values, S, seg[:s], seg[s:])
            if z > best_z:
                best_z, best_s = z, s
        if best_s is not None and best_z > z_crit:
            stack += [seg[:best_s], seg[best_s:]]
        else:
            segments.append(seg)
    return sorted(segments, key=lambda g: -max(g))


def segment_select(comps: list, S: np.ndarray, z_group: float, z_anchor: float,
                   fallback: str = "anchor", grouping: str = "greedy", groups: list | None = None) -> np.ndarray:
    """Weights over `comps` (anchor last): the most recent homogeneous group
    that the anchor does not contradict, pooled with the anchor at minimum
    variance. If every group is contradicted, fall back to the anchor alone
    ("anchor") or to the least-contradicted group pooled with it ("least")."""
    h = len(comps) - 1
    alpha = np.zeros(h + 1)
    if h == 0:
        alpha[-1] = 1.0
        return alpha
    values = np.array([c.value for c in comps])
    best_z, best_group = np.inf, None
    if groups is None:
        grouper = changepoint_groups if grouping == "changepoint" else homogeneous_groups
        groups = grouper(values[:h], S[:h, :h], z_group)
    for group in groups:
        w = select_weights(S[np.ix_(group, group)], np.zeros(len(group)), 0.0)
        diff = w @ values[group] - values[h]
        var = w @ S[np.ix_(group, group)] @ w + S[h, h] - 2 * w @ S[group, h]
        z = abs(diff) / np.sqrt(max(var, 1e-18))
        if z <= z_anchor:
            best_group = group
            break
        if z < best_z:
            best_z, best_group = z, group
    else:
        if fallback == "anchor":
            alpha[-1] = 1.0
            return alpha
    idx = best_group + [h]
    alpha[idx] = select_weights(S[np.ix_(idx, idx)], np.zeros(len(idx)), 0.0)
    return alpha


def shrink_select(comps: list, S: np.ndarray, cfg, lam: float, groups: list | None = None) -> np.ndarray:
    """Soft version of segment_select: debiased-MSE weights with the bias of
    every component measured against the reference r = segment_select's
    (newest anchor-consistent segment + anchor) instead of the noisy anchor.

    d = V - (r . V),  Cov(d) = P S P^T with P = I - 1 r^T. Assumes the
    reference is unbiased (it passed the anchor veto); older segments whose
    discrepancy is small relative to its noise keep partial weight instead
    of an all-or-nothing split.
    """
    if cfg.shrink_ess > 0:
        # reference by evidence, not calendar: newest windows until cumulative ESS >= target
        idx, total = [len(comps) - 1], 0.0
        for k in range(len(comps) - 2, -1, -1):
            if total >= cfg.shrink_ess:
                break
            idx.append(k)
            total += comps[k].diagnostics.ess
        ref = np.zeros(len(comps))
        ref[idx] = select_weights(S[np.ix_(idx, idx)], np.zeros(len(idx)), 0.0)
    elif cfg.shrink_ref > 0:
        # reference: the most recent `shrink_ref` eligible windows + anchor
        idx = list(range(max(0, len(comps) - 1 - cfg.shrink_ref), len(comps)))
        ref = np.zeros(len(comps))
        ref[idx] = select_weights(S[np.ix_(idx, idx)], np.zeros(len(idx)), 0.0)
    else:
        ref = segment_select(comps, S, cfg.z_group, cfg.z_anchor, cfg.fallback, cfg.grouping, groups)
    values = np.array([c.value for c in comps])
    d = values - ref @ values
    P = np.eye(len(comps)) - np.outer(np.ones(len(comps)), ref)
    C = P @ S @ P.T
    return select_weights(S, np.zeros(len(comps)), 0.0, Q=mse_objective(S, d, C, lam))


# ------------------------------------------------------------------ O-DRIFT

@dataclass
class ODriftConfig:
    lam: float = 1.0
    selection: str = "segment"   # "segment" | "mse" (debiased mixture MSE) | "bound" (worst-case)
    z_group: float = 3.0         # homogeneity test threshold between windows
    z_anchor: float = 2.0        # anchor veto threshold for a homogeneous group
    fallback: str = "anchor"     # all groups vetoed -> "anchor" alone | "least" contradicted group + anchor
    grouping: str = "greedy"     # "greedy" window-vs-group | "changepoint" binary segmentation
    shrink_ref: int = 0          # "shrink": 0 = segment reference; m > 0 = newest m windows + anchor
    shrink_ess: float = 0.0      # "shrink": > 0 = newest windows until cumulative ESS >= this (overrides m)
    context_alpha: float = 0.0   # > 0: density-ratio transport only for windows whose context-shift test rejects
    balanced: bool = False       # pool each reward segment with balanced (mixture) propensities
    joint_segments: bool = False  # segment reward models: one joint fit (shared base + per-segment offsets)
    segment_trend: str = ""       # joint fit only: "" none | "segment" per-segment slope | "shared" one
                                  # within-segment slope common to all segments; evaluated at the target time
    model_anchor: bool = False    # joint fit only: anchor records join the newest segment at t = 1, so the
                                  # target-time prediction is interpolated, not extrapolated
    anchor_sn: bool = False      # self-normalized DR for the anchor component
    delta_variance: float = 0.025
    delta_bias: float = 0.025
    delta_diagnostics: float = 0.01
    min_ess: float = 50.0
    min_anchor: int = 50


@dataclass
class ODriftResult:
    status: str                  # "certified" | "not_certified"
    value: float
    se: float
    bias_allowance: float        # sum_k alpha_k Bbar_k
    half_width: float
    weights: dict                # component index -> alpha
    bounds: dict                 # component index -> Bbar
    reasons: dict                # window index -> reason codes for ineligible windows
    z_v: float = 1.96
    se_hetero: float | None = None  # DL-inflated SE (informative, not certified)

    def interval(self) -> tuple:
        return self.value - self.half_width, self.value + self.half_width

    def covers(self, truth: float) -> bool:
        lo, hi = self.interval()
        return lo <= truth <= hi


def fit_all_nuisances(history: list, anchor: LoggedData, X_target: np.ndarray, n_actions: int,
                      cfg: ODriftConfig, seed: int = 0, segments: list | None = None) -> tuple:
    """Policy-independent nuisances for every window and the anchor (or None).

    segments: reward-homogeneous window groups (e.g. from the reward monitor).
    When given, one cross-fitted reward model is fit per segment on the pooled
    segment data and shared by its windows -- valid because the windows share
    q -- instead of one model per window. Context ratios stay per window.
    """
    if segments is not None and cfg.joint_segments:
        hist = _joint_segment_nuisances(history, X_target, n_actions, cfg, seed, segments,
                                        anchor if cfg.model_anchor and len(anchor) else None)
    elif segments is None:
        hist = [fit_nuisances(d, X_target, n_actions, seed=seed + k, context_alpha=cfg.context_alpha)
                for k, d in enumerate(history)]
    else:
        hist = [None] * len(history)
        for j, seg in enumerate(segments):
            pooled = LoggedData.concat([history[k] for k in seg])
            q_src, q_tgt = crossfit_rewards(pooled, n_actions, seed=seed + 100 * j, extra_X=X_target)
            for k in seg:
                d, m = history[k], pooled.window == k
                shift = cfg.context_alpha <= 0 or context_shift_pvalue(d.X, X_target) < cfg.context_alpha
                w = context_ratio(d.X, X_target, seed=seed + k) if shift else np.ones(len(d))
                hist[k] = Nuisances(q_src[m], q_tgt, w, ratio_function(d.X, X_target) if shift else None)
    anc = (fit_nuisances(anchor, X_target, n_actions, same_distribution=True, seed=seed + 999)
           if len(anchor) >= cfg.min_anchor else None)
    return hist, anc


def _joint_segment_nuisances(history: list, X_target: np.ndarray, n_actions: int, cfg: ODriftConfig,
                             seed: int, segments: list, anchor: LoggedData | None = None) -> list:
    """One cross-fitted logistic model on all history: shared per-action linear base plus
    a per-segment linear offset (segments[0], the newest, is the baseline). Segment models
    then share the base's misspecification error, so the drift estimate q_ref - q_old used
    by drift-corrected reuse is the offset alone rather than a difference of two separately
    misspecified fits on different context distributions."""
    from sklearn.linear_model import LogisticRegression
    from .nuisance import _features
    seg_of = {k: j for j, seg in enumerate(segments) for k in seg}
    if anchor is not None:  # anchor (window index K) is fit as part of the newest segment
        seg_of[len(history)] = 0
    pooled = LoggedData.concat(history + ([anchor] if anchor is not None else []))
    sid = np.array([seg_of[k] for k in pooled.window])
    J, M = len(segments), len(X_target)

    t_all = pooled.window / len(history)  # target / anchor time = 1

    def feats(X, A, s_ids, t):
        base = _features(X, A, n_actions)
        cols = [base] + [base * (s_ids == j)[:, None] for j in range(1, J)]
        if cfg.segment_trend == "segment":  # within-segment drift linear in time (gradual drift)
            cols += [base * ((s_ids == j) * t)[:, None] for j in range(J)]
        elif cfg.segment_trend == "shared":  # one slope, identified within segments (offsets absorb levels)
            cols += [base * t[:, None]]
        elif cfg.segment_trend:
            raise ValueError(cfg.segment_trend)
        return np.hstack(cols)

    from scipy.special import expit
    from .nuisance import block_logits
    P = n_actions * (1 + X_target.shape[1])  # coefficients per feature block (same layout as _features)

    def logits(coef, X, s_ids, t):
        """All-action logits of the fitted joint model, block by block (no feature materialization)."""
        blk = [coef[i * P:(i + 1) * P] for i in range(len(coef) // P)]
        L = block_logits(X, blk[0], n_actions)
        for j in range(1, J):
            L = L + (s_ids == j)[:, None] * block_logits(X, blk[j], n_actions)
        if cfg.segment_trend == "segment":
            for j in range(J):
                L = L + ((s_ids == j) * t)[:, None] * block_logits(X, blk[J + j], n_actions)
        elif cfg.segment_trend == "shared":
            L = L + t[:, None] * block_logits(X, blk[J], n_actions)
        return L

    folds = np.random.default_rng(seed).permutation(len(pooled)) % 2
    q_src = np.empty((len(pooled), n_actions))
    q_tgt = np.zeros((J, M, n_actions))
    for f in range(2):
        tr = folds != f
        model = LogisticRegression(C=1.0, max_iter=3000, fit_intercept=False)
        model.fit(feats(pooled.X[tr], pooled.A[tr], sid[tr], t_all[tr]), pooled.R[tr])
        coef, te = model.coef_.ravel(), np.where(~tr)[0]
        q_src[te] = expit(logits(coef, pooled.X[te], sid[te], t_all[te]))
        for j in range(J):
            q_tgt[j] += expit(logits(coef, X_target, np.full(M, j), np.ones(M))) / 2
    hist = []
    for k, d in enumerate(history):
        shift = cfg.context_alpha <= 0 or context_shift_pvalue(d.X, X_target) < cfg.context_alpha
        w = context_ratio(d.X, X_target, seed=seed + k) if shift else np.ones(len(d))
        hist.append(Nuisances(q_src[pooled.window == k], q_tgt[seg_of[k]], w,
                              ratio_function(d.X, X_target) if shift else None))
    return hist


def build_components(history: list, anchor: LoggedData, X_target: np.ndarray, pi, n_actions: int,
                     cfg: ODriftConfig, seed: int = 0, nuisances: tuple | None = None,
                     reference_windows: list | None = None) -> tuple:
    """Transported DR components for every window plus the anchor (or None).

    reference_windows (drift-corrected reuse): windows of the newest reward-
    homogeneous segment, whose shared reward model supplies the direct-method
    term for EVERY window. Windows outside it keep only their own residual
    correction w rho (R - q_own), so their rewards enter through their own
    model's misspecification, not their (drifted) level:
        V_k^corr = mean_j m_ref(X*_j) + mean_i w rho (R_i - q_own(X_i, A_i)),
    with bias equal to the difference in misspecification between the two
    segment models on P_T -- zero when the drift stays within the model class.
    """
    if nuisances is None:
        nuisances = fit_all_nuisances(history, anchor, X_target, n_actions, cfg, seed)
    hist_nuis, anc_nuis = nuisances
    z_diag = norm.ppf(1 - cfg.delta_diagnostics / (2 * (len(history) + 1)))
    comps = [transported_dr(d, X_target, pi, n_actions, k, z_diag, cfg.min_ess, nuis=nu)
             for k, (d, nu) in enumerate(zip(history, hist_nuis))]
    if reference_windows:
        dm_ref = (pi.probs(X_target) * hist_nuis[reference_windows[0]].q_tgt).sum(axis=1)
        for c in comps:
            if c.index not in reference_windows:
                c.dm_scores = dm_ref
                c.value = float(dm_ref.mean() + c.corr_scores.mean())
    anchor_comp = None
    if anc_nuis is not None:
        anchor_comp = transported_dr(anchor, X_target, pi, n_actions, len(history), z_diag,
                                     cfg.min_ess, same_distribution=True, nuis=anc_nuis,
                                     self_normalize=cfg.anchor_sn)
    return comps, anchor_comp


def combine(comps: list, anchor_comp, cfg: ODriftConfig, lam: float | None = None,
            bound_comps: tuple | None = None, window_groups: list | None = None) -> ODriftResult:
    """Select weights and form the transfer-robust interval.

    `bound_comps` = (comps, anchor_comp) from an independent split, used for
    Bbar and alpha (sample-split variant); defaults to the same components
    (plug-in variant).
    """
    lam = cfg.lam if lam is None else lam
    reasons = {c.index: c.diagnostics.reasons for c in comps if not c.diagnostics.eligible}
    z_v = norm.ppf(1 - cfg.delta_variance / 2)
    if anchor_comp is None:
        # V_T is not identified from historical rewards alone (proposal §10, impossibility note)
        eligible = [c for c in comps if c.diagnostics.eligible] or comps
        S = covariance(eligible)
        alpha = select_weights(S, np.zeros(len(eligible)), 0.0)
        v = float(sum(a * c.value for a, c in zip(alpha, eligible)))
        return ODriftResult("not_certified", v, float(np.sqrt(alpha @ S @ alpha)), np.inf, np.inf,
                            {c.index: float(a) for a, c in zip(alpha, eligible)}, {}, reasons, z_v)

    sel_comps, sel_anchor = bound_comps if bound_comps is not None else (comps, anchor_comp)
    keep = [i for i, c in enumerate(sel_comps) if c.diagnostics.eligible]
    sel = [sel_comps[i] for i in keep] + [sel_anchor]
    S_sel = covariance(sel)
    z_k = norm.ppf(1 - cfg.delta_bias / (2 * len(keep))) if keep else 0.0
    B = bias_bounds(sel, sel_anchor, S_sel, z_k)
    groups = None
    if window_groups is not None:
        # map window indices (from the reward monitor) to positions among eligible components
        pos = {sel_comps[i].index: j for j, i in enumerate(keep)}
        groups = [g for g in ([pos[k] for k in grp if k in pos] for grp in window_groups) if g]
    if cfg.selection == "segment":
        alpha = segment_select(sel, S_sel, cfg.z_group, cfg.z_anchor, cfg.fallback, cfg.grouping, groups)
    elif cfg.selection == "shrink":
        alpha = shrink_select(sel, S_sel, cfg, lam, groups)
    elif cfg.selection == "mse":
        d_sel, C_sel = diff_moments(sel, S_sel)
        alpha = select_weights(S_sel, B, lam, Q=mse_objective(S_sel, d_sel, C_sel, lam))
    else:
        alpha = select_weights(S_sel, B, lam)

    est = [comps[i] for i in keep] + [anchor_comp]
    S_est = covariance(est)
    value = float(sum(a * c.value for a, c in zip(alpha, est)))
    se = float(np.sqrt(max(alpha @ S_est @ alpha, 0.0)))
    d_est, C_est = diff_moments(est, S_est)
    z_b = norm.ppf(1 - cfg.delta_bias / 2)
    allowance = float(abs(alpha @ d_est) + z_b * np.sqrt(max(alpha @ C_est @ alpha, 0.0)))
    return ODriftResult("certified", value, se, allowance, z_v * se + allowance,
                        {c.index: float(a) for a, c in zip(alpha, est)},
                        {c.index: float(b) for b, c in zip(B, est)}, reasons, z_v)


def split_half(data: LoggedData, rng: np.random.Generator) -> tuple:
    m = rng.permutation(len(data)) % 2 == 0
    return data.subset(m), data.subset(~m)


@dataclass
class SplitEpisode:
    """Halves of every window, the anchor and the target contexts, with fitted nuisances."""
    sel: tuple   # (history halves, anchor half, X_target half, nuisances)
    est: tuple


def make_split(history: list, anchor: LoggedData, X_target: np.ndarray, n_actions: int,
               cfg: ODriftConfig, seed: int = 0) -> SplitEpisode:
    rng = np.random.default_rng(seed + 7)
    halves = [split_half(d, rng) for d in history]
    a_sel, a_est = split_half(anchor, rng)
    xt = rng.permutation(len(X_target))
    parts = []
    for i, (a, x) in enumerate(((a_sel, X_target[xt[::2]]), (a_est, X_target[xt[1::2]]))):
        h = [hv[i] for hv in halves]
        parts.append((h, a, x, fit_all_nuisances(h, a, x, n_actions, cfg, seed + 50 * i)))
    return SplitEpisode(*parts)


def odrift_split(split: SplitEpisode, pi, n_actions: int, cfg: ODriftConfig) -> ODriftResult:
    """Bounds, eligibility and weights from the selection half; estimate,
    SE and bias allowance from the estimation half (Proposition 2 conditions)."""
    sel = build_components(*split.sel[:3], pi, n_actions, cfg, nuisances=split.sel[3])
    est = build_components(*split.est[:3], pi, n_actions, cfg, nuisances=split.est[3])
    if sel[1] is None or est[1] is None:
        return combine(est[0], None, cfg)
    for c_sel, c_est in zip(sel[0], est[0]):
        c_est.diagnostics.reasons = c_sel.diagnostics.reasons
    return combine(est[0], est[1], cfg, bound_comps=sel)


def odrift(history: list, anchor: LoggedData, X_target: np.ndarray, pi, n_actions: int,
           cfg: ODriftConfig = ODriftConfig(), split: bool = False, seed: int = 0,
           lams: list | None = None) -> dict:
    """Run O-DRIFT for each lambda in `lams` (components are computed once).

    split=False: plug-in -- bounds, weights and final estimate share data.
    split=True: bounds/weights from one half of every window, the anchor and
    the target contexts; final estimate and SE from the other half
    (the condition under which Proposition 2's coverage argument holds).
    """
    lams = [cfg.lam] if lams is None else lams
    if not split:
        comps, anc = build_components(history, anchor, X_target, pi, n_actions, cfg, seed)
        return {lam: combine(comps, anc, cfg, lam) for lam in lams}
    sp = make_split(history, anchor, X_target, n_actions, cfg, seed)
    return {lam: odrift_split(sp, pi, n_actions, replace(cfg, lam=lam)) for lam in lams}


def choose_reference(comps: list, anc: Component, cfg: ODriftConfig, segments: list) -> list | None:
    """Newest monitor segment the anchor does not contradict (least-contradicted
    fallback), judged on uncorrected components. None = anchor alone."""
    keep = [i for i, c in enumerate(comps) if c.diagnostics.eligible]
    sel = [comps[i] for i in keep] + [anc]
    pos = {comps[i].index: j for j, i in enumerate(keep)}
    groups = [g for g in ([pos[k] for k in grp if k in pos] for grp in segments) if g]
    alpha_ref = segment_select(sel, covariance(sel), cfg.z_group, cfg.z_anchor, cfg.fallback, groups=groups)
    ref_windows = [sel[j].index for j in np.flatnonzero(alpha_ref[:-1] > 0)]
    if not ref_windows:
        return None
    return next(seg for seg in segments if ref_windows[0] in seg)


POOL = dict(selection="segment", z_group=np.inf, z_anchor=np.inf)


def odrift_corrected(history: list, anchor: LoggedData, X_target: np.ndarray, pi, n_actions: int,
                     cfg: ODriftConfig, nuisances: tuple, segments: list, mode: str = "pool") -> ODriftResult:
    """Full O-DRIFT with drift-corrected reuse (plug-in: one data set for everything).

    1. Reference segment: `choose_reference` on uncorrected components.
    2. Every window outside the reference is drift-corrected (reference model's
       direct-method term + its own residual correction; see build_components).
    3. mode="pool": all eligible corrected windows + anchor at minimum variance;
       mode="shrink": debiased-MSE shrinkage toward the reference segment.
    """
    comps, anc = build_components(history, anchor, X_target, pi, n_actions, cfg, nuisances=nuisances)
    if anc is None:
        return combine(comps, None, cfg)
    ref_segment = choose_reference(comps, anc, cfg, segments)
    if ref_segment is None:  # anchor alone: nothing to correct toward
        return combine(comps, anc, cfg, window_groups=segments)
    comps_c, anc_c = build_components(history, anchor, X_target, pi, n_actions, cfg, nuisances=nuisances,
                                      reference_windows=ref_segment)
    if mode == "pool":
        res = combine(comps_c, anc_c, replace(cfg, **POOL))
        res.se_hetero = heterogeneity_se(res, comps_c, anc_c)
        return res
    return combine(comps_c, anc_c, replace(cfg, selection="shrink"), 1.0,
                   window_groups=[ref_segment] + [s for s in segments if s is not ref_segment])


def make_corrected_split(history: list, anchor: LoggedData, X_target: np.ndarray, n_actions: int,
                         cfg: ODriftConfig, monitor_alpha: float, seed: int = 0) -> tuple:
    """Split episode for the honest-interval variant: monitor segments are found on
    the selection half and used for the segment reward models of BOTH halves."""
    from .monitor import ResidualStats, reward_changepoints
    rng = np.random.default_rng(seed + 7)
    halves = [split_half(d, rng) for d in history]
    a_sel, a_est = split_half(anchor, rng)
    xt = rng.permutation(len(X_target))
    sel_hist, est_hist = [h[0] for h in halves], [h[1] for h in halves]
    segments = reward_changepoints(ResidualStats(sel_hist, n_actions), monitor_alpha)
    parts = []
    for i, (h, a, x) in enumerate(((sel_hist, a_sel, X_target[xt[::2]]), (est_hist, a_est, X_target[xt[1::2]]))):
        parts.append((h, a, x, fit_all_nuisances(h, a, x, n_actions, cfg, seed + 50 * i, segments=segments)))
    return parts[0], parts[1], segments


def odrift_corrected_split(split: tuple, pi, n_actions: int, cfg: ODriftConfig) -> ODriftResult:
    """Proposition-2 variant: segments, eligibility, reference and weights from the
    selection half; estimate, SE and bias allowance from the estimation half."""
    sel, est, segments = split
    comps_s, anc_s = build_components(*sel[:3], pi, n_actions, cfg, nuisances=sel[3])
    if anc_s is None:
        return combine(build_components(*est[:3], pi, n_actions, cfg, nuisances=est[3])[0], None, cfg)
    ref_segment = choose_reference(comps_s, anc_s, cfg, segments)
    ref = ref_segment or None
    sel_c = build_components(*sel[:3], pi, n_actions, cfg, nuisances=sel[3], reference_windows=ref)
    est_c = build_components(*est[:3], pi, n_actions, cfg, nuisances=est[3], reference_windows=ref)
    for c_s, c_e in zip(sel_c[0], est_c[0]):
        c_e.diagnostics.reasons = c_s.diagnostics.reasons
    pool_cfg = replace(cfg, **POOL) if ref_segment is not None else cfg
    return combine(est_c[0], est_c[1], pool_cfg, bound_comps=sel_c)


def heterogeneity_se(res: ODriftResult, comps: list, anc: Component) -> float:
    """SE of a pooled O-DRIFT estimate inflated by DerSimonian-Laird between-window
    heterogeneity tau^2 among the eligible historical components (anchor excluded).

    Not a certified interval: it absorbs residual disagreement between corrected
    windows (imperfect correction, within-segment drift) but cannot see bias that is
    common to all windows -- only current outcomes can (cf. the anchor-limited
    certified interval).
    """
    keep = [c for c in comps if c.diagnostics.eligible] + [anc]
    alpha = np.array([res.weights.get(c.index, 0.0) for c in keep])
    S = covariance(keep)
    h = len(keep) - 1
    if h < 2:
        return res.se
    v = np.array([c.value for c in keep[:h]])
    w = 1 / np.diag(S)[:h]
    m = (w * v).sum() / w.sum()
    Q = (w * (v - m) ** 2).sum()
    tau2 = max(0.0, (Q - (h - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    S = S.copy()
    S[:h, :h] += tau2 * np.eye(h)
    return float(np.sqrt(max(alpha @ S @ alpha, 0.0)))


def balanced_segment_components(history: list, anchor: LoggedData, X_target: np.ndarray, pi, n_actions: int,
                                cfg: ODriftConfig, nuisances: tuple, segments: list, logger_fns: list,
                                reference_segment: list | None, window_comps: list) -> tuple:
    """One component per reward-homogeneous segment, pooling its eligible windows with
    balanced weights (logger drift -> balancing; Agarwal et al. 2017, Kallus et al. 2021):

        W_i = pi(a_i|x_i) / sum_k (n_k/N) mu_k(a_i|x_i) / w_k(x_i),

    the exact importance weight from the target to the segment's pooled sampling
    distribution sum_k (n_k/N) p_k(x) mu_k(a|x), with p_k = p_T / w_k. logger_fns[k](X)
    returns window k's event-time logger probabilities (assumption A4). Segments other
    than the reference are drift-corrected: reference model's DM term + their own
    residual correction against their own segment model.
    """
    hist_nuis, anc_nuis = nuisances
    eligible = {c.index for c in window_comps if c.diagnostics.eligible}
    z_diag = norm.ppf(1 - cfg.delta_diagnostics / (2 * (len(history) + 1)))
    pi_t = pi.probs(X_target)
    ref = reference_segment if reference_segment is not None else segments[0]
    dm_ref = (pi_t * hist_nuis[ref[0]].q_tgt).sum(axis=1)
    comps = []
    for j, seg in enumerate(segments):
        ks = [k for k in seg if k in eligible]
        if not ks:
            continue
        data = LoggedData.concat([history[k] for k in ks])
        q_src = np.concatenate([hist_nuis[k].q_src for k in ks])
        N = len(data)
        denom = np.zeros(N)
        for k in ks:
            mu_k = logger_fns[k](data.X)[np.arange(N), data.A]
            w_k = hist_nuis[k].w_fn(data.X) if hist_nuis[k].w_fn is not None else 1.0
            denom += len(history[k]) / N * mu_k / w_k
        p_src = pi.probs(data.X)
        W = p_src[np.arange(N), data.A] / denom
        corr = W * (data.R - q_src[np.arange(N), data.A])
        dm = dm_ref if (reference_segment is not None and seg is not ref) else (pi_t * hist_nuis[ks[0]].q_tgt).sum(axis=1)
        diag = diagnose(data, p_src, n_actions, np.ones(N), z_diag, cfg.min_ess)
        diag.reasons = [r for r in diag.reasons if r != "propensity_inconsistent"]  # mixture data: checked per window
        diag.ess = effective_sample_size(W)
        comps.append(Component(1000 + j, float(dm.mean() + corr.mean()), dm, corr, diag,
                               members={k: len(history[k]) / N for k in ks}))
    anc = None
    if anc_nuis is not None:
        anc = next(c for c in [transported_dr(anchor, X_target, pi, n_actions, len(history), z_diag, cfg.min_ess,
                                              same_distribution=True, nuis=anc_nuis)])
    return comps, anc


def odrift_balanced(history: list, anchor: LoggedData, X_target: np.ndarray, pi, n_actions: int,
                    cfg: ODriftConfig, nuisances: tuple, segments: list, logger_fns: list) -> ODriftResult:
    """O-DRIFT with balanced within-segment pooling + drift-corrected reuse across segments."""
    comps, anc = build_components(history, anchor, X_target, pi, n_actions, cfg, nuisances=nuisances)
    if anc is None:
        return combine(comps, None, cfg)
    ref_segment = choose_reference(comps, anc, cfg, segments)
    seg_comps, anc_c = balanced_segment_components(history, anchor, X_target, pi, n_actions, cfg, nuisances,
                                                   segments, logger_fns, ref_segment, comps)
    if ref_segment is None or not seg_comps:
        return combine(comps, anc, cfg, window_groups=segments)
    res = combine(seg_comps, anc_c, replace(cfg, **POOL))
    res.se_hetero = heterogeneity_se(res, seg_comps, anc_c)
    # report weights per window (segment weight split by record share) + anchor (index K)
    window_w = {k: 0.0 for k in range(len(history) + 1)}
    for c in seg_comps:
        for k, share in c.members.items():
            window_w[k] += res.weights.get(c.index, 0.0) * share
    window_w[len(history)] = res.weights.get(len(history), 0.0)
    res.weights = window_w
    res.reasons.update({c.index: c.diagnostics.reasons for c in comps if not c.diagnostics.eligible})
    return res
