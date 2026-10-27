# Notes: why the residual monitor is mechanism-aware (derivation for the author)

Working notes, not paper text. They state what is true and what the experiments check, so the
author can write the proposition and proof in their own words.

## Setup

Window k has context law P_k(x), logger mu_k(a|x), conditional reward q_k(x,a) = E[R | x, a, k].
The monitor fits one pooled per-action model q~ on all history and, for a block of windows B and
action a, uses the mean residual

    m_B(a) = mean of (R_i - q~(X_i, a)) over records i in B with A_i = a.

Two blocks L, R are compared with sum_a (m_L(a) - m_R(a))^2 / v(a), v(a) the binomial
variance estimate (chi-square with df = number of actions with enough records).

## Claim 1: invariance to context and logger drift (correct specification)

Assume (i) no reward drift within L u R: q_k = q for every k in L u R; (ii) q~ -> q (the
model class contains q and the pooled fit is consistent; with cross-fitting, or as n grows).
Then for every window k in L u R and every action a:

    E[R - q(X, a) | A = a, window k] = E_{P_k(x | a)} [ q_k(x, a) - q(x, a) ] = 0,

whatever P_k(x) and mu_k(a|x) are, because conditioning on (x, a) removes both: P_k(x | a) is
proportional to P_k(x) mu_k(a|x) and only reweights a quantity that is identically zero.
So E[m_L(a) - m_R(a)] = 0 for all a, and the statistic is asymptotically chi-square: the test
has size alpha under ANY context or logger drift. Only reward drift (q_k != q on the support)
moves m_B(a), so only reward drift creates power.

## Claim 2: mechanism-blind statistics are not invariant

Overall mean reward (COPE's similarity |rbar_k - rbar_T|, our "mean" ablation):

    E[R | window k] = sum_a  integral  P_k(x) mu_k(a|x) q(x, a) dx,

which changes with P_k and mu_k even when q is fixed. Consequences:
- false drift: context or logger drift alone gives a non-zero mean difference, so the test
  rejects with probability -> 1 as n grows (S1, S2);
- missed drift: reward drift can be cancelled by a context or logger change in the opposite
  direction (same expected reward, different q), so the test can have no power.

Per-action raw mean (our "action" ablation): E[R | a, k] = E_{P_k(x|a)}[q(x, a)]. This removes
the action mix, but P_k(x | a) still depends on P_k(x) and on mu_k(a|x) as a function of x, so
context drift and context-dependent logger drift still shift it.

## Honest caveat (misspecification)

If q is outside the model class, the residual has a non-zero conditional mean
E_{P_k(x|a)}[q - q~] that depends on the window's context law, so context drift can trigger the
monitor (false splits). This is the S4 misspec-3 failure (10/32 false splits, PLAN 28). A richer
basis (rff32) shrinks q - q~ and therefore the false-split rate. State this as the limitation of
Claim 1 instead of hiding it.

## Evidence already in hand / pending

- tests/test_v3.py: S2 (logger switches only) -> monitor 1 segment, mean statistic > 1 segment.
- tests/test_method.py: S1 (context drift only) -> monitor 1 segment; S3a -> correct split.
- dev3 grouping ablation (ablate_group[action|mean, alpha]): same O-DRIFT pipeline, only the
  statistic changes; expected to lose in S1/S2 (needless splits shrink the reference) and to
  gain nothing in S3/S4. Report false-split rates per regime alongside RMSE.
