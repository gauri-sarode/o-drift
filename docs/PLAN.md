# O-DRIFT → PAKDD 2027 plan

Source proposal: `~/Downloads/Not_All_Drift_Is_Equal_Full_Research_Proposal.pdf`
(originally written for IEEE BigData 2026; that deadline passed).

## Venue facts (PAKDD 2027 Research Track)

- Submission: **Fri 20 Nov 2026, 23:59 AoE** via CMT (cmt3.research.microsoft.com/PAKDD2027)
- Notification 26 Feb 2027; camera-ready 19 Mar 2027; in-person, Wellington NZ, 29 Jun–2 Jul 2027
- Springer LNCS, **≤ 12 pages including references and appendices**; abstract ≤ 200 words
- **Double-blind**: no author identifiers, own prior work in third person, no Netflix/employer framing
- arXiv is OK only if the submission does not cite the arXiv version
- **Generative AI**: allowed only for grammatical correction and must be disclosed on the
  submission form. Paper prose is written by the authors; Claude is used for code,
  experiments, analysis and review — disclose accordingly.
- Track: Research Track (no central LLM/agentic component → not the LLM track)

## Scope changes from the BigData proposal

| Kept | Cut / deferred |
|---|---|
| RQ1 mechanism-aware selection vs. tuned temporal baselines | RQ4 incremental/distributed systems evaluation (§11, Fig. 6) |
| RQ2 coverage + abstention | Proposition 3 (oracle inequality) — stretch only |
| RQ3 diagnostic complementarity (ablations) | Streaming watermarks, Spark, 1M–100M scaling |
| Prop. 1 (transported DR), Prop. 2 (honest interval), impossibility example | OPFV / Kallus as must-haves (optional baselines) |
| Simulator S0–S4, S7 (S5, S6 if time) + Open Bandit semi-synthetic | Claims of natural longitudinal drift |

### Claim discipline (decided before experiments)

A worst-case bias bound calibrated on an anchor contains the anchor's own radius
(`r_anchor,cal`, computed on the calibration *half*), so the transfer-robust interval
can never be narrower than an anchor-only interval on the full anchor. Therefore:

- **Headline endpoints:** point-estimate RMSE and policy-selection regret.
- **Coverage framing:** "valid (near-nominal) where pooled/recency estimators undercover",
  compared explicitly against the anchor-only estimator — not "narrower intervals".
- Tighter certified intervals would require a structural drift assumption (e.g. groupwise
  drift); only pursue if time allows, and state the assumption.

## Timeline (29 days)

| Dates | Milestone | Gate |
|---|---|---|
| Oct 22–29 | Simulator w/ MC ground truth; IPS/SNIPS/DM/DR; cross-fit reward model; S0/S2 | Pooled estimators unbiased + ~95% coverage in S0 and S2 |
| Oct 30–Nov 5 | Context density ratio, transported DR, anchor bounds B̄_k, α-selection, abstention; S1/S3/S4/S7 | Truth V_{T←k} recovered; bound covers true b_k at ≥ 1−δ |
| Nov 6–12 | Baselines: recent-k DR, decay DR (tuned on dev), context-DR, Liu 2023, Zhan 2021; OBD semi-synthetic; ablations, ≥100 seeds; **one** run on frozen test regimes | Protocol freeze honored |
| Nov 13–20 | Figures/tables, propositions + proofs, author-written LNCS draft, citation check, anonymization, submit | Submitted by Nov 19 (1 day buffer) |

## 12-page LNCS budget (incl. refs)

| Section | Pages |
|---|---|
| Abstract + Intro + contributions | 1.5 |
| Related work | 1.0 |
| Setting, assumptions, impossibility | 1.5 |
| Method (transported DR, bounds, selection, abstention, propositions) | 2.5 |
| Experimental setup | 1.0 |
| Results + ablations | 2.5 |
| Limitations + conclusion | 0.5 |
| References | 1.5 |

## Figures/tables (reduced set)

1. Fig. 1 — mechanism illustration: context-shifted old window retained vs reward-shifted recent window rejected (doubles as Fig. 5 of proposal)
2. Fig. 2 — RMSE vs drift intensity per regime, all methods
3. Fig. 3 — coverage vs abstention / anchor size
4. Table 1 — related-work matrix; Table 2 — main results (test regimes); Table 3 — ablations

## Method decision log (dev split only; test untouched)

1. **Worst-case bounds in the selector** (proposal §4.5, `selection="bound"`): every
   B̄_k contains the anchor's own noise, so the objective collapses onto the anchor
   (S3a dev smoke: RMSE 0.033 vs pooled 0.009, oracle 0.006). Kept as ablation.
2. **Debiased mixture-MSE selector** (`selection="mse"`): E[(β·d)²] − Var(β·d) is
   unbiased for the squared mixture bias, but its noise is anchor-scale, so in S0 it
   still pays heavily (0.015 vs pooled 0.0046). Kept as ablation.
3. **Reward-homogeneity grouping + anchor veto** (`selection="segment"`, default):
   after context transport and propensity correction, transported window values can
   only disagree through reward drift, and window-vs-window comparisons have
   per-window precision (sd ≈ 0.015–0.03; group means tighter) instead of anchor-scale
   (≈ 0.035–0.07). [Corrected 2026-10-22: an earlier note said 0.005 — that was the
   pooled sd, not per-window.] Take the
   most recent homogeneous group the anchor does not contradict. Dev smoke: S0 0.0042
   (pooled 0.0046), S3a 0.0058 (oracle 0.0053).
   → This is the paper's core mechanism claim: *context and policy drift are removed by
   transport/propensities; what remains between windows is reward drift, which is
   detectable without the anchor; the anchor only has to arbitrate between groups.*
4. Interval: mixture-level allowance |β·d| + z·sd(β·d) on the held-out half (split
   variant) — valid by Prop. 2 but never narrower than anchor-only (as predicted).

## Related-work items to add (novelty risk)

- **MAGIC** (Thomas & Brunskill, ICML 2016): blends estimators by minimizing
  variance + bias², bias estimated as distance to a reference estimator's CI —
  close to our "bound" ablation. Must cite and contrast.
- Liu et al. 2023 Reg-DR: old data only through r_hat; correction on current data.
- Zhan et al. 2021: variance-stabilizing weights, stationary reward.
- Heterogeneity testing (Cochran's Q / meta-analysis) and change-point detection —
  our grouping step is an instance; cite as such rather than claim novelty for it.
- Pre-test estimators (Bancroft 1944; Judge & Bock): the anchor veto is a pre-test;
  expect the known mid-intensity risk bump — report it.
5. **Anchor false-veto failure (dev S1 diagnostic)**: grouping is correct (one group),
   but heavy-tailed anchor estimates for noisy candidate policies trigger the veto
   (e.g. anchor 0.600 vs group 0.465, z≈2.7) and the fallback is anchor-alone → large
   error. Candidate fixes to compare on dev: higher z_anchor (already in grid), fallback
   to least-contradicted group instead of anchor-only, robust (self-normalized) anchor.
6. **Dev results with all baselines (n=200/cell, 4 candidate policies)**: tuned O-DRIFT
   (old anchor-only fallback) was 2–3.4x worse than the best tuned baseline in every cell.
   Least-contradicted fallback cuts that 2–3x (anchor-only outputs were only ~2% of runs
   but carried huge heavy-tailed errors). z_group sweep shows a hard trade-off, no
   setting dominates: z_g=3 matches transport DR in S0/S1/S2 but loses to recent-6 under
   reward drift (S3a 12.3 vs 7.8); z_g=2 helps S3 but false-splits in S1 (9.2 vs 6.6).
   Root cause: at intensity 0.5, per-window transfer bias (~0.005–0.024) is below
   per-window noise (~0.02), so reward drift is not reliably detectable.
   Oracle headroom over tuned recency is only ~10% (S3a/S3b), ~0% (S4).
   Worst-case ratio to per-cell best (dev, optimistic): O-DRIFT z_g=2/z_a=3 1.54,
   transport DR 1.71, recent-6 2.01, decay-0.8 2.16, pooled 3.25.
   Self-normalized anchor: no effect.
