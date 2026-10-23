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
7. **Round 2 dev method work (2026-10-23)**: (a) change-point (binary segmentation)
   grouping: small gain over greedy (S3a 12.3 -> 10.9). (b) model-based (DM) detection:
   dropped — DM window values are as noisy as DR (Bernoulli noise x weights dominates;
   SEs verified calibrated: empirical 0.0165 vs est. 0.0156). (c) Shrinkage w.r.t. the
   detected segment: no effect (if no split is found the reference already contains the
   drifted windows). (d) **Shrinkage w.r.t. the newest m windows + anchor** (debiased
   MSE): best so far. m=6 wins the tuning rule (mean log RMSE -4.649):
   S0 7.7, S1 8.1, S2 8.2, S3a 7.9, S3b 10.4, S4 18.4 (x1e-3).
   Worst-case ratio to per-cell best: 1.43 (transport 1.71, recent-6 2.01, decay 2.16,
   pooled 3.25); mean ratio 1.24 (transport 1.29, decay 1.38, recent-6 1.45).
   Still not best in any single cell. Caveat: many O-DRIFT variants were explored on
   dev, so its dev robustness is optimistic; the test split is the honest check.
8. **S4 diagnosis + evidence-based reference (2026-10-23)**: in S4 the oldest windows
   (logger A) are the most precise (sd ~0.02, ESS ~500) and also the reward-shifted ones
   (b 0.013-0.034); recent windows are 2-3x noisier (ESS 45-150). Minimum-variance
   weighting concentrates ~70% weight on the precise biased windows (transport DR 19.8),
   whereas pooled DR's per-record weighting dilutes the bias (13.6). A cumulative-ESS
   reference (targets 1500/3000/4500) did not help (S4 18.7-19.9 vs shrink-6 18.4): the
   reference grows into the old windows and inherits the same problem. Same information
   limit as S3a — b below reference noise. shrink-6 remains the tuning-rule winner.
9. **Reward-drift monitor (2026-10-23) — the key change.** Detect reward drift on the
   conditional reward function, not the scalar policy value: pooled per-action
   quadratic logistic model on all history; per-(window, action) residual means;
   binary segmentation with a chi-square test over actions (Bonferroni over split
   points). Detection check (20 episodes/regime): 0% false splits in S0/S1/S2 (context
   and logger shift do not fool it), 100% exact split in S3a at intensity 0.5 and 0.25.
   The scalar value test had z ~ 1-2 for the same drift; the per-action test pools
   evidence across actions whose shifts partially cancel in the policy value.
   Full dev (monitor α=0.001 + shrinkage, z_a=3): S0 5.8, S1 6.6, S2 6.5, S3a 8.0,
   S3b 10.9, S4 17.8 → ratio to per-cell best 1.08/1.00/1.07/1.03/1.09/1.33,
   worst 1.33, mean 1.10 (best baseline: transport 1.29 mean / 1.71 worst).
   This is the proposal's "mechanism-specific observability" made concrete: reward
   drift is detected from per-action conversion residuals, then the anchor only
   arbitrates between already-detected segments.
10. Context-shift gate: LR test of a source-vs-target logistic classifier; transport
   only when it rejects (S0/S2 false-reject ~1%; S1 shifts >= ~0.06 sd detected).
   Motivation: always-on density ratios cost ~8% RMSE in S0/S2.
11. **Per-segment reward models (2026-10-23)**: fit one cross-fitted reward model per
   reward-monitor segment, shared by its windows (valid: the windows share q by the
   monitor's test). Removes most of the residual gap to pooled DR in S0/S2 that per-window
   models (5k rows each) caused. Context gate: negligible effect (kept at α=0.01, harmless).
   Best variant (monitor α=0.001, segment models, shrinkage vs newest segment, z_a=3,
   least fallback, ctx gate 0.01) on full dev:
   S0 5.62 | S1 6.42 | S2 6.16 | S3a 7.71 | S3b 10.57 | S4 17.54  (x1e-3)
   ratio to per-cell best: 1.04 | 0.98 | 1.02 | 0.99 | 1.06 | 1.31; worst 1.31, mean 1.07.
   Best or within 6% of the best tuned baseline in 5/6 cells — each baseline has a
   >=1.48x failure somewhere. S4 remains the weak cell (precise-but-biased old data).
   Still unlikely to meet the frozen per-cell 0.9x bar; the supportable claim is
   uniform near-optimality across drift mechanisms.
12. **Drift-corrected reuse (2026-10-23) — fixes S4, first variant to beat the best
   baselines.** Reference = newest monitor segment the anchor does not contradict;
   windows outside it use the reference segment's model for the DM term and keep their
   own residual correction (bias = misspecification difference between segment models;
   zero when drift stays within the model class — state this assumption in the paper).
   Then pool all eligible corrected windows + anchor at minimum variance.
   Full dev (z_a=3, monitor α=0.001, segment models, ctx gate 0.01):
   S0 5.62 | S1 6.42 | S2 6.16 | S3a 6.79 | S3b 9.59 | S4 11.72  (x1e-3)
   ratio to per-cell best: 1.04 | 0.98 | 1.02 | 0.87 | 0.96 | 0.88; worst 1.04, mean 0.96.
   On dev this meets the frozen go/no-go shape (S3a, S4 <= 0.90; S0-S2 <= 1.10) —
   paired CIs still to be computed in the main evaluation. Shrink mode is dominated.
   **Method work stops here** (deadline risk; dev-overfitting risk).
13. Final O-DRIFT hyperparameter sweep (dev): flat surface. Monitor α 1e-4/1e-3 and
   z_anchor 3-5 all give ratios ~1.04/0.98/1.02/0.85-0.87/0.93-0.96/0.86-0.88 (mean 0.95);
   α=0.01 slightly worse in S4 (0.96). No knife-edge tuning. Formal selection by the
   protocol rule happens in the full 03_evaluate dev run.
14. **Intervals (2026-10-23)**. Split-half certified bound is vacuous (half-width ~0.21,
   2.5x anchor-only). Prototype on dev (240 policy-episodes/cell):
   - certified: anchor DR with a history-trained model (= Liu et al. regression-assisted
     DR) -> coverage 0.92-0.93, half-width 0.076 (anchor-limited floor; cite Liu, not ours).
   - informative: O-DRIFT SE inflated by DerSimonian-Laird between-window tau^2 ->
     coverage 0.93-0.95 without reward drift (hw ~0.012), 0.91 / 0.78 / 0.85 in S3a / S3b / S4
     (pooled DR 0.56 / 0.52 / 0.86). Residual S3b gap = bias common to the reference
     segment, visible only to the anchor.
   Reporting: point estimates headline; both interval types reported; split-half bound
   relabelled as an ablation. The FROZEN go/no-go coverage criterion (split coverage
   >= 0.93) is kept unchanged and reported as-is, with the explicit note that it passes
   only because the split bound is vacuous.
15. **Strong-baseline suite + balanced within-segment pooling (2026-10-23)**. Added SLOPE,
   MAGIC, anchor-validated selection, transport+recent/decay, multi-logger balanced DR,
   monitor-informed selection, oracle-tuned recent/decay. Balanced DR beat O-DRIFT in S2
   (ratio 1.20) -> O-DRIFT now pools windows within a monitor segment with mixture
   propensities (per-window version kept as `ablate_nobalance`). Full dev run v4
   (200 policy-episodes/cell, tuned odrift [1e-4, 5.0]), O-DRIFT vs best tuned baseline:
   S0 1.03 (CI incl. 0), S1 0.94*, S2 1.02 vs balanced_dr (CI incl. 0), S3a 0.83*,
   S3b 0.92*, S4 0.85* (*paired CI excludes 0). Beats oracle-tuned recent/decay in every
   drifted cell. Hetero interval coverage 0.95/0.96/0.95/0.92/0.74/0.86. Dev go/no-go:
   >=2 reward-drift cells at <=0.9x (S3a, S4) and S0-S2 <=1.1x -> passes on dev.
   S3b is 0.92 (not <=0.9). Results: results/03_dev.json, log results/logs/03_dev_v4.log.
16. **Pre-freeze review (2026-10-23)**. Per-env/per-policy breakdown of dev v4 showed S4's
   0.85 hid a loss in env 2 (1.39 vs MinVar) and losses for near-optimal policies in S4/S3b.
   Diagnosis (scratch diag_s4): the drift-corrected OLD segment carries -20..-30e-3 bias in
   env 2 = error in the estimated drift q_ref - q_old, because two separately fit, misspecified
   models are fit on different context distributions (old third vs the context-shifted middle
   third) and their misfits do not cancel. Candidate fix: one joint model (shared per-action
   linear base + per-segment linear offsets). Dev prototype (100 eps/cell, scripts/06):
   S4 11.20 -> 10.16 (env 2 12.8 -> 11.6), S3a +0.6%, S3b +3%, S0-S2 identical. Not
   hand-adopted: added as odrift_grid.joint_segments [false, true]; the frozen tuning rule
   decides, the other is an ablation.
   Additions in the same pass, all registered before any run of them:
   - baselines: forecast (Chandak et al. 2020, GLS polynomial in window time), time-aware DR
     (reward model x polynomial time), transport+balanced+decay composite. Smoke: composite
     within 4% of O-DRIFT in S4, time-aware DR = recent-6 in S3b; forecast weak (variance).
   - S8 misspecified (out-of-class) reward drift: test split [0.5, 1.0]; dev run diagnostic
     only, never tuned on; reported as robustness, not a go/no-go cell. Existing env draws
     verified bit-identical (new param drawn last).
   - Bug fix: balanced O-DRIFT reported weights keyed by segment, so per-window weights were
     all 0 and the S7 go/no-go check passed trivially. Now split by record share; test added.
   - Pre-registered: per-env/per-policy breakdowns, bias/sd split, search-budget counts.
17. **Dev v5 (2026-10-23, full suite incl. forecast/time-aware/composite)**. Tuning picked
   joint segment models: odrift [1e-4, 5.0, joint]. Ratio vs best tuned baseline per cell:
   S0 1.03, S1 0.94*, S2 1.02, S3a 0.83*, S3b **1.12 (loss, vs time-aware DR; CI excludes 0)**,
   S4 0.82* (vs new composite tbdecay[0.9]; env 2 fixed 1.39 -> 0.84). Dev go/no-go passes
   (S3a, S4). Mean log RMSE across cells: O-DRIFT -4.97, composite -4.79, tdecay -4.69,
   time-aware DR -4.53 -> no single baseline is competitive everywhere; time-aware DR wins
   only where drift is exactly linear in time (its own model class) and is 2x worse in S4,
   coverage 0.69-0.78. Report S3b as a stated loss. Results: results/03_dev.json (v5).
