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
18. **S3b fix attempt: within-segment time trend (2026-10-24) — rejected**. Joint segment
   model + linear time term evaluated at the target time, per-segment slope or one shared
   within-segment slope (scripts/06, full dev seeds, results/06_trend.json). RMSE x1e3
   joint / per-seg trend / shared trend: S0 5.53/8.83/8.83, S1 6.18/10.01/10.01,
   S2 5.23/9.50/9.50, S3a 6.46/9.90/9.15, S3b 9.44/10.23/9.99, S4 10.26/18.77/16.69,
   S8 6.70/10.25/9.22. The trend removes S3b's bias (+3.9 -> +0.5/+1.2) but extrapolation
   variance from history-only fits outweighs it in every cell, S3b included, so a gated
   (pre-tested) trend cannot beat time-aware DR (8.44) either. Not added to the grid; option
   kept in code as a documented negative result. Likely reason time-aware DR wins S3b: its
   time model is fit with the anchor at t = 1 (interpolation, not extrapolation) and its
   global linear-in-time model is exactly S3b's drift. S3b reported as a stated loss.
19. **Anchor in the joint reward model + S8b diagnostic (2026-10-24)**. results/06_anchor_trend.json,
   full dev seeds. RMSE x1e3 joint / +anchor / +anchor+shared trend / +anchor+segment trend:
   S0 5.53/5.49/8.57/8.57, S1 6.18/6.18/9.89/9.89, S2 5.23/5.24/9.38/9.38,
   S3a 6.46/6.41/8.88/9.66, S3b 9.44/9.23/9.74/9.84, S4 10.26/10.40/15.94/17.54,
   S8 6.70/6.65/8.96/9.87; mean log (dev) -4.970/-4.974/-4.590/-4.558.
   Anchoring the trend at t = 1 does NOT remove the trend's variance cost: S0 goes 5.5 -> 8.6,
   the same as time-aware DR (8.59). The cost is endpoint-of-regression variance, inherent to
   any time-slope model; it only pays off when drift is exactly linear in time (S3b). So S3b
   cannot be won without losing ~55% in every other cell -> stated loss, with this
   bias-variance explanation (strong paper point). Anchor-in-model without trend: negligible
   gain (-0.004) and weakens the anchor's independence for reference selection -> not adopted.
   S8b@0.5 dev diagnostic (results/03_dev_S8b_diagnostic.json): O-DRIFT 7.07 vs best
   (tbdecay) 6.83, ratio 1.04, CI [-0.31, +0.80]e-3; monitor splits 177/200; drift is weak
   at value level (pooled bias +0.3) -> inconclusive as a stress test; test S8b@1.0 decides.
20. **METHOD FREEZE (2026-10-24, user confirmed)**. Final: O-DRIFT [monitor alpha 1e-4,
   z_anchor 5.0, joint segment models], balanced within-segment pooling, drift-corrected
   reuse, hetero interval; all baselines at results/tuned.json. Test split (15 cells incl.
   S5/S6/S7/S8/S8b) + no-anchor run executed once from this commit. Pre-launch smoke
   checks on DEV seeds only (S5@0.5, S6@0.0, S8b@1.0 x 4 episodes, no-anchor S0) confirmed
   the pipeline runs; their outputs were deleted unanalysed (S8b@1.0 is a test intensity
   but those were dev-seed episodes; no test data touched).
21. **TEST RESULTS (2026-10-24, one shot, frozen config; results/03_test.json,
   03_test_anchor0.json, 04_test_summary.json, logs/04_test.log)**. Go/no-go: 5/5 PASS
   (wins <=0.9x with CI<0 in S3a@1.0 0.89, S4@0.25 0.82, S4@1.0 0.75, S5@1.0 0.85 -- S5 is
   the held-out drift type; S0/S1/S2 <= 1.1x; split coverage (vacuous bound, as pre-noted);
   S7 deficient weight 1.000; no-anchor never certified 1.000).
   Other cells, O-DRIFT / best tuned baseline: S1@0.5 0.96*, S1@2.0 1.01, S0 1.04 (CI>0),
   S2 1.07 (CI>0), S3a@0.25 0.91*, S5@0.5 1.04, S6 1.02, S8@0.5 1.00.
   Losses: S3b@0.25 1.35, S3b@1.0 2.31 (vs time-aware DR; O-DRIFT 22.6 still beats every
   other baseline, best non-timedr = recent-6 34.8; bias -13.2 at test intensity, sign
   opposite to dev); S7 1.44 vs MinVar (price of excluding support-deficient windows);
   S8@1.0 1.10, S8b@1.0 1.13 vs recent-6 (misspecified drift at high intensity: mild loss,
   no breakdown; bias ~0, variance cost of correction).
   Informative (hetero) interval: 0.94-0.96 without reward drift, 0.88-0.90 S3a/S4@0.25/S8,
   poor in S3b (0.65/0.36), S4@1.0 (0.74), S5 (0.33/0.15) -> must be reported as a
   limitation; certified Liu interval 0.91-0.92 everywhere (hw ~0.08).
   Fix after test: 04_analyze JSON bool serialisation only (no re-tuning, no method change).
22. **Anchor-consistency check for the model-based interval (2026-10-24, dev only) — rejected**.
   Rule: z = |V_odrift - V_ref| / SE(diff) (anchor's weight in O-DRIFT accounted for);
   z > c -> report certified Liu interval instead of the hetero interval. Ref = anchor DR or
   Liu; c in {1, 1.5, 2, 2.5}. Fallback rate is the same in every dev cell (~0.30/0.13/0.07/0.03
   = the null rate): zero power. Anchor-difference SE ~41e-3 vs biases to detect 4-25e-3
   (needs ~115e-3 for 80% power). Coverage gets WORSE in every cell (e.g. S0 0.95 -> 0.89-0.93,
   S3b 0.73 -> 0.69-0.76): fallback fires when the anchor is unlucky, and the certified interval
   is then centered near that unlucky anchor (selection effect). Not run on test. Paper:
   information-limit paragraph -- with n_anchor = 1000 the anchor cannot validate the history
   at the bias scale that matters; detecting a 4e-3 bias needs ~10^5-10^6 anchor records.
23. **Open Bandit semi-synthetic setup (2026-10-24; design pre-registered in protocol splits.real
   before any OBD result)**. scripts/07_prepare_obd.py -> data/obd_men.npz (gitignored; OBD is
   CC BY 4.0, cite Saito et al.); src/odrift/obd.py; 03_evaluate --split obd (frozen tuned.json,
   no tuning on OBD). Two pre-result amendments, both recorded in the protocol:
   (a) base reward: raw x^2 of heavy-tailed one-hot PCs overfit (logit sd ~6); now standardized
       x^2 + CV-chosen L2 (C=1) -> logit sd 0.60, per-user CTR 1-21%, nonlinear share 0.40.
   (b) candidate policies [0, .1, .2, .3]: simulator noise levels at 34 actions give ESS/n
       0.007-0.02 (no overlap; a 4-episode smoke had every method off by more than the value);
       chosen by an overlap-only rule (noisiest ESS/n 0.07 = simulator's 0.068).
   Engineering (behaviour-preserving, verified max diff 3e-16 vs old code, tests pass):
   per-action logistic predictions computed from coefficients instead of materializing the
   (n*A, A(1+d)) feature matrix -> OBD episode 374 s -> 88 s.
24. **Anchor-size sweep (2026-10-26; pre-registered anchor_sweep_cells, frozen config, test seeds;
   results/03_test_anchor{100,10000}.json, summary results/05_anchor_sweep.json)**. History is
   12 x 5000. Ratios O-DRIFT / best non-oracle baseline at n_anchor 100 / 1000 / 10000 (paired
   bootstrap CI in the summary file): S0 1.05/1.04/1.05, S1@2 0.99/1.01/1.03, S2 1.07/1.07/1.06,
   S3a@1 1.15 (CI 0.86-1.60)/0.89/0.93, S3b@1 2.26/2.31/2.48 (vs timedr), S4@1 0.62/0.75/1.17,
   S5@1 0.77/0.85/2.40 (vs anchor-only DR), S8@1 1.21/1.10/1.14.
   - Certified (Liu) interval: coverage 0.78 at n=100 (normal approximation fails; hw 0.20),
     0.92-0.93 at 1000 (hw 0.08) and 10000 (hw 0.027). Anchor-only DR covers 0.92-0.94 at all sizes.
   - O-DRIFT RMSE is nearly anchor-independent: the anchor only picks/validates the reference.
   - Post-hoc finding (label as such): O-DRIFT does not exploit a large anchor. At n=10000 the
     anchor-only DR (RMSE 14e-3) beats it in S5 (33.7e-3) and recent-window transport beats it in
     S4. Its residual bias (~20-35e-3) is below z_anchor=5 x the anchor SE, so the fixed
     threshold never rejects. Scope statement for the paper: O-DRIFT targets the small-anchor
     regime (anchor << history); with a large anchor, check the anchor-only estimator, or
     scale the threshold with n_anchor (future work; not tuned here).
25. **Open Bandit results (2026-10-26; frozen config, pre-registered cells, run once;
   results/03_obd.json)**. RMSE x1e-3, O-DRIFT vs the best non-oracle baseline:
   S0 3.1 vs 3.1 (1.01), S2 2.9 vs 2.9 (0.98, rank 1), S3a@.5 8.4 vs recent[6] 4.4 (1.91),
   S3a@1 11.5 vs 4.4 (2.61), S4@.5 17.5 vs timedr 8.8 (1.98), S4@1 47.9 vs recent[8] 10.5 (4.55),
   S5@1 68.0 vs Liu 20.2 (3.37). Under reward drift O-DRIFT errors are mostly bias (+6 to +60e-3),
   and its interval coverage collapses in S4@1 (0.55) and S5 (0.01). The no-balancing ablation
   is just as biased, so balancing is not the cause; the drift-corrected reuse is.
   Post-hoc diagnosis (label as such): the simulator's base reward is about 98% linear (the
   nonlinear share of logit variance is 0.02 at misspec=0.5), so the per-action logistic model
   is close to correct there. The OBD base fit to real clicks has a nonlinear share of 0.41,
   and its drift (g sd 1.84) is 3x the base logit sd (0.61). The correction assumes the
   reward-model class captures base + drift; on realistic rewards it does not, and the reused
   old windows carry bias that the 1000-row anchor (anchor-only DR RMSE 35.7e-3) cannot
   detect. In-class/low-misspecification simulator results therefore overstate O-DRIFT.
26. **v2 pre-registration (2026-10-27, user approved before any v2 code or run; protocol `v2`,
   splits dev2/test2/real2)**. Motivated by item 25, so v2 is a post-hoc revision and must be
   reported as one. One change: reference-validated reuse. Corrected segment components are
   weighted by debiased MSE, with bias measured against the reference (reference segment + anchor),
   not min-variance pooled. Tuned: reuse_lam in {0 (= v1), 0.5, 1, 2, 4}. Fixed at v1's tuned values:
   monitor 1e-4, z_anchor 5, joint. New simulator setting misspec 3.0, matched to OBD's nonlinear
   share by rule. dev2 = v1 dev + 3 misspec cells, and every family is re-tuned there (tuned_v2.json).
   One-shot tests on fresh seeds: test2 (env 500-503) and real2 (OBD env 400-403).
   Criteria: v1's five go/no-go on test2 at misspec 0.5, plus robustness on the reward-drift cells
   (misspec 3 + real2): no ratio > 1.5 and ratio <= 1.0 in at least half.
   Amendment (a), before any dev2 run: the shrink_select form (unbiased bias^2 = d^2 - Var d) moved
   weight toward old segments as lam grew, because a negative bias^2 estimate cancels the segment's
   own variance (seen in 3 dev episodes). Replaced by the positive part, with reference and anchor
   treated as unbiased: `reference_validated_weights`. Unit tests check that a constructed biased
   segment is dropped and that no detectable bias gives the v1 weights. The v1 dev path reproduces
   the stored dev results to 2e-13.
27. **dev2 results (2026-10-27; results/03_dev2.json, logs/04_dev2.log, tuned_v2.json)**. Tuning rule
   picks reuse_lam = 0.5 (mean log RMSE -4.854 vs v1 -4.828; flat: 1.0 -4.845, 2.0 -4.835, 4.0 -4.826).
   Baselines re-tuned on dev2: recent 12, decay 0.9, liu 6, slope 3, magic 1.96, trecent 8, tdecay 0.8,
   forecast 1, timedr 1, tbdecay 0.8.
   Ratios of v2 to the best tuned baseline (v1 RMSE / v2 RMSE x1e-3):
   S0 1.03, S1 0.94*, S2 1.02, S3a 0.86* (6.46 / 6.68), S3b 1.12, S4 1.02 (10.26 / 12.43),
   S0#m3 1.01, S3a#m3 0.88* (6.36 / 6.45), S4#m3 2.05 (36.0 / 22.3; tbdecay 10.9).
   The misspec-3 simulator reproduces the OBD failure only for gradual drift (S4), not for S3a.
   v2 cuts S4#m3 error by 38% but still loses about 2x, and costs about 20% in S4 at misspec 0.5.
28. **Dev iteration after dev2 (2026-10-27, user chose to iterate; dev seeds only;
   scripts/08_diag_s4_dev.py, scripts/08_proto_v2_dev.py)**. S4 (not gradual drift: old third has a
   reward shift under exploratory logger A, middle third a context shift, newest third a poorly
   exploring logger) at misspec 3, 32 episodes:
   - The drift correction overshoots: the old segment's bias goes from +12.7e-3 uncorrected to
     -30 to -40e-3 corrected. v1 gives it most of the weight because it is the most precise.
   - The reference segment is noisy (ESS ~740 out of 40k, sd ~21e-3) because its loggers barely
     explore, so validating against it cannot separate a -37e-3 bias from noise.
   - 10/32 episodes split the stable middle third from the newest (a false split caused by
     misspecification under context shift).
   Prototype "both": each old segment enters corrected and uncorrected, weighted by the
   reference-validated MSE (covariance now accounts for the shared correction data). On 48
   episodes per cell, S4#m3: v1 34.3, v2(0.5) 21.8, both(0.5) 19.3, best baseline 11.7 (tbdecay).
   S4 at misspec 0.5: v1 10.1, v2 14.1, both 15.1. Mean log over cells: v1 2.100, v2 2.084,
   both 2.095, per-cell best baseline 2.041.
   Conclusion: no reuse variant gets below about 19-20 in S4#m3, the floor set by the reference
   noise. tbdecay wins there through its decay prior, not through validation.
29. **Flexible reward model for every method (2026-10-27, user request; dev only so far)**.
   - `nuisance.set_basis`: one context basis shared by every reward model (all estimators and
     baselines; the monitor keeps its quadratic model). Linear (as before) or random Fourier
     features phi(x) = [x, sqrt(2) cos(x Omega + b)]: D features, Gaussian-kernel bandwidth
     sqrt(d), with a fixed Omega and b. RFF is a generic smooth basis, deliberately not the
     [x, x^2] class that generates OBD.
   - Basis rule (method-agnostic, on nuisance quality): the smallest D within 0.1% of the best
     cross-fitted log-loss on dev2 episodes (scripts/09_basis_logloss_dev.py, 4 episodes per
     env and cell). Mean log-loss: linear 0.5718, rff16 0.5603, rff32 0.5552, rff64 0.5549,
     so rff32. Misspec 3: 0.607 -> 0.565; misspec 0.5: 0.535 -> 0.539 (slight overfit).
   - Engineering: `fit_per_action`. The penalised per-action logistic objective separates by
     action exactly, so every reward model (plain, joint-segment, time-aware) is now fit
     action by action at tol 1e-8. The old single big fits stopped at tol 1e-4, with
     coefficient errors of 7.6e-3 (plain) and 0.2 (time-aware). New fits are within 3e-7 /
     5e-5 of a tightly converged big fit. Estimates therefore shift slightly vs the stored
     v1 dev (median 2e-4, max 2e-3 for timedr): optimization error of the OLD fits.
   - The comparison is linear vs rff32 dev2, both with the new fits; every family is re-tuned.
30. **Linear vs rff32 dev2, both on the new fits (2026-10-27)**. Ratio of O-DRIFT RMSE to
   the best tuned baseline, v1 / v2(0.5), x = results/04_dev2_{linear,rff32}.log:
   | cell | linear | rff32 |
   |---|---|---|
   | S0 | 1.03/1.03 | 1.06/1.06 |
   | S1 | 0.94/0.94 | 0.89/0.89 |
   | S2 | 1.02/1.02 | 1.04/1.04 |
   | S3a | 0.83/0.86 | 0.91/0.92 |
   | S3b | 1.12/1.13 | 1.08/1.08 |
   | S4 | 0.90/1.05 | 0.96/1.11 |
   | S0#m3 | 1.01/1.01 | 1.06/1.06 |
   | S3a#m3 | 0.86/0.87 | 0.96/0.96 |
   | S4#m3 | 3.41/2.08 | 1.57/1.55 |
   RFF cuts O-DRIFT's S4#m3 RMSE (v1 37.0 -> 16.5, x1e-3) but adds variance in the
   well-specified cells (S3a 6.47 -> 7.14); baselines barely move (tbdecay S4#m3 10.9 -> 10.5).
   Tuning: linear picks v2 lam 0.5 (tuned_v3_linear.json), rff32 picks lam 0 = v1
   (tuned_v3_rff32.json). With the new linear fits v1 is better than v2 everywhere except S4#m3.
   **Post-hoc dev check (scripts/10_basis_select_dev.py), labeled as such**: choose the basis per
   episode by cross-fitted log-loss of the pooled reward model (linear vs rff32). It picks linear
   in 100% of misspec-0.5 episodes and rff32 in 100% of misspec-3 episodes, so the selected
   method gets the linear numbers in well-specified cells and the rff32 numbers under
   misspecification: v1 max ratio 1.57 (S4#m3), v2 1.55. The v2 robustness bar (<= 1.5) would
   still fail narrowly on dev. Decision on v3 pending with the user.
31. **v3 dev round (2026-10-27): novelty evidence + S4#m3 (user: "lets do b again", "support novelty").**
   - Prior-art check: COPE (Natsubori, Ushiku & Saito, ICLR 2025) is the closest method: target
     cluster of source domains chosen by |rbar_k - rbar_T| (mechanism-blind), multiple importance
     weighting within the cluster, regression for domain-specific effects; out-of-cluster data
     discarded. Also new for related work: Chandak et al. 2022 (OPEN), Shimizu et al. 2025 (OPFV).
   - Added: COPE baseline (windows = domains, anchor = target domain, size grid {2,4,7,13});
     grouping ablation (same O-DRIFT pipeline, segmentation statistic replaced by per-action raw
     reward or overall raw reward = COPE's statistic); per-episode basis selection (PLAN 30);
     notes on the monitor's invariance (docs/notes_monitor_invariance.md).
   - S4#m3 diagnosis with rff32 (32 episodes): own-DM bias of old segments +13, corrected -10:
     the correction overshoots. Prototype (scripts/11_proto_offset_dev.py, 12 seeds x 4 envs):
     shrinking the segment offsets (offset_scale s; L2 x 1/s^2) -- mean log RMSE s=1 2.009,
     0.5 1.977, 0.25 1.940, 0.1 1.981, per-cell best baseline 2.010. s=0.25: S4#m3 10.5 vs best
     baseline 10.9; S4 8.8 vs 11.7; S3a 6.5 vs 7.6; S3b 9.1 vs 8.2. Added as the odrift3 family
     (grid {1, 0.5, 0.25, 0.1}), tuned on dev3. Read: ridge prior on drift size = partial pooling
     between corrected and uncorrected reuse.
   - Smoke (3 seeds x 4 envs): in S2 the blind statistics split into 2-3 segments (O-DRIFT 1):
     RMSE 7.2-7.4 vs 5.7 -- the mechanism claim in miniature.
   - Full dev3 running (results/03_dev3.json). Test3/obd3 (test2/real2 seeds) only after the
     user approves the v3 pre-registration (protocol v3.status -> FROZEN).
32. **dev3 (2026-10-28; results/03_dev3.json, tuned_v3.json, logs/04_dev3.log)**. Tuned: odrift3
   offset_scale 0.25, COPE size 13, blind grouping mean@1e-4. odrift3 / best tuned baseline: S0 1.03,
   S1 0.99, S2 1.02, S3a 0.83*, S3b 1.19 (CI>0, vs timedr), S4 0.76*, S0#m3 1.06 (CI>0), S3a#m3 0.91*,
   S4#m3 1.08 (was 1.57 under v2 rff32, 3.41 linear). Dev go/no-go analogues pass. Hetero coverage
   0.95/0.96/0.94/0.92/0.72/0.92/0.94/0.88/0.82 (S3b gap as before).
   COPE[13] / odrift3: S3a 2.06, S3b 1.39, S4 1.52, S3a#m3 2.17, S4#m3 1.18; ties without reward drift
   (13 = all windows, i.e. COPE degenerates to balanced pooling; smaller clusters lose).
   Grouping ablation (same pipeline, statistic swapped), false-split rate n_seg>1 residual / mean / action:
   S1 0.00 / 0.31 / 1.00, S2 0.00 / 1.00 / 1.00, S0 0 / 0 / 0. Missed splits in S3a: mean statistic
   finds the change in 0.76 of episodes vs 1.00. RMSE ablation(mean) / odrift3: S1 1.08, S2 1.27,
   S3a 1.02, S3b 0.98, S4 0.97, S3a#m3 1.09, S4#m3 1.06. Draft go_no_go_v3 (N1 vs COPE, N2 ablation)
   written to protocol.yaml for user approval; status still DEV.
33. **v3 TEST RESULTS (2026-10-28, one shot, frozen v3; results/03_test3.json, 03_test3_anchor0.json,
   03_obd3.json, 04_test3_summary.json, 04_obd3_summary.json, logs/04_{test3,obd3}.log)**.
   Verdict: the five v1 criteria PASS (wins S4@0.25 0.86, S4@1.0 0.65; S0/S1/S2 1.00-1.07; split coverage;
   S7 1.000; no anchor 1.000). Novelty N1 and N2 PASS: O-DRIFT beats tuned COPE (size 13) in 15/15
   reward-drift cells (0.19-0.93x), ties without reward drift; monitor false-split rate 0.00 in S0/S1/S2 vs
   blind grouping 0.22/0.61 (S1) and 0.52 (S2), ablation 1.07-1.13x worse there and never better
   anywhere (1.01-2.35x in reward-drift cells). Robustness FAILS: misspec-3 sim cells pass (0.98, 0.81,
   0.80, 0.99) but OBD3 ratios are 3.33 / 7.74 / 3.14 / 6.87 / 3.63 (S3a .5/1, S4 .5/1, S5), all bias
   (+14 to +68e-3). Other test losses: S3a@1.0 1.40, S3b 1.28 / 3.02 (vs timedr), S7 1.52, S6 1.13;
   hetero coverage 0.22-0.34 in S5, 0.27 in S3b@1.0.
   Post-hoc (label as such): basis selection chose linear in every OBD episode; offset shrinkage
   0.25 is what makes OBD worse than v1 (S3a@1 36.4 vs v1 10.0; S4@1 73.8 vs 31.8): the ridge prior on
   drift size, tuned on simulator drift, under-corrects OBD's 3x-larger drift. v1 itself still loses
   (ratios ~1.6-2.4 vs monitor_select_dr). monitor_select_dr (our monitor + newest segment only) is the
   best OBD method in S3a/S4 (4.7, 13.5).
34. **dev4 (2026-10-29; results/03_dev4.json, tuned_v4.json, logs/04_dev4.log)**. Gated reuse, z_gate fixed
   1.96 (not tuned; dev mean log RMSE 3.0 -4.901, 1.96 -4.866, 1.0 -4.814 vs v1 -4.905, v3 -4.964).
   odrift4[1.96] / best tuned baseline: S0 1.03, S1 0.99, S2 1.02, S3a 0.86*, S3b 1.14 (CI>0), S4 1.11
   (v1 was 0.86-ish, v3 0.76), S0#m3 1.06, S3a#m3 0.96, S4#m3 1.72 (CI>0; v1 1.57, v3 1.08).
   Dev analogues of the frozen criteria FAIL: only one <= 0.9 win (S3a); S4#m3 > 1.5. N1 fails in
   S4#m3 (1.35x COPE). Mechanism: in gradual drift (S4) the reference is the newest, small segment, so
   the gate drops useful corrected segments and variance rises (bias -> ~0, sd up). The gate cannot
   tell OBD-type bias from useful reuse at the anchor/reference sample sizes. v4 NOT taken to test
   (test4/real4 seeds untouched). Negative result, report as such.
35. **v5 smoke, stopped before dev5 (2026-10-29; 12 dev episodes x S2, S4, S4#m3)**. Rolling-origin back-test
   (src/odrift/backtest.py, protocol v5 DEV) chooses among v1 reuse / shrink 0.25 / reference-only.
   RMSE x1e-3 odrift5 vs v1 / v3 / reference: S2 7.2 vs 5.7 / 5.7 / 8.3; S4 18.6 vs 11.9 / 10.4 / 20.0;
   S4#m3 19.4 vs 13.0 / 9.8 / 21.6. Back-test RMSEs are 10-130e-3, 5-10x the real errors: the validation
   value is DR on 4000 rows of one window (windows are 5000 rows, 1000 go to the pseudo-anchor;
   anchor DR RMSE is 42e-3 at 1000 rows, so ~21e-3 here), so selection is noise-driven. Same limit as the
   anchor gate (v4): a window is only ~5x the anchor. [corrected 2026-10-29] Finding: with per-window samples of this size no
   data-driven check can detect reuse bias of ~10e-3 (minimal detectable bias ~ 2.8 sd/sqrt(n)), which is
   why every validation-based fix trades the simulator against OBD. v5 not run on dev; test4/real4 unused.
36. **Revisions table, matched dev episodes (2026-10-29; results/03_dev5_r13.json = dev5 with the first 13
   data seeds per env, 52 episodes/cell; scripts/13_revisions_table.py -> paper/tables/tab_revisions.tex,
   results/13_revisions_summary.json)**. RMSE ratio to best tuned baseline v1 / v3 / v4 / v5:
   S4 0.89 / 0.71 / 1.32 / 1.53; S4#m3 1.32 / 0.94 / 1.61 / 1.79; S1 v5 1.41; S2 v5 1.24. v5 choices are
   ~1/3 each in every cell (incl. no-drift cells); v5 worse than the best fixed version in 6/9 cells
   (up to 2.1x), slightly better in S3b and S0#m3. Paper text corrected accordingly ("every cell" was wrong).
   Paper: title changed by user to "O-Drift: Observability-Aware Off-Policy Evaluation in Evolving
   Recommender Systems"; observability framing added; kept at 12 pages (PAKDD format, venue TBD).
37. **Anchor-size sweep, POST-HOC (2026-10-29; protocol v3.anchor_sweep_v3; scripts/run_v3_anchor_sweep.sh,
   scripts/14_anchor_sweep.py -> results/14_anchor_sweep.json, paper/figures/fig_anchor.pdf)**. Frozen v3 on
   the test3 seeds, eight cells, n_anchor 100 and 10000 (1000 = the original test3, 0 = test3 without anchor).
   Run note: at n_anchor 10000, S3b took ~3.5 h (other cells ~10 min each); the run was stopped by request
   after S5 was checkpointed, S8 was rerun alone with --cells S8@1.0 and merged into
   results/03_test3_anchor10000.json (same grid, seeds and code). RMSE x1e-3 O-Drift at 100/1000/10000:
   S4 14.2/14.1/11.1 (no anchor 68.5); S3a 9.9/9.8/8.8; S8 7.0/6.9/6.3. Anchor-only DR 153/42/12.
   Finding: O-Drift barely depends on anchor size (segmentation uses history; 100 rows identify the
   reference), but under gradual/seasonal drift (S3b, S5) a 10000-row anchor alone beats it (11.9 vs
   23.5 / 29.3) and its coverage stays <0.30, as reuse bias does not shrink with n_T. Added to the paper
   as a short post-hoc paragraph (text only; figure kept out for the 12-page limit). Bib venue names
   abbreviated to stay at 12 pages.
38. **v6 pre-registration and real-reward test (2026-10-29; protocol v6, FROZEN before any v6 result).** User:
   "do everything make a strong paper"; reviewers of the user's earlier papers criticised simple baselines.
   Baseline audit: COPE matches Natsubori et al. Eqs. 9-10 (pooled reward model, MIW with density ratio in the
   target cluster, mean-reward clustering, size tuned); OPFV was only approximated by time-aware DR, so OPFV
   (Shimizu et al., KDD 2025, Eq. 4 + Eq. 7 feature choice) was implemented (baselines.opfv; degree fixed = 1).
   Real data: scripts/15_prepare_obd_real.py (all 3 campaigns, 19 time blocks, BTS item frequencies per position
   as pi_b, BTS on-policy CTR as truth). real6 = 21 tasks x 20 episodes. Smoke runs: dev6 (dev seeds) and one
   real task with errors suppressed. Result (scripts/16_real_analysis.py): E1 PASS, O-Drift RMSE 1.141e-3 vs
   best baseline COPE[13] 1.086e-3, ratio 1.05 [1.015, 1.092] (task-cluster bootstrap); per campaign men 1.105,
   women 1.068, all 0.966 [0.943, 0.995]; monitor split rate 0.007; coverage 0.745. Truth varies 4.1-8.0e-3
   across target blocks, below estimator noise -> pooling near-optimal, as predicted in expectation_real6.
   v3 shrinkage beats v1 on men (1.49 vs 3.56). OPFV 4.56 (chooses fine features, high variance).
   Failure: in 4/420 episodes the 1000-row anchor had zero clicks; its plug-in variance is 0 and
   method.select_weights' warm start divides by it -> NaN for O-Drift and every component-pooling baseline
   (transport family). Analysis on the 416 common episodes; failures reported. Code left unchanged while
   test6 runs.
39. **Post-hoc after v6 (2026-10-29; user: "run everything").** (a) Zero-click anchor fix:
   ODriftConfig.anchor_var_floor (default False = frozen behaviour, so every pre-registered run is unchanged):
   an anchor whose rewards are all equal gets the correction variance of one pseudo-success,
   mean(rho^2) p(1-p)/n with p = 1/(n+2) (method.laplace_var_floor), instead of 0. scripts/18_real_floor.py
   reruns frozen O-Drift on all 420 real6 episodes with and without it; the run without it must reproduce
   results/03_real6.json exactly (asserted). Spot check before the full run: the 4 failed episodes all have 0
   anchor clicks and get finite estimates (6.12/4.51/4.01/4.60e-3 vs truth 7.01/4.49/4.49/4.10e-3); an
   episode with clicks is unchanged. Reported as post-hoc; the pre-registered E1 result stays the 416-episode
   one. (b) scripts/17_runtime.py: per-episode wall clock of O-Drift and the main baselines, each end to end
   with its own models, single thread, after all pool runs (machine idle), dev-type seeds 90000+ (outside
   every split). (c) scripts/run_v6_posthoc.sh chains: 04_analyze test6 -> 18 -> 17 after v6_noanchor.done.
40. **v6 test6 replication result (2026-10-30; results/03_test6.json, 03_test6_anchor0.json,
   04_test6_summary.json).** Frozen v3 method on the fresh test4 environments (500-series seeds never
   used before), 23 cells x 300 episodes, no missing estimates. All 14 registered criteria PASS: C1 (wins
   S3a@0.25 0.84, S4@0.25 0.88), C2-C5, R1/R2 on misspec-3 only (max 1.16, 3/4 <= 1), N1 (COPE beaten
   15/15), N2 (0 false splits; blind 0.01-0.77), N3 (OPFV beaten 15/15, ratios 0.15-0.77). Not replicated:
   S4@1.0 0.65 -> 1.01 (tbdecay), S4@1.0#m3 0.99 -> 1.16. Replicated losses: S3b@1.0 2.49, S3a@1.0 1.21,
   S7 1.41. OPFV's Eq. 7 rule is nearly uniform over features (all-windows chosen 19% in S0) because its
   bias term is measured against the finest-feature (anchor-only) estimate. Runtime (17_runtime, idle
   machine, one thread): O-Drift 1.0 s sim / 1.5 s OBD per episode and policy; COPE 0.58/0.27; pooled
   0.49/0.11. Paper: replication + OPFV columns in Table 1, related-work table dropped for space (back to
   12 pages). Fix found while writing: 18_real_floor first compared against the oracle-tuned decay grid;
   now restricted to the tuned baselines of 16_real_analysis.
   Floor rerun (18_real_floor, corrected baseline set): frozen O-Drift reproduced on all 420 episodes; the
   floor changes exactly the 4 zero-click episodes; on all 420, ratio to COPE[13] 1.051 [1.015, 1.092],
   coverage 0.748 (same as the registered 416-episode result). One sentence added to the paper, labelled post hoc.
