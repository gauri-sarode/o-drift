"""Analysis of the real-reward Open Bandit test (protocol v6: E1-E3).

Reads results/03_real6.json and data/obd_real_*.npz (truth noise) and writes
results/16_real6_summary.json and paper/tables/tab_real.tex.

Episodes in which any reported method returned no estimate are excluded for every method (common
episodes) and counted per method. RMSE over episodes; ratio to the best tuned baseline with a
paired cluster bootstrap over the 21 tasks; noise-corrected RMSE sqrt(max(MSE - mean truth var, 0)).

    PYTHONPATH=src .venv/bin/python scripts/16_real_analysis.py
"""
import json
from pathlib import Path

import numpy as np

from odrift import obd_real

ROOT = Path(__file__).resolve().parents[1]
TUNED = json.loads((ROOT / "results/tuned_v6.json").read_text())
OD = f"odrift3[{TUNED['odrift3']}]"
V1 = "odrift[{}]".format(",".join(str(x) for x in TUNED["odrift"]))
BASELINES = ["pooled_dr", "anchor_dr", f"recent_dr[{TUNED['recent']}]", f"decay_dr[{TUNED['decay']}]", "minvar_dr",
             f"liu_reg[{TUNED['liu']}]", "transport_dr", f"slope[{TUNED['slope']}]", f"magic[{TUNED['magic']}]",
             f"trecent[{TUNED['trecent']}]", f"tdecay[{TUNED['tdecay']}]", f"forecast[{TUNED['forecast']}]",
             f"timedr[{TUNED['timedr']}]", f"tbdecay[{TUNED['tbdecay']}]", f"cope[{TUNED['cope']}]",
             "anchor_validated", "balanced_dr", "monitor_select_dr", f"opfv[{TUNED['opfv']}]"]
LABEL = {"pooled_dr": "Pooled DR", "anchor_dr": "Anchor-only DR", "minvar_dr": "Min-variance pooling",
         "transport_dr": "Transport DR", "anchor_validated": "Anchor-validated", "balanced_dr": "Balanced DR",
         "monitor_select_dr": "Monitor-select"}


def label(m):
    if m in LABEL:
        return LABEL[m]
    fam = m.split("[")[0]
    return {"recent_dr": "Recent-$k$ DR", "decay_dr": "Decayed DR", "liu_reg": "Regression-assisted DR",
            "slope": "SLOPE", "magic": "MAGIC", "trecent": "Transport recent", "tdecay": "Transport decay",
            "forecast": "Forecasting", "timedr": "Time-aware DR", "tbdecay": "Transport balanced decay",
            "cope": "COPE", "opfv": "OPFV", "odrift3": r"\odrift{}", "odrift": r"\odrift{} ($s=1$)"}.get(fam, m)


def main():
    cell = json.loads((ROOT / "results/03_real6.json").read_text())["cells"]["REAL@0.0"]
    reps, envs = cell["reps"], np.array(cell["env_of_rep"])
    methods = BASELINES + [OD, V1]
    est = {m: np.array([r[m][0]["est"] for r in reps]) for m in methods}
    hw = {m: np.array([r[m][0]["hw"] for r in reps]) for m in methods}
    truth = np.array([cell["truths"][str(e)][0] for e in envs])
    tvar = np.empty(len(envs))
    for e in np.unique(envs):
        camp, b = obd_real.task_of(int(e))
        d = obd_real.load(camp)
        p, n = d["ctr"][b], d["ctr_n"][b]
        tvar[envs == e] = p * (1 - p) / n
    failed = {m: int((~np.isfinite(est[m])).sum()) for m in methods}
    ok = np.all([np.isfinite(est[m]) for m in methods], axis=0)
    sq = {m: (est[m][ok] - truth[ok]) ** 2 for m in methods}
    env_ok, tv = envs[ok], tvar[ok].mean()
    rm = {m: float(np.sqrt(sq[m].mean())) for m in methods}
    best = min(BASELINES, key=rm.get)

    tasks = np.unique(env_ok)
    rng = np.random.default_rng(0)
    by_task = {m: {t: sq[m][env_ok == t] for t in tasks} for m in methods}

    def boot(a, b, task_set, n_boot=10_000):
        draws = rng.choice(task_set, size=(n_boot, len(task_set)))
        out = np.empty(n_boot)
        for i, ts in enumerate(draws):
            sa = np.concatenate([by_task[a][t] for t in ts])
            sb = np.concatenate([by_task[b][t] for t in ts])
            out[i] = np.sqrt(sa.mean()) / np.sqrt(sb.mean())
        return np.percentile(out, [2.5, 97.5]).tolist()

    ci = boot(OD, best, tasks)
    split = float(np.mean([r[OD][0]["n_segments"] > 1 for r, k in zip(reps, ok) if k]))
    cov = float(np.mean(np.abs(est[OD][ok] - truth[ok]) <= hw[OD][ok]))
    per_camp = {}
    for ci_, camp in enumerate(obd_real.CAMPAIGNS):
        sel = (env_ok // 100) == ci_
        r_c = {m: float(np.sqrt(sq[m][sel].mean())) for m in methods}
        b_c = min(BASELINES, key=r_c.get)
        per_camp[camp] = dict(rmse=r_c, best=b_c, ratio=r_c[OD] / r_c[b_c],
                              ci=boot(OD, b_c, tasks[(tasks // 100) == ci_]),
                              split=float(np.mean([r[OD][0]["n_segments"] > 1 for r, k, e in zip(reps, ok, envs)
                                                   if k and e // 100 == ci_])))
    summary = dict(episodes=len(reps), common=int(ok.sum()), failed=failed, rmse=rm, best_baseline=best,
                   ratio=rm[OD] / rm[best], ratio_ci=ci, E1_pass=bool(rm[OD] <= 1.10 * rm[best]),
                   split_rate=split, coverage=cov, mean_truth_sd=float(np.sqrt(tv)),
                   rmse_noise_corrected={m: float(np.sqrt(max(sq[m].mean() - tv, 0))) for m in methods},
                   mean_truth=float(truth.mean()), per_campaign=per_camp)
    (ROOT / "results/16_real6_summary.json").write_text(json.dumps(summary, indent=2))

    order = sorted(methods, key=rm.get)
    print(f"episodes {len(reps)}, common {ok.sum()}; failures {[(m, v) for m, v in failed.items() if v]}")
    print(f"mean truth {truth.mean() * 1e3:.2f}e-3, truth sd {np.sqrt(tv) * 1e3:.3f}e-3")
    for m in order:
        pc = "  ".join(f"{c}:{per_camp[c]['rmse'][m] * 1e3:.3f}" for c in per_camp)
        print(f"{m:32s} RMSE {rm[m] * 1e3:.3f}e-3  {pc}")
    print(f"O-Drift {OD}: ratio to best ({best}) {rm[OD] / rm[best]:.3f} CI {np.round(ci, 3).tolist()}  "
          f"E1 {'PASS' if summary['E1_pass'] else 'FAIL'}; split rate {split:.3f}; coverage {cov:.3f}")
    for c, v in per_camp.items():
        print(f"  {c}: best {v['best']} ratio {v['ratio']:.3f} CI {np.round(v['ci'], 3).tolist()} split {v['split']:.3f}")

    cols = [OD, "pooled_dr", f"cope[{TUNED['cope']}]", f"slope[{TUNED['slope']}]", "monitor_select_dr",
            f"decay_dr[{TUNED['decay']}]", f"magic[{TUNED['magic']}]", "transport_dr", f"forecast[{TUNED['forecast']}]",
            f"opfv[{TUNED['opfv']}]", "anchor_dr"]
    short = {OD: r"\odrift{}", "pooled_dr": "Pooled", f"cope[{TUNED['cope']}]": "COPE", f"slope[{TUNED['slope']}]": "SLOPE",
             "monitor_select_dr": "Mon.-sel.", f"decay_dr[{TUNED['decay']}]": "Decay", f"magic[{TUNED['magic']}]": "MAGIC",
             "transport_dr": "Transp.", f"forecast[{TUNED['forecast']}]": "Forec.", f"opfv[{TUNED['opfv']}]": "OPFV",
             "anchor_dr": "Anchor"}
    def row(name, r):
        lo = min(r[m] for m in cols[1:])
        return name + " & " + " & ".join((r"\textbf{%.2f}" if r[m] <= lo else "%.2f") % (r[m] * 1e3) for m in cols) + r" \\"
    rows = [row("All tasks", rm)] + [row({"men": "Men", "women": "Women", "all": "All items"}[c], per_camp[c]["rmse"])
                                     for c in per_camp]
    tex = "\n".join([
        r"\begin{table}[t]\centering\scriptsize\setlength{\tabcolsep}{2.6pt}",
        r"\caption{Real rewards and natural drift (Open Bandit, pre-registered test v6): RMSE ($\times10^{-3}$) against "
        r"BTS's on-policy click rate (mean " + f"{truth.mean() * 1e3:.1f}" + r", noise sd " + f"{np.sqrt(tv) * 1e3:.2f}" +
        r"), 21 tasks $\times$ 20 episodes (" + f"{ok.sum()}" + r" with an estimate from every method). Bold: lowest in the row. "
        r"Ratio of \odrift{} to the best baseline: " + f"{rm[OD] / rm[best]:.2f}" + r" [" + f"{ci[0]:.2f}, {ci[1]:.2f}" + r"].}",
        r"\label{tab:real}",
        r"\begin{tabular}{@{}l" + "c" * len(cols) + r"@{}}\toprule",
        " & " + " & ".join(short[m] for m in cols) + r" \\\midrule",
        *rows,
        r"\bottomrule\end{tabular}",
        r"\end{table}"])
    (ROOT / "paper/tables/tab_real.tex").write_text(tex + "\n")
    print("wrote results/16_real6_summary.json, paper/tables/tab_real.tex")


if __name__ == "__main__":
    main()
