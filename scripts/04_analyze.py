"""Analyze 03_evaluate outputs.

    dev : apply the protocol's family-wise tuning rule -> results/tuned.json
    test: main table, paired bootstrap vs best tuned baseline, go/no-go check

    .venv/bin/python scripts/04_analyze.py --split dev
    .venv/bin/python scripts/04_analyze.py --split test
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FAMILY = re.compile(r"^(recent_dr|decay_dr|liu_reg|odrift_split|odrift2|odrift|slope|magic|trecent|tdecay|forecast|timedr|tbdecay)\[(.+)\]$")
FAMILY_KEY = {"recent_dr": "recent", "decay_dr": "decay", "liu_reg": "liu", "odrift": "odrift", "odrift2": "odrift2",
              "slope": "slope", "magic": "magic", "trecent": "trecent", "tdecay": "tdecay",
              "forecast": "forecast", "timedr": "timedr", "tbdecay": "tbdecay"}
BASELINES = ["pooled_dr", "anchor_dr", "recent_dr", "decay_dr", "minvar_dr", "liu_reg", "transport_dr"]


def per_rep(cell, method, field="err"):
    """(reps, policies) array of a field."""
    return np.array([[p[field] for p in r[method]] for r in cell["reps"]], dtype=float)


def rmse(cell, method):
    return float(np.sqrt(np.mean(per_rep(cell, method) ** 2)))


def coverage(cell, method, field="hw"):
    err, hw = per_rep(cell, method), per_rep(cell, method, field)
    return float(np.mean(np.abs(err) <= hw)) if np.isfinite(hw).all() else float("nan")


def regret(cell, method):
    """Mean policy-selection regret: V(best) - V(argmax of estimates)."""
    est = per_rep(cell, method, "est")
    out = []
    for r, env in enumerate(cell["env_of_rep"]):
        truths = np.array(cell["truths"][str(env)])
        out.append(truths.max() - truths[np.argmax(est[r])])
    return float(np.mean(out))


def paired_bootstrap(cell, a, b, n_boot=10_000, seed=0):
    """95% CI of RMSE(a) - RMSE(b), resampling replications (policies kept together)."""
    sa = (per_rep(cell, a) ** 2).mean(axis=1)
    sb = (per_rep(cell, b) ** 2).mean(axis=1)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(sa), size=(n_boot, len(sa)))
    d = np.sqrt(sa[idx].mean(axis=1)) - np.sqrt(sb[idx].mean(axis=1))
    return float(np.sqrt(sa.mean()) - np.sqrt(sb.mean())), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def tune(data):
    """Family-wise: minimize mean over dev cells of log RMSE."""
    scores = defaultdict(lambda: defaultdict(list))
    for cell in data["cells"].values():
        for method in cell["reps"][0]:
            m = FAMILY.match(method)
            if m and m.group(1) in FAMILY_KEY:
                scores[FAMILY_KEY[m.group(1)]][m.group(2)].append(np.log(rmse(cell, method)))
    tuned, report = {}, {}
    for fam, settings in scores.items():
        means = {s: float(np.mean(v)) for s, v in settings.items()}
        best = min(means, key=means.get)
        report[fam] = dict(sorted(means.items(), key=lambda kv: kv[1]))
        parsed = [int(x) if x.lstrip("-").isdigit() else float(x) for x in best.split(",")]
        int_families = {"recent", "liu", "trecent", "forecast", "timedr"}  # window counts; everything else stays float
        tuned[fam] = parsed if len(parsed) > 1 else (int(parsed[0]) if fam in int_families else parsed[0])
    return tuned, report


def table(data, methods):
    rows = []
    for key, cell in data["cells"].items():
        for m in methods:
            if m not in cell["reps"][0]:
                continue
            rows.append(dict(cell=key, method=m, rmse=rmse(cell, m), cov=coverage(cell, m), regret=regret(cell, m),
                             cov_narrow=coverage(cell, m, "hw_narrow") if "hw_narrow" in cell["reps"][0][m][0] else None))
    return rows


def print_table(rows):
    cells = list(dict.fromkeys(r["cell"] for r in rows))
    methods = list(dict.fromkeys(r["method"] for r in rows))
    by = {(r["cell"], r["method"]): r for r in rows}
    print("\nRMSE (x1e3)  [coverage@95]  {regret x1e3}")
    print(f"{'method':24s}" + "".join(f"{c:>20s}" for c in cells))
    for m in methods:
        line = f"{m:24s}"
        for c in cells:
            r = by.get((c, m))
            if r is None:
                line += f"{'':>20s}"
                continue
            cov = "" if np.isnan(r["cov"]) else f"[{r['cov']:.2f}]"
            line += f"{r['rmse'] * 1e3:7.2f}{cov:>7s}{{{r['regret'] * 1e3:4.1f}}}"
        print(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test", "dev2", "test2", "obd2"], required=True)
    ap.add_argument("--file", default=None)
    args = ap.parse_args()
    data = json.loads((ROOT / (args.file or f"results/03_{args.split}.json")).read_text())
    v2 = args.split.endswith("2")
    tuned_path = ROOT / ("results/tuned_v2.json" if v2 else "results/tuned.json")

    if args.split.startswith("dev"):
        tuned, report = tune(data)
        for fam, means in report.items():
            print(f"{fam:8s} " + "  ".join(f"{s}:{v:.3f}" for s, v in list(means.items())[:5]))
        tuned_path.write_text(json.dumps(tuned, indent=2))
        print(f"\ntuned: {tuned}  -> {tuned_path.relative_to(ROOT)}")
    else:
        tuned = json.loads(tuned_path.read_text())

    def name(fam, prefix):
        v = tuned[fam]
        return f"{prefix}[{','.join(str(x) for x in v)}]" if isinstance(v, list) else f"{prefix}[{v}]"

    od, ods = name("odrift", "odrift"), name("odrift", "odrift_split")
    od1 = od
    if v2:  # the method under test is v2; v1 (same fits) is reported alongside
        od = name("odrift2", "odrift2")
    tuned_baselines = ["pooled_dr", "anchor_dr", name("recent", "recent_dr"), name("decay", "decay_dr"),
                       "minvar_dr", name("liu", "liu_reg"), "transport_dr"]
    for fam, prefix in (("slope", "slope"), ("magic", "magic"), ("trecent", "trecent"), ("tdecay", "tdecay"),
                        ("forecast", "forecast"), ("timedr", "timedr"), ("tbdecay", "tbdecay")):
        if fam in tuned:
            tuned_baselines.append(name(fam, prefix))
    tuned_baselines += ["anchor_validated", "balanced_dr", "monitor_select_dr"]
    # per-cell oracle-tuned upper bounds (use the truth to pick k / gamma): reported, never "tuned"
    for key, cell in data["cells"].items():
        for fam in ("recent_dr", "decay_dr"):
            members = [m for m in cell["reps"][0] if m.startswith(fam + "[")]
            if members:
                best_m = min(members, key=lambda m: rmse(cell, m))
                for r in cell["reps"]:
                    r[f"{fam}_oracletuned"] = r[best_m]
    methods = tuned_baselines + ([od1] if v2 else []) + [od, ods, "recent_dr_oracletuned", "decay_dr_oracletuned",
                                 "ablate_mse", "ablate_bound", "oracle"]
    rows = table(data, methods)
    print_table(rows)

    print(f"\nO-DRIFT ({od}) vs best tuned baseline per cell: RMSE diff, paired 95% CI, ratio")
    summary = {}
    for key, cell in data["cells"].items():
        if od not in cell["reps"][0]:
            continue
        base = min((m for m in tuned_baselines if m in cell["reps"][0]), key=lambda m: rmse(cell, m))
        d, lo, hi = paired_bootstrap(cell, od, base)
        ratio = rmse(cell, od) / rmse(cell, base)
        cov = coverage(cell, ods)
        liu = name("liu", "liu_reg")
        summary[key] = dict(best_baseline=base, diff=d, ci=[lo, hi], ratio=ratio, coverage_split=cov,
                            coverage_hetero=coverage(cell, od), halfwidth_hetero=float(np.mean(per_rep(cell, od, "hw"))),
                            coverage_plugin_narrow=coverage(cell, od, "hw_narrow"),
                            ratio_v1=rmse(cell, od1) / rmse(cell, base) if v2 else None,
                            coverage_certified_liu=coverage(cell, liu),
                            halfwidth_certified_liu=float(np.mean(per_rep(cell, liu, "hw"))))
        print(f"  {key:10s} vs {base:18s} diff={d * 1e3:+6.2f}e-3 CI=[{lo * 1e3:+6.2f},{hi * 1e3:+6.2f}] "
              f"ratio={ratio:.2f}  cov(hetero)={summary[key]['coverage_hetero']:.3f}"
              f"[hw {summary[key]['halfwidth_hetero'] * 1e3:.1f}]  cov(narrow)={summary[key]['coverage_plugin_narrow']:.3f}"
              f"  cov(certified Liu)={summary[key]['coverage_certified_liu']:.3f}[hw {summary[key]['halfwidth_certified_liu'] * 1e3:.1f}]")

    # pre-registered breakdowns (protocol endpoints.breakdowns)
    print("\nBreakdown: ratio to best tuned baseline per env seed | per candidate policy ; O-DRIFT bias/sd x1e3")
    for key, cell in data["cells"].items():
        if key not in summary:
            continue
        base = summary[key]["best_baseline"]
        eo = np.array([[p["err"] for p in r[od]] for r in cell["reps"]])
        eb = np.array([[p["err"] for p in r[base]] for r in cell["reps"]])
        env = np.array(cell["env_of_rep"]).astype(str)
        by_env = [np.sqrt((eo[env == e] ** 2).mean() / (eb[env == e] ** 2).mean()) for e in sorted(set(env))]
        by_pol = np.sqrt((eo ** 2).mean(0) / (eb ** 2).mean(0))
        summary[key].update(ratio_by_env=[float(x) for x in by_env], ratio_by_policy=by_pol.tolist(),
                            bias=float(eo.mean()), sd=float(eo.std()))
        print(f"  {key:10s} env " + " ".join(f"{x:.2f}" for x in by_env) + " | pol " + " ".join(f"{x:.2f}" for x in by_pol)
              + f" ; bias {eo.mean() * 1e3:+.1f} sd {eo.std() * 1e3:.1f}")

    if args.split in ("test", "test2"):
        go = []
        main_cells = {k: v for k, v in summary.items() if "#m" not in k}  # v2: v1 criteria on misspec-0.5 cells
        full_summary, summary = summary, main_cells
        drift = [k for k in summary if k.split("@")[0] in ("S3a", "S3b", "S4", "S5")]
        wins = [k for k in drift if summary[k]["ratio"] <= 0.90 and summary[k]["ci"][1] < 0]
        go.append((f"RMSE <= 0.9x best in >= 2 reward-drift cells (won {len(wins)}: {wins})", bool(len(wins) >= 2)))
        stat = [k for k in summary if k.split("@")[0] in ("S0", "S1", "S2")]
        go.append(("RMSE <= 1.1x best in S0/S1/S2", bool(all(summary[k]["ratio"] <= 1.10 for k in stat))))
        go.append(("coverage(split) >= 0.93 everywhere", bool(all(summary[k]["coverage_split"] >= 0.93 for k in summary))))
        s7 = [k for k in data["cells"] if k.startswith("S7@")]
        if s7:
            cell = data["cells"][s7[0]]
            frac_ok = np.mean([sum(p["weights"][w] for w in p["deficient"]) <= 0.05
                               for r in cell["reps"] for p in r[od]])
            go.append((f"S7: weight on support-deficient windows <= 0.05 in >= 90% (got {frac_ok:.3f})", bool(frac_ok >= 0.90)))
        no_anchor = ROOT / f"results/03_{args.split}_anchor0.json"
        if no_anchor.exists():
            na = json.loads(no_anchor.read_text())
            na_key = od if v2 else "odrift"
            statuses = [p["status"] for c in na["cells"].values() for r in c["reps"] for p in r[na_key]]
            frac = np.mean([s_ == "not_certified" for s_ in statuses])
            go.append((f"no anchor -> not certified in 100% (got {frac:.3f})", bool(frac == 1.0)))
        else:
            go.append(("no-anchor run missing (run 03_evaluate --split test --n-anchor 0)", False))
        summary = full_summary
        if v2:  # pre-registered robustness criterion (protocol v2.go_no_go_v2)
            robust = {k: summary[k]["ratio"] for k in summary if "#m" in k and k.split("@")[0] in ("S3a", "S4", "S5", "S8")}
            obd2 = ROOT / "results/04_obd2_summary.json"
            if obd2.exists():
                robust.update({f"obd2:{k}": v["ratio"] for k, v in json.loads(obd2.read_text())["summary"].items()
                               if k.split("@")[0] in ("S3a", "S4", "S5")})
            else:
                go.append(("obd2 summary missing (run 04_analyze --split obd2 first)", False))
            print("  robustness cells: " + "  ".join(f"{k}={v:.2f}" for k, v in robust.items()))
            go.append((f"robustness: no reward-drift cell (misspec 3 + obd2) ratio > 1.5 (max {max(robust.values()):.2f})",
                       bool(max(robust.values()) <= 1.5)))
            n_ok = sum(v <= 1.0 for v in robust.values())
            go.append((f"robustness: ratio <= 1.0 in >= half ({n_ok}/{len(robust)})", bool(2 * n_ok >= len(robust))))
        for crit, ok in go:
            print(f"  [{'PASS' if ok else 'FAIL'}] {crit}")
        (ROOT / f"results/04_{args.split}_summary.json").write_text(json.dumps(dict(summary=summary, go=go), indent=2))
    elif args.split == "obd2":
        (ROOT / "results/04_obd2_summary.json").write_text(json.dumps(dict(summary=summary), indent=2))


if __name__ == "__main__":
    main()
