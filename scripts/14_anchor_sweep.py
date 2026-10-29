"""Post-hoc anchor-size sweep for the frozen v3 configuration (protocol v3.anchor_sweep_v3).

Reads results/03_test3_anchor{0,100,10000}.json and results/03_test3.json (anchor 1000) for the
sweep cells and writes paper/figures/fig_anchor.pdf and results/14_anchor_sweep.json:
RMSE of O-Drift, anchor-only DR and pooled DR, and coverage of O-Drift's interval, per anchor size.

    PYTHONPATH=src .venv/bin/python scripts/14_anchor_sweep.py
"""
import json
from pathlib import Path

import matplotlib
import numpy as np
import yaml

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OD = "odrift3[0.25]"
SIZES = [0, 100, 1000, 10000]


def load(n):
    path = ROOT / ("results/03_test3.json" if n == 1000 else f"results/03_test3_anchor{n}.json")
    return json.loads(path.read_text())["cells"]


def rmse(cell, m):
    return float(np.sqrt(np.mean([[p["err"] ** 2 for p in r[m]] for r in cell["reps"]]))) * 1e3


def coverage(cell, m):
    return float(np.mean([[abs(p["err"]) <= p["hw"] for p in r[m]] for r in cell["reps"]]))


def main():
    cells = yaml.safe_load((ROOT / "configs/protocol.yaml").read_text())["simulator"]["anchor_sweep_cells"]
    data = {n: load(n) for n in SIZES}
    out = {}
    for key in cells:
        row = {}
        for n in SIZES:
            c = data[n][key]
            row[n] = dict(odrift=rmse(c, OD), pooled=rmse(c, "pooled_dr"),
                          anchor=rmse(c, "anchor_dr") if n else None,
                          cov=coverage(c, OD) if n else None)
        out[key] = row
        print(f"{key:9s} " + "  ".join(f"n={n}: od {r['odrift']:5.1f} anc {r['anchor'] or float('nan'):5.1f} "
                                       f"cov {r['cov'] if r['cov'] is not None else float('nan'):.2f}"
                                       for n, r in row.items()))
    (ROOT / "results/14_anchor_sweep.json").write_text(json.dumps(out, indent=2))

    fig, axes = plt.subplots(1, 2, figsize=(4.8, 2.0))
    xs = [1, 2, 3]  # anchor sizes 100, 1000, 10000 (log-spaced positions)
    for key in cells:
        axes[0].plot(xs, [out[key][n]["odrift"] / out[key][n]["anchor"] for n in SIZES[1:]], marker="o", ms=2.5,
                     lw=0.8, label=key.split("@")[0])
        axes[1].plot(xs, [out[key][n]["cov"] for n in SIZES[1:]], marker="o", ms=2.5, lw=0.8)
    axes[0].axhline(1, color="0.6", lw=0.6)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("RMSE O-Drift / anchor-only", fontsize=6)
    axes[1].axhline(0.95, color="0.6", lw=0.6)
    axes[1].set_ylabel("coverage (nominal 0.95)", fontsize=6)
    axes[1].set_ylim(0, 1.02)
    for ax in axes:
        ax.set_xticks(xs, ["100", "1,000", "10,000"], fontsize=5)
        ax.set_xlabel("anchor size", fontsize=6)
        ax.tick_params(axis="y", labelsize=5)
    axes[0].legend(fontsize=4.5, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(ROOT / "paper/figures/fig_anchor.pdf")
    print("wrote paper/figures/fig_anchor.pdf, results/14_anchor_sweep.json")


if __name__ == "__main__":
    main()
