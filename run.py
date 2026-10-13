import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import core, data

OUT = Path("reports")


def main():
    OUT.mkdir(exist_ok=True)
    df = data.generate_customers()
    km_all = core.kaplan_meier(df["duration_days"], df["churned"])
    by_contract = core.retention_table(df, "contract")
    by_plan = core.retention_table(df, "plan")
    cox = core.cox_model(df)

    a = df[df["contract"] == "monthly"]
    b = df[df["contract"] == "annual"]
    chi2, p = core.logrank_test(a["duration_days"], a["churned"], b["duration_days"], b["churned"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for key, part in df.groupby("contract"):
        km = core.kaplan_meier(part["duration_days"], part["churned"])
        axes[0].step(km.index, km["survival"], where="post", label=key, color="black" if key == "annual" else "grey")
        axes[0].fill_between(km.index, km["ci_low"], km["ci_high"], step="post", alpha=0.15, color="grey")
    axes[0].set_xlabel("days since signup")
    axes[0].set_ylabel("share still subscribed")
    axes[0].legend()
    axes[0].set_title("Retention by contract type")
    cox_plot = cox.sort_values("hazard_ratio")
    y = range(len(cox_plot))
    axes[1].errorbar(cox_plot["hazard_ratio"], list(y),
                     xerr=[cox_plot["hazard_ratio"] - cox_plot["ci_low"], cox_plot["ci_high"] - cox_plot["hazard_ratio"]],
                     fmt="o", color="black", capsize=3)
    axes[1].axvline(1, linestyle="--", color="grey")
    axes[1].set_yticks(list(y))
    axes[1].set_yticklabels(cox_plot.index)
    axes[1].set_xlabel("hazard ratio (95% CI)")
    axes[1].set_title("Cox model")
    fig.tight_layout()
    fig.savefig(OUT / "survival.png", dpi=130)

    by_contract.round(3).to_csv(OUT / "retention_by_contract.csv")
    summary = {
        "customers": int(len(df)),
        "censored_share": round(float(1 - df["churned"].mean()), 3),
        "overall_retention_day_365": round(core.survival_at(km_all, 365), 3),
        "retention_day_365_by_contract": {k: round(float(v), 3) for k, v in by_contract["day_365"].items()},
        "logrank_contract_p_value": float(f"{p:.3g}"),
        "hazard_ratios": {k: round(float(v), 2) for k, v in cox["hazard_ratio"].items()},
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
