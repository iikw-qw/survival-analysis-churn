import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.duration.hazard_regression import PHReg


def kaplan_meier(durations, events):
    """Kaplan-Meier estimate with Greenwood confidence interval."""
    d = np.asarray(durations, float)
    e = np.asarray(events, int)
    times = np.unique(d[e == 1])
    surv, var_sum = 1.0, 0.0
    rows = [(0.0, 1.0, 1.0, 1.0)]
    for t in times:
        at_risk = (d >= t).sum()
        deaths = ((d == t) & (e == 1)).sum()
        surv *= 1 - deaths / at_risk
        if at_risk > deaths:
            var_sum += deaths / (at_risk * (at_risk - deaths))
        se = surv * np.sqrt(var_sum)
        rows.append((t, surv, max(0.0, surv - 1.96 * se), min(1.0, surv + 1.96 * se)))
    return pd.DataFrame(rows, columns=["time", "survival", "ci_low", "ci_high"]).set_index("time")


def survival_at(km, t):
    return float(km["survival"][km.index <= t].iloc[-1])


def median_survival(km):
    below = km.index[km["survival"] <= 0.5]
    return float(below[0]) if len(below) else float("nan")


def logrank_test(d1, e1, d2, e2):
    """Two-group log-rank test; returns chi-square statistic and p-value."""
    d = np.r_[d1, d2]
    e = np.r_[e1, e2]
    g = np.r_[np.zeros(len(d1)), np.ones(len(d2))]
    obs_minus_exp, var = 0.0, 0.0
    for t in np.unique(d[e == 1]):
        at_risk = d >= t
        n = at_risk.sum()
        n1 = (at_risk & (g == 0)).sum()
        deaths = ((d == t) & (e == 1)).sum()
        d1_t = ((d == t) & (e == 1) & (g == 0)).sum()
        obs_minus_exp += d1_t - deaths * n1 / n
        if n > 1:
            var += deaths * (n1 / n) * (1 - n1 / n) * (n - deaths) / (n - 1)
    chi2 = obs_minus_exp**2 / var
    return float(chi2), float(stats.chi2.sf(chi2, 1))


def cox_model(df):
    x = pd.DataFrame(
        {
            "premium": (df["plan"] == "premium").astype(float),
            "standard": (df["plan"] == "standard").astype(float),
            "annual": (df["contract"] == "annual").astype(float),
            "tickets": df["tickets"].astype(float),
            "age": df["age"].astype(float),
        }
    )
    model = PHReg(df["duration_days"].to_numpy(), x, status=df["churned"].to_numpy(), ties="breslow").fit()
    ci = model.conf_int()
    out = pd.DataFrame(
        {
            "hazard_ratio": np.exp(model.params),
            "ci_low": np.exp(ci[:, 0]),
            "ci_high": np.exp(ci[:, 1]),
            "p_value": model.pvalues,
        },
        index=x.columns,
    )
    return out


def retention_table(df, by, horizons=(90, 180, 365)):
    rows = {}
    for key, part in df.groupby(by):
        km = kaplan_meier(part["duration_days"], part["churned"])
        rows[key] = {f"day_{h}": survival_at(km, h) for h in horizons}
        rows[key]["median_days"] = median_survival(km)
        rows[key]["customers"] = len(part)
    return pd.DataFrame(rows).T
