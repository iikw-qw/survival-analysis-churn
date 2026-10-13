import numpy as np
import pytest

from src import core, data


def test_km_without_censoring_is_empirical_survival():
    km = core.kaplan_meier([1, 2, 3, 4], [1, 1, 1, 1])
    assert core.survival_at(km, 2) == pytest.approx(0.5)
    assert core.survival_at(km, 4) == pytest.approx(0.0)


def test_km_handles_censoring():
    # at t=2 the person at risk count is 3 (one censored at 1 has already left)
    km = core.kaplan_meier([1, 2, 3, 4], [0, 1, 1, 1])
    assert core.survival_at(km, 2) == pytest.approx(1 - 1 / 3)


def test_km_is_monotone_and_bounded():
    df = data.generate_customers(n=1000, seed=1)
    km = core.kaplan_meier(df["duration_days"], df["churned"])
    assert km["survival"].is_monotonic_decreasing
    assert km["survival"].between(0, 1).all()
    assert (km["ci_low"] <= km["survival"] + 1e-12).all()


def test_logrank_detects_difference_and_not_noise():
    df = data.generate_customers(n=3000, seed=2)
    m, a = df[df.contract == "monthly"], df[df.contract == "annual"]
    _, p_diff = core.logrank_test(m.duration_days, m.churned, a.duration_days, a.churned)
    assert p_diff < 0.001
    half = np.arange(len(df)) % 2 == 0
    x, y = df[half], df[~half]
    _, p_same = core.logrank_test(x.duration_days, x.churned, y.duration_days, y.churned)
    assert p_same > 0.01


def test_cox_recovers_effect_directions():
    df = data.generate_customers(n=4000, seed=3)
    hr = core.cox_model(df)["hazard_ratio"]
    assert hr["annual"] < 1 and hr["premium"] < 1 and hr["tickets"] > 1
