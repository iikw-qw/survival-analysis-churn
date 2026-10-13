import numpy as np
import pandas as pd


def generate_customers(n=5000, seed=33, observation_days=720):
    """Time-to-churn with right censoring.

    Hazard depends on plan, contract type and support tickets. Customers who are still
    active at the end of the observation window are censored.
    """
    rng = np.random.default_rng(seed)
    plan = rng.choice(["basic", "standard", "premium"], n, p=[0.5, 0.35, 0.15])
    contract = rng.choice(["monthly", "annual"], n, p=[0.65, 0.35])
    tickets = rng.poisson(1.2, n)
    age = rng.integers(18, 75, n)

    base_scale = 600.0
    log_hr = (
        np.where(plan == "premium", -0.5, np.where(plan == "standard", -0.2, 0.0))
        + np.where(contract == "annual", -0.8, 0.0)
        + 0.18 * tickets
        - 0.008 * (age - 40)
    )
    hazard_scale = base_scale * np.exp(-log_hr)
    # Weibull with shape 1.3 (hazard grows slowly with tenure)
    true_time = hazard_scale * rng.weibull(1.3, n)
    entry = rng.integers(0, observation_days, n)
    follow_up = observation_days - entry
    observed = np.minimum(true_time, follow_up)
    churned = true_time <= follow_up

    return pd.DataFrame(
        {
            "customer_id": np.arange(n),
            "plan": plan,
            "contract": contract,
            "tickets": tickets,
            "age": age,
            "duration_days": np.maximum(observed, 1).round(),
            "churned": churned.astype(int),
        }
    )
