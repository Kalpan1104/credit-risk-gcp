"""
Synthetic personal-loan applications.

`regime="downturn"` simulates an economic shock: lower incomes, higher debt
loads and utilisation, more credit inquiries and more defaults. It's what the
monitoring job should notice.
"""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from credit_risk import config as C


def generate(n=20_000, seed=0, regime="normal"):
    rng = np.random.default_rng(seed)
    shock = 1.0 if regime == "downturn" else 0.0

    age = rng.normal(41, 12, n).clip(21, 80).round()
    history = ((age - 18) * rng.uniform(0.35, 1.0, n) + rng.gamma(1.5, 1.0, n)).clip(0.2, age - 17).round(1)
    employment = (rng.gamma(2.0, 3.0, n)).clip(0, age - 18).round(1)
    income = np.exp(rng.normal(11.0 + 0.005 * (age - 40) - 0.10 * shock, 0.45, n)).clip(12_000, 600_000).round(-2)
    home = rng.choice(C.ALLOWED["home_ownership"], n, p=[0.42, 0.43, 0.15])
    purpose = rng.choice(C.ALLOWED["purpose"], n, p=[0.42, 0.20, 0.10, 0.10, 0.06, 0.12])
    term = rng.choice([36, 60], n, p=[0.7, 0.3])
    loan = (income * rng.uniform(0.05, 0.45, n)).clip(1_000, 60_000).round(-2)
    dti = (rng.gamma(3.0, 5.5, n) + 6 * shock).clip(0, 60).round(1)
    util = (rng.beta(2.2, 3.2, n) + 0.12 * shock).clip(0, 1.2).round(3)
    delinq = rng.poisson(0.25 + 0.15 * shock, n)
    inquiries = rng.poisson(0.9 + 1.0 * shock, n)
    accounts = rng.poisson(8, n) + 1

    z = (
        -3.15
        + 0.065 * (dti - 18)
        + 3.0 * (util - 0.4)
        + 0.75 * delinq
        + 0.30 * inquiries
        - 0.60 * (np.log(income) - 11)
        - 0.010 * (history - 18)
        - 0.03 * employment
        + 0.35 * (term == 60)
        + 1.1 * (loan / income)
        + np.select([purpose == "medical", purpose == "credit_card", purpose == "home_improvement"], [0.3, 0.15, -0.2], 0)
        - 0.15 * (home == "own")
        + 0.35 * shock
        + rng.normal(0, 0.25, n)
    )
    defaulted = rng.binomial(1, 1 / (1 + np.exp(-z)))

    return pd.DataFrame({
        "application_id": [f"A{seed:04d}-{i:07d}" for i in range(n)],
        "age": age.astype(int), "annual_income": income, "employment_years": employment,
        "home_ownership": home, "purpose": purpose, "loan_amount": loan, "term_months": term,
        "dti": dti, "credit_utilization": util, "credit_history_years": history,
        "delinquencies_2y": delinq, "inquiries_6m": inquiries, "open_accounts": accounts,
        C.TARGET: defaulted,
    })


def validate(df, require_target=True):
    """Data contract. Returns a list of problems (empty = valid)."""
    needed = ["application_id", *C.NUMERIC, *C.CATEGORICAL] + ([C.TARGET] if require_target else [])
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return [f"missing columns: {missing}"]
    if df.empty:
        return ["no rows"]
    problems = []
    if df["application_id"].duplicated().any():
        problems.append("duplicate application_id values")
    nulls = [c for c in C.NUMERIC + C.CATEGORICAL if df[c].isna().any()]
    if nulls:
        problems.append(f"missing values in {nulls}")
    for col, (lo, hi) in C.RANGES.items():
        vals = pd.to_numeric(df[col], errors="coerce")
        bad = int(((vals < lo) | (vals > hi) | vals.isna()).sum())
        if bad:
            problems.append(f"{col}: {bad} values outside [{lo}, {hi}]")
    for col, allowed in C.ALLOWED.items():
        unknown = set(df[col].dropna().unique()) - set(allowed)
        if unknown:
            problems.append(f"{col}: unexpected values {sorted(map(str, unknown))}")
    if require_target and set(df[C.TARGET].unique()) != {0, 1}:
        problems.append(f"{C.TARGET} must contain both 0 and 1")
    return problems


def fingerprint(df):
    """Short content hash, stored with each model version for data lineage."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes()).hexdigest()[:16]


def write_splits(out_dir="data"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    generate(20_000, seed=0).to_csv(out / "train.csv", index=False)
    generate(8_000, seed=1001).to_csv(out / "holdout.csv", index=False)      # fixed: every version judged on it
    generate(3_000, seed=2002, regime="downturn").to_csv(out / "downturn_sample.csv", index=False)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="data")
    out = write_splits(p.parse_args().out)
    for name in ["train", "holdout", "downturn_sample"]:
        df = pd.read_csv(out / f"{name}.csv")
        print(f"{name}: {len(df)} rows, default rate {df[C.TARGET].mean():.1%}, "
              f"{'valid' if not validate(df) else validate(df)}")
