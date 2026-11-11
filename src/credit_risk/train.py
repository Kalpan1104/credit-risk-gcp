"""
Train a new scorecard version and put it in the registry (NOT in production; promote.py decides that).

    python -m credit_risk.train                                  # local registry
    python -m credit_risk.train --registry gs://BUCKET/registry  # Cloud Storage registry
    python -m credit_risk.train --regime downturn                # retrain on post-shock data
"""
import argparse
import os
import subprocess
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from credit_risk import config as C
from credit_risk.data import fingerprint, generate, validate, write_splits
from credit_risk.model import coefficient_signs, evaluate, predict_pd, reference_profile, train
from credit_risk.registry import Registry

C_GRID = [0.05, 0.2, 1.0]


def git_sha():
    if os.getenv("GITHUB_SHA"):
        return os.environ["GITHUB_SHA"][:7]
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):   # git missing, or not a git checkout
        return "unknown"


def load_data(data_dir, regime):
    data_dir = Path(data_dir)
    if not (data_dir / "train.csv").exists():
        write_splits(data_dir)
    train_df = pd.read_csv(data_dir / "train.csv")
    if regime == "downturn":
        # what a lender would have after a shock: last year's book plus recent post-shock applications
        train_df = pd.concat([train_df, generate(12_000, seed=3003, regime="downturn")], ignore_index=True)
    holdout = pd.read_csv(data_dir / "holdout.csv")
    for name, df in [("train", train_df), ("holdout", holdout)]:
        problems = validate(df)
        if problems:
            raise SystemExit(f"{name} failed the data contract: {problems}")
    return train_df, holdout


def select_regularisation(train_df):
    fit, val = train_test_split(train_df, test_size=0.2, stratify=train_df[C.TARGET], random_state=0)
    scores = {c: roc_auc_score(val[C.TARGET], predict_pd(train(fit, c), val)) for c in C_GRID}
    return max(scores, key=scores.get), {str(k): round(v, 4) for k, v in scores.items()}


def model_card(version, metrics, info):
    rows = "\n".join(f"| {k} | {v} |" for k, v in metrics.items() if not isinstance(v, (dict, list)))
    signs = ", ".join(f"{k} {'↑' if v > 0 else '↓'}" for k, v in metrics["coefficient_signs"].items())
    return f"""# Model card: loan default scorecard {version}

**Purpose:** estimate the probability that a personal-loan applicant defaults, turn it into a 300-850
score, and recommend approve / review / decline. Synthetic data only; not for real lending decisions.

**Model:** standardised logistic regression (C={info['C']}), sklearn {info['sklearn_version']}.
Age is not an input; it is used only for the fairness check.

**Training data:** {info['n_train']} applications (regime: {info['regime']}, fingerprint `{info['train_fp']}`).
**Holdout:** {info['n_holdout']} applications (fingerprint `{info['holdout_fp']}`). Commit `{info['git_sha']}`.

## Holdout metrics
| metric | value |
|---|---|
{rows}

**Risk direction per feature (must match policy):** {signs}

## Limitations
- Synthetic applicants; real bureau data is messier and has delayed labels (defaults show up months later).
- Fairness is checked for age groups only, via the four-fifths rule on approvals.
"""


def main(registry_uri=None, data_dir="data", regime="normal"):
    registry = Registry(registry_uri or C.REGISTRY_URI)
    train_df, holdout = load_data(data_dir, regime)
    best_c, grid = select_regularisation(train_df)
    bundle = train(train_df, best_c)

    metrics = evaluate(bundle, holdout)
    metrics["coefficient_signs"] = coefficient_signs(bundle)
    metrics["eval_mode"] = "holdout"
    if regime == "downturn":
        metrics["eval_mode"] = "recent"
        metrics["recent_spec"] = C.RECENT_EVAL
        metrics["recent"] = evaluate(bundle, generate(**C.RECENT_EVAL))
    metrics["c_grid_val_auc"] = grid
    info = {"C": best_c, "sklearn_version": bundle["sklearn_version"], "n_train": len(train_df),
            "n_holdout": len(holdout), "regime": regime, "train_fp": fingerprint(train_df),
            "holdout_fp": fingerprint(holdout), "git_sha": git_sha()}
    metrics["lineage"] = info

    version = registry.next_version()
    bundle["version"] = version
    where = registry.save_version(version, bundle, metrics, reference_profile(bundle, train_df),
                                  model_card(version, {k: v for k, v in metrics.items() if k != "lineage"}, info))
    print(f"Registered {version} at {where}")
    print(f"  holdout: AUC {metrics['auc']}  KS {metrics['ks']}  calibration gap {metrics['calibration_gap']}  "
          f"AIR under_30 {metrics['air_under_30']}  AIR 62_plus {metrics['air_62_plus']}")
    if "recent" in metrics:
        r = metrics["recent"]
        print(f"  recent:  AUC {r['auc']}  Brier {r['brier']}  calibration gap {r['calibration_gap']}")
    if os.getenv("GITHUB_OUTPUT"):
        with open(os.environ["GITHUB_OUTPUT"], "a") as f:
            f.write(f"version={version}\n")
    return version, metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--registry", default=None)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--regime", choices=["normal", "downturn"], default="normal")
    a = p.parse_args()
    main(a.registry, a.data_dir, a.regime)
