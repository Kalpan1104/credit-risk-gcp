"""
The credit scorecard: features, model, metrics, score scaling and reason codes.

Logistic regression is the industry default for credit decisions because every
decision can be explained, which lenders must do (adverse action notices).
"""
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_risk import config as C

# Engineered inputs. Age is deliberately NOT here.
MODEL_NUMERIC = ["log_income", "employment_years", "loan_to_income", "term_60", "dti", "credit_utilization",
                 "credit_history_years", "delinquencies_2y", "inquiries_6m", "open_accounts"]

REASONS = {
    "dti": "Debt-to-income ratio is high",
    "credit_utilization": "High utilisation of existing credit",
    "delinquencies_2y": "Recent delinquencies on file",
    "inquiries_6m": "Many recent credit inquiries",
    "log_income": "Income is low relative to similar applicants",
    "loan_to_income": "Requested amount is large relative to income",
    "credit_history_years": "Short credit history",
    "employment_years": "Short time in current employment",
    "term_60": "Longer loan term",
    "open_accounts": "Number of open accounts",
    "home_ownership": "Housing status",
    "purpose": "Loan purpose",
}


def engineer(df):
    out = pd.DataFrame(index=df.index)
    out["log_income"] = np.log(df["annual_income"].astype(float))
    out["loan_to_income"] = df["loan_amount"] / df["annual_income"]
    out["term_60"] = (df["term_months"] == 60).astype(float)
    for col in ["employment_years", "dti", "credit_utilization", "credit_history_years",
                "delinquencies_2y", "inquiries_6m", "open_accounts"]:
        out[col] = df[col].astype(float)
    for col in C.CATEGORICAL:
        out[col] = df[col].astype(str)
    return out


def build_pipeline(C_reg=0.5):
    pre = ColumnTransformer([
        ("num", StandardScaler(), MODEL_NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore",
                              categories=[C.ALLOWED[c] for c in C.CATEGORICAL]), C.CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=C_reg, max_iter=2000))])


def train(df, C_reg=0.5):
    X = engineer(df)
    pipe = build_pipeline(C_reg).fit(X, df[C.TARGET])
    transformed = pipe.named_steps["pre"].transform(X)
    transformed = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    return {
        "pipeline": pipe,
        "feature_names": list(pipe.named_steps["pre"].get_feature_names_out()),
        "feature_means": transformed.mean(axis=0).tolist(),
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "C": C_reg,
    }


def predict_pd(bundle, df):
    return bundle["pipeline"].predict_proba(engineer(df))[:, 1]


def to_score(pd_values):
    """Probability of default -> 300-850 points (higher = safer)."""
    p = np.clip(np.asarray(pd_values, dtype=float), 1e-6, 1 - 1e-6)
    factor = C.PDO / np.log(2)
    offset = C.BASE_SCORE - factor * np.log(C.BASE_ODDS)
    return np.clip(np.round(offset + factor * np.log((1 - p) / p)), C.SCORE_MIN, C.SCORE_MAX).astype(int)


def decide(pd_value):
    if pd_value < C.APPROVE_BELOW_PD:
        return "approve"
    return "decline" if pd_value > C.DECLINE_ABOVE_PD else "review"


def reason_codes(bundle, df, top=3):
    """Top factors pushing each applicant's risk up, relative to the average applicant."""
    pre = bundle["pipeline"].named_steps["pre"]
    coef = bundle["pipeline"].named_steps["clf"].coef_[0]
    X = pre.transform(engineer(df))
    X = X.toarray() if hasattr(X, "toarray") else X
    contrib = (X - np.asarray(bundle["feature_means"])) * coef
    names = bundle["feature_names"]
    out = []
    for row in contrib:
        order = np.argsort(-row)
        reasons = []
        for i in order:
            if row[i] <= 0 or len(reasons) == top:
                break
            base = names[i].split("__", 1)[1]
            key = next((c for c in C.CATEGORICAL if base.startswith(c + "_")), base)
            msg = REASONS.get(key, key)
            if msg not in reasons:
                reasons.append(msg)
        out.append(reasons)
    return out


# ---------- evaluation ----------

def ks_statistic(y, p):
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


AGE_GROUPS = {"under_30": (0, 30), "62_plus": (62, 200)}


def adverse_impact_ratios(df, pd_values):
    """
    Approval rate of each age group divided by the approval rate of applicants aged 30-61.
    1.0 = equal treatment; the "four-fifths rule" flags anything under 0.8.
    """
    approved = np.asarray(pd_values) < C.APPROVE_BELOW_PD
    age = df[C.PROTECTED].to_numpy()
    reference = approved[(age >= 30) & (age < 62)].mean()
    out = {}
    for name, (lo, hi) in AGE_GROUPS.items():
        mask = (age >= lo) & (age < hi)
        out[name] = round(float(approved[mask].mean() / reference), 4) if mask.any() and reference > 0 else None
    return out


def coefficient_signs(bundle):
    coef = bundle["pipeline"].named_steps["clf"].coef_[0]
    names = [n.split("__", 1)[1] for n in bundle["feature_names"]]
    lookup = dict(zip(names, coef))
    mapping = {"annual_income": "log_income"}
    return {feat: float(np.sign(lookup[mapping.get(feat, feat)])) for feat in C.EXPECTED_SIGNS}


def evaluate(bundle, df):
    y = df[C.TARGET].to_numpy()
    p = predict_pd(bundle, df)
    decisions = [decide(v) for v in p]
    return {
        "auc": round(float(roc_auc_score(y, p)), 4),
        "gini": round(2 * float(roc_auc_score(y, p)) - 1, 4),
        "ks": round(ks_statistic(y, p), 4),
        "brier": round(float(brier_score_loss(y, p)), 4),
        "mean_pd": round(float(p.mean()), 4),
        "default_rate": round(float(y.mean()), 4),
        "calibration_gap": round(abs(float(p.mean() - y.mean())), 4),
        "approval_rate": round(decisions.count("approve") / len(decisions), 4),
        "decline_rate": round(decisions.count("decline") / len(decisions), 4),
        **{f"air_{k}": v for k, v in adverse_impact_ratios(df, p).items()},
        "approved_default_rate": round(float(y[p < C.APPROVE_BELOW_PD].mean()), 4) if (p < C.APPROVE_BELOW_PD).any() else None,
    }


def _quantile_edges(values, bins=10):
    edges = np.unique(np.quantile(values, np.linspace(0, 1, bins + 1)))
    return [float(e) for e in edges]


def share_in_bins(values, edges):
    e = np.array(edges, dtype=float)
    e[0], e[-1] = -np.inf, np.inf
    counts = np.histogram(np.asarray(values, dtype=float), e)[0]
    return (counts / max(len(values), 1)).tolist()


def reference_profile(bundle, df):
    """Training-time distributions the monitoring job compares production traffic against."""
    p = predict_pd(bundle, df)
    scores = to_score(p)
    prof = {"score_edges": _quantile_edges(scores), "features": {},
            "approval_rate": float(np.mean(p < C.APPROVE_BELOW_PD)), "mean_pd": float(p.mean()), "n": len(df)}
    prof["score_share"] = share_in_bins(scores, prof["score_edges"])
    for col in ["annual_income", "dti", "credit_utilization", "inquiries_6m", "loan_amount", "credit_history_years"]:
        edges = _quantile_edges(df[col])
        prof["features"][col] = {"edges": edges, "share": share_in_bins(df[col], edges)}
    return prof
