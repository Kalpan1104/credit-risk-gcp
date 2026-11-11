"""Every threshold and name in one place. Environment variables override the cloud settings."""
import os

# ---- where things live (local paths or gs:// URIs) ----
REGISTRY_URI = os.getenv("REGISTRY_URI", "local_registry")          # e.g. gs://my-bucket/registry
SINK = os.getenv("PREDICTION_SINK", "jsonl:logs/predictions.jsonl")  # or bigquery:project.dataset.table
REPORTS_URI = os.getenv("REPORTS_URI", "reports")                    # e.g. gs://my-bucket/reports
MODEL_REFRESH_SECONDS = int(os.getenv("MODEL_REFRESH_SECONDS", "60"))

# ---- applicant schema ----
NUMERIC = ["age", "annual_income", "employment_years", "loan_amount", "term_months", "dti",
           "credit_utilization", "credit_history_years", "delinquencies_2y", "inquiries_6m", "open_accounts"]
CATEGORICAL = ["home_ownership", "purpose"]
ALLOWED = {
    "home_ownership": ["rent", "mortgage", "own"],
    "purpose": ["debt_consolidation", "credit_card", "home_improvement", "car", "medical", "other"],
    "term_months": [36, 60],
}
RANGES = {
    "age": (18, 100), "annual_income": (5_000, 2_000_000), "employment_years": (0, 60),
    "loan_amount": (500, 100_000), "dti": (0, 80), "credit_utilization": (0, 1.5),
    "credit_history_years": (0, 70), "delinquencies_2y": (0, 30), "inquiries_6m": (0, 30),
    "open_accounts": (0, 80),
}
TARGET = "defaulted"
# Age is checked for fairness but never used as a model input
PROTECTED = "age"

# ---- scorecard scaling: 600 points at 50:1 good:bad odds, 20 points to double the odds ----
BASE_SCORE, BASE_ODDS, PDO = 600, 50, 20
SCORE_MIN, SCORE_MAX = 300, 850

# ---- credit policy ----
APPROVE_BELOW_PD = 0.08
DECLINE_ABOVE_PD = 0.20           # in between -> manual review

# ---- promotion gate ----
MIN_AUC = 0.72
MIN_KS = 0.30
MAX_CALIBRATION_GAP = 0.02        # |mean predicted PD - observed default rate| on holdout
MIN_ADVERSE_IMPACT_RATIO = 0.80   # "four-fifths rule": each age group's approval rate vs ages 30-61
# "Better" = lower Brier score (captures both ranking and calibration) on the evaluation set
MIN_BRIER_IMPROVEMENT = 0.0005
# After a shock, models are compared on recent (out-of-time) applications instead of the old holdout
RECENT_EVAL = {"n": 5000, "seed": 4004, "regime": "downturn"}
# Business logic: risk must move in this direction as each feature grows (model-risk sanity check)
EXPECTED_SIGNS = {"dti": +1, "credit_utilization": +1, "delinquencies_2y": +1, "inquiries_6m": +1,
                  "annual_income": -1, "credit_history_years": -1, "employment_years": -1}

# ---- monitoring ----
PSI_WARN, PSI_ALERT = 0.10, 0.25
APPROVAL_RATE_SHIFT_ALERT = 0.10  # absolute change vs training-time approval rate
MONITOR_WINDOW_HOURS = int(os.getenv("MONITOR_WINDOW_HOURS", "24"))
MIN_MONITOR_ROWS = 50
