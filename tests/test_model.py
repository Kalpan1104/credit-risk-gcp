import numpy as np
import pandas as pd
import pytest

from credit_risk import config as C
from credit_risk.model import (coefficient_signs, decide, evaluate, predict_pd, reason_codes,
                               reference_profile, to_score)


def test_holdout_quality(bundle, holdout_df):
    m = evaluate(bundle, holdout_df)
    assert m["auc"] >= C.MIN_AUC and m["ks"] >= C.MIN_KS
    assert m["calibration_gap"] <= C.MAX_CALIBRATION_GAP
    assert m["air_under_30"] >= C.MIN_ADVERSE_IMPACT_RATIO and m["air_62_plus"] >= C.MIN_ADVERSE_IMPACT_RATIO


def test_risk_moves_the_right_way(bundle):
    assert coefficient_signs(bundle) == {k: float(v) for k, v in C.EXPECTED_SIGNS.items()}


def test_age_is_not_a_model_input(bundle):
    assert not any("age" == n.split("__")[1] for n in bundle["feature_names"])


def test_score_scaling():
    base_pd = 1 / (1 + C.BASE_ODDS)                 # 50:1 good:bad -> base score
    assert to_score([base_pd])[0] == C.BASE_SCORE
    doubled = 1 / (1 + 2 * C.BASE_ODDS)             # twice the odds -> +PDO points
    assert to_score([doubled])[0] == C.BASE_SCORE + C.PDO
    s = to_score([1e-9, 0.01, 0.1, 0.5, 0.999999])
    assert list(s) == sorted(s, reverse=True) and s.min() >= C.SCORE_MIN and s.max() <= C.SCORE_MAX


@pytest.mark.parametrize("pd_value, expected", [(0.02, "approve"), (0.12, "review"), (0.4, "decline")])
def test_policy(pd_value, expected):
    assert decide(pd_value) == expected


def test_reason_codes_explain_a_risky_applicant(bundle, holdout_df):
    risky = holdout_df.head(1).copy()
    risky["dti"], risky["credit_utilization"], risky["delinquencies_2y"] = 45.0, 0.95, 3
    reasons = reason_codes(bundle, risky)[0]
    assert len(reasons) == 3
    assert {"Debt-to-income ratio is high", "High utilisation of existing credit",
            "Recent delinquencies on file"} & set(reasons)
    assert predict_pd(bundle, risky)[0] > C.DECLINE_ABOVE_PD


def test_reference_profile(bundle, train_df):
    prof = reference_profile(bundle, train_df)
    assert abs(sum(prof["score_share"]) - 1) < 1e-9
    assert set(prof["features"]) >= {"dti", "credit_utilization", "annual_income"}
    assert 0 < prof["approval_rate"] < 1


def test_unknown_category_does_not_crash(bundle, holdout_df):
    odd = holdout_df.head(2).copy()
    odd["purpose"] = "wedding"          # the API's contract would reject this; the model must not crash
    assert np.isfinite(predict_pd(bundle, pd.DataFrame(odd))).all()
