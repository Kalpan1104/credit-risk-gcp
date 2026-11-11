import pytest

from credit_risk import config as C
from credit_risk.data import fingerprint, generate, validate


def test_reproducible_and_valid():
    a, b = generate(2000, seed=3), generate(2000, seed=3)
    assert a.equals(b) and validate(a) == []
    assert fingerprint(a) == fingerprint(b) != fingerprint(generate(2000, seed=4))


def test_default_rate_realistic():
    assert 0.06 < generate(10000, seed=1)[C.TARGET].mean() < 0.16


def test_downturn_is_riskier():
    normal, shock = generate(8000, seed=2), generate(8000, seed=2, regime="downturn")
    assert shock[C.TARGET].mean() > normal[C.TARGET].mean() + 0.1
    assert shock["dti"].mean() > normal["dti"].mean()


@pytest.mark.parametrize("mutate, expected", [
    (lambda d: d.drop(columns=["dti"]), "missing columns"),
    (lambda d: d.iloc[0:0], "no rows"),
    (lambda d: d.assign(dti=d["dti"].where(d.index != 0, 500)), "dti"),
    (lambda d: d.assign(purpose=d["purpose"].where(d.index != 0, "yacht")), "purpose"),
    (lambda d: d.assign(term_months=d["term_months"].where(d.index != 0, 48)), "term_months"),
    (lambda d: d.assign(application_id="same"), "duplicate"),
    (lambda d: d.assign(annual_income=d["annual_income"].where(d.index != 0, None)), "missing values"),
])
def test_contract(mutate, expected):
    assert any(expected in p for p in validate(mutate(generate(300, seed=5))))


def test_scoring_requests_need_no_label():
    assert validate(generate(10, seed=1).drop(columns=[C.TARGET]), require_target=False) == []
