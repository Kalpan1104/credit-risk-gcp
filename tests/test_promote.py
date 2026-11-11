import pytest

from credit_risk import config as C
from credit_risk import promote as P
from credit_risk import train as T
from credit_risk.registry import Registry

SIGNS = {k: float(v) for k, v in C.EXPECTED_SIGNS.items()}
GOOD = {"auc": 0.77, "ks": 0.39, "calibration_gap": 0.005, "brier": 0.082,
        "air_under_30": 0.87, "air_62_plus": 1.1, "coefficient_signs": SIGNS}


def with_(**kw):
    return {**GOOD, **kw}


def test_first_model_passes():
    assert P.gate(GOOD, GOOD) == (True, [])


@pytest.mark.parametrize("change, expected", [
    ({"auc": 0.65}, "AUC"), ({"ks": 0.2}, "KS"),
    ({"air_under_30": 0.7}, "under_30"), ({"air_62_plus": 0.75}, "62_plus"),
    ({"coefficient_signs": {**SIGNS, "dti": -1.0}}, "wrong way with dti"),
])
def test_holdout_checks(change, expected):
    ok, reasons = P.gate(with_(**change), GOOD)
    assert not ok and any(expected in r for r in reasons)


def test_calibration_checked_on_evaluation_set():
    ok, reasons = P.gate(GOOD, with_(calibration_gap=0.05))
    assert not ok and "calibration" in reasons[0]


def test_must_beat_champion_on_brier():
    assert not P.gate(GOOD, with_(brier=0.0818), champ_eval=with_(brier=0.082))[0]
    assert P.gate(GOOD, with_(brier=0.081), champ_eval=with_(brier=0.082))[0]


def test_lifecycle_with_shock_and_rollback(tmp_path, data_dir):
    reg_uri = str(tmp_path / "registry")
    reg = Registry(reg_uri)
    T.main(reg_uri, data_dir)
    assert P.promote(reg) and reg.champion()["version"] == "v1"
    T.main(reg_uri, data_dir)                                  # same data -> no improvement
    assert not P.promote(reg) and reg.champion()["version"] == "v1"
    T.main(reg_uri, data_dir, regime="downturn")               # retrained after a shock
    assert reg.metrics("v3")["eval_mode"] == "recent"
    assert P.promote(reg) and reg.champion() == {**reg.champion(), "version": "v3", "previous": "v1"}
    P.rollback(reg)
    assert reg.champion()["version"] == "v1"
    with pytest.raises(SystemExit):
        P.promote(Registry(str(tmp_path / "empty")))
