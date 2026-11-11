import json

import pytest

from credit_risk import config as C
from credit_risk.data import generate
from credit_risk.registry import Registry
from credit_risk.service import ModelCache, NoChampion, run_monitor, score_payload
from credit_risk.sinks import BigQuerySink, JsonlSink, open_sink
from tests.fakes import FakeBigQueryClient


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


@pytest.fixture
def registry(tmp_path, bundle, train_df):
    from credit_risk.model import reference_profile
    reg = Registry(str(tmp_path / "reg"))
    for v in ("v1", "v2"):
        reg.save_version(v, {**bundle, "version": v}, {"auc": 0.77}, reference_profile(bundle, train_df), "#")
    reg.set_champion("v1", None, "test")
    return reg


def test_cache_picks_up_promotions_after_refresh(registry):
    clock = Clock()
    cache = ModelCache(registry, refresh=60, clock=clock)
    assert cache.get()[1] == "v1"
    registry.set_champion("v2", "v1", "promoted")
    clock.t = 30
    assert cache.get()[1] == "v1"          # not re-checked yet
    clock.t = 61
    assert cache.get()[1] == "v2"


def test_cache_keeps_serving_when_registry_breaks(registry, monkeypatch):
    clock = Clock()
    cache = ModelCache(registry, refresh=1, clock=clock)
    cache.get()
    monkeypatch.setattr(registry, "champion", lambda: (_ for _ in ()).throw(OSError("bucket unreachable")))
    clock.t = 5
    assert cache.get()[1] == "v1"


def test_no_champion(tmp_path):
    with pytest.raises(NoChampion):
        ModelCache(Registry(str(tmp_path / "empty"))).get()


def test_score_payload(registry, tmp_path, applicants):
    sink = JsonlSink(tmp_path / "p.jsonl")
    status, body = score_payload({"applicants": applicants}, ModelCache(registry), sink)
    assert status == 200 and body["model_version"] == "v1" and len(body["results"]) == 5
    r = body["results"][0]
    assert C.SCORE_MIN <= r["score"] <= C.SCORE_MAX and r["decision"] in {"approve", "review", "decline"}
    assert all(res["reasons"] for res in body["results"] if res["decision"] != "approve")
    logged = [json.loads(line) for line in (tmp_path / "p.jsonl").read_text().splitlines()]
    assert len(logged) == 5 and "dti" in logged[0]["features"]


@pytest.mark.parametrize("payload, status", [
    (None, 400), ({"applicants": []}, 400), ({"applicants": "x"}, 400),
    ({"applicants": [{"age": 30}]}, 422),
])
def test_bad_requests(registry, tmp_path, payload, status):
    assert score_payload(payload, ModelCache(registry), JsonlSink(tmp_path / "p.jsonl"))[0] == status


def test_logging_failure_does_not_block_decision(registry, applicants):
    class Broken:
        def write(self, rows):
            raise RuntimeError("bigquery down")
    assert score_payload({"applicants": applicants}, ModelCache(registry), Broken())[0] == 200


def test_monitor_flags_downturn_and_writes_report(registry, tmp_path):
    sink = JsonlSink(tmp_path / "p.jsonl")
    cache = ModelCache(registry)
    for regime, expected in (("normal", "ok"), ("downturn", "alert")):
        (tmp_path / "p.jsonl").unlink(missing_ok=True)
        apps = generate(300, seed=11, regime=regime).drop(columns=[C.TARGET]).to_dict("records")
        score_payload({"applicants": apps}, cache, sink)
        result = run_monitor(registry, sink, str(tmp_path / "reports"), hours=1)
        assert result["status"] == expected
    assert "approval_rate_shift" in result["alerts"]
    assert "Action needed" in (tmp_path / "reports" / "monitoring" / "latest.md").read_text()


def test_bigquery_sink_roundtrip():
    client = FakeBigQueryClient()
    sink = open_sink("bigquery:proj.ds.predictions", client=client)
    assert isinstance(sink, BigQuerySink)
    sink.write([{"request_id": "r1", "ts": "2026-01-01T00:00:00+00:00", "model_version": "v1", "pd": 0.1,
                 "score": 600, "decision": "review", "features": {"dti": 20.0}, "reasons": ["x"]}])
    assert client.table == "proj.ds.predictions" and isinstance(client.rows[0]["features"], str)
    rows = sink.read_since(24)
    assert rows[0]["features"] == {"dti": 20.0} and "INTERVAL 24 HOUR" in client.queries[0]


def test_unknown_sink():
    with pytest.raises(ValueError):
        open_sink("kafka:topic")
