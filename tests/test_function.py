"""The real Cloud Run function entry points, called with Flask requests like the Functions Framework does."""
import importlib
import json
import sys
from pathlib import Path

import pytest

flask = pytest.importorskip("flask")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "function"))

from credit_risk import config as C
from credit_risk.model import reference_profile
from credit_risk.registry import Registry


@pytest.fixture
def fn(tmp_path, bundle, train_df, monkeypatch):
    reg = Registry(str(tmp_path / "reg"))
    reg.save_version("v1", {**bundle, "version": "v1"}, {}, reference_profile(bundle, train_df), "#")
    reg.set_champion("v1", None, "test")
    monkeypatch.setattr(C, "REGISTRY_URI", str(tmp_path / "reg"))
    monkeypatch.setattr(C, "SINK", f"jsonl:{tmp_path / 'p.jsonl'}")
    monkeypatch.setattr(C, "REPORTS_URI", str(tmp_path / "reports"))
    main = importlib.import_module("main")
    main._state.clear()
    return main


def call(handler, method="POST", body=None, path="/"):
    app = flask.Flask("test")
    with app.test_request_context(path, method=method, json=body):
        text, status, headers = handler(flask.request)
    assert headers["Content-Type"] == "application/json"
    return status, json.loads(text)


def test_get_reports_version(fn):
    assert call(fn.score, "GET") == (200, {"status": "ok", "model_version": "v1"})


def test_post_scores(fn, applicants):
    status, body = call(fn.score, body={"applicants": applicants})
    assert status == 200 and len(body["results"]) == len(applicants)


def test_wrong_method(fn):
    assert call(fn.score, "PUT")[0] == 405


def test_monitor_endpoint(fn, applicants):
    call(fn.score, body={"applicants": applicants})
    status, body = call(fn.monitor, path="/?hours=2")
    assert status == 200 and body["window_hours"] == 2 and body["status"] == "insufficient_data"
