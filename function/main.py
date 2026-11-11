"""
Cloud Run functions entry points (Python Functions Framework).

  score    POST {"applicants": [...]} -> PD, 300-850 score, decision, reason codes
           GET  -> which model version is being served
  monitor  POST (called daily by Cloud Scheduler) -> drift report written to Cloud Storage

Deployed with --entry-point=score / --entry-point=monitor. Configuration comes from
environment variables set at deploy time (REGISTRY_URI, PREDICTION_SINK, REPORTS_URI).
"""
import json

from credit_risk import config as C
from credit_risk.registry import Registry
from credit_risk.service import ModelCache, NoChampion, log, run_monitor, score_payload
from credit_risk.sinks import open_sink

_state = {}  # reused across requests on a warm instance


def _deps():
    if not _state:
        registry = Registry(C.REGISTRY_URI)
        _state.update(registry=registry, cache=ModelCache(registry), sink=open_sink(C.SINK))
    return _state


def _json(body, status=200):
    return json.dumps(body), status, {"Content-Type": "application/json"}


def score(request):
    deps = _deps()
    if request.method == "GET":
        try:
            _, version = deps["cache"].get()
            return _json({"status": "ok", "model_version": version})
        except NoChampion as e:
            return _json({"status": "no model", "detail": str(e)}, 503)
    if request.method != "POST":
        return _json({"error": "use GET or POST"}, 405)
    status, body = score_payload(request.get_json(silent=True), deps["cache"], deps["sink"])
    if status >= 400 and status != 503:
        log("WARNING", "rejected scoring request", status=status, error=body.get("error"))
    return _json(body, status)


def monitor(request):
    deps = _deps()
    hours = request.args.get("hours", type=int) or C.MONITOR_WINDOW_HOURS
    return _json(run_monitor(deps["registry"], deps["sink"], C.REPORTS_URI, hours))
