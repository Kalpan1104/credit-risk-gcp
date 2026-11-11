"""
Request handling shared by the Cloud Run functions and local runs. No web framework in here,
so it can be tested directly.
"""
import json
import time
import uuid
from datetime import datetime, timezone

import pandas as pd
import sklearn

from credit_risk import config as C
from credit_risk.data import validate
from credit_risk.model import decide, predict_pd, reason_codes, to_score
from credit_risk.monitoring import analyse, to_markdown
from credit_risk.storage import open_storage

MAX_BATCH = 500


def log(severity, message, **fields):
    """Structured log line; Cloud Logging turns it into a searchable entry with this severity."""
    print(json.dumps({"severity": severity, "message": message, **fields}, default=str), flush=True)


class NoChampion(Exception):
    pass


class ModelCache:
    """
    Keeps the champion model in memory and re-checks champion.json at most every `refresh` seconds,
    so promotions and rollbacks reach a running function without redeploying it.
    """

    def __init__(self, registry, refresh=C.MODEL_REFRESH_SECONDS, clock=time.monotonic):
        self.registry, self.refresh, self.clock = registry, refresh, clock
        self.bundle, self.version, self._checked = None, None, None

    def get(self):
        now = self.clock()
        if self._checked is None or now - self._checked >= self.refresh:
            self._checked = now
            try:
                pointer = self.registry.champion()
                if pointer and pointer["version"] != self.version:
                    bundle = self.registry.load_bundle(pointer["version"])
                    if bundle.get("sklearn_version") != sklearn.__version__:
                        log("WARNING", "sklearn version differs from training",
                            trained_with=bundle.get("sklearn_version"), running=sklearn.__version__)
                    self.bundle, self.version = bundle, pointer["version"]
                    log("INFO", "loaded champion model", version=self.version)
            except Exception as e:  # keep serving the model we have
                log("ERROR", "could not refresh champion", error=str(e), still_serving=self.version)
        if self.bundle is None:
            raise NoChampion("No champion model yet. Train and promote one first.")
        return self.bundle, self.version


def score_payload(payload, cache, sink):
    """Returns (http_status, body)."""
    if not isinstance(payload, dict) or not isinstance(payload.get("applicants"), list):
        return 400, {"error": 'expected JSON like {"applicants": [{...}, ...]}'}
    applicants = payload["applicants"]
    if not 1 <= len(applicants) <= MAX_BATCH:
        return 400, {"error": f"send between 1 and {MAX_BATCH} applicants"}
    df = pd.DataFrame(applicants)
    if "application_id" not in df.columns:
        df["application_id"] = [str(uuid.uuid4()) for _ in range(len(df))]
    df["application_id"] = df["application_id"].astype(str)
    problems = validate(df, require_target=False)
    if problems:
        return 422, {"error": "invalid applicants", "problems": problems}
    try:
        bundle, version = cache.get()
    except NoChampion as e:
        return 503, {"error": str(e)}

    pds = predict_pd(bundle, df)
    scores = to_score(pds)
    reasons = reason_codes(bundle, df)
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    results, rows = [], []
    for i, app in enumerate(df.to_dict("records")):
        decision = decide(float(pds[i]))
        rid = str(uuid.uuid4())
        results.append({"application_id": app["application_id"], "request_id": rid,
                        "probability_of_default": round(float(pds[i]), 4), "score": int(scores[i]),
                        "decision": decision, "reasons": reasons[i] if decision != "approve" else []})
        feats = {c: (float(app[c]) if c in C.NUMERIC else str(app[c])) for c in C.NUMERIC + C.CATEGORICAL}
        rows.append({"request_id": rid, "ts": now, "model_version": version, "pd": round(float(pds[i]), 6),
                     "score": int(scores[i]), "decision": decision, "features": feats, "reasons": reasons[i]})
    try:
        sink.write(rows)
    except Exception as e:  # never fail a credit decision because logging failed, but make it loud
        log("ERROR", "prediction logging failed", error=str(e), n=len(rows))
    return 200, {"model_version": version, "results": results}


def run_monitor(registry, sink, reports_uri, hours=C.MONITOR_WINDOW_HOURS):
    pointer = registry.champion()
    if not pointer:
        return {"status": "no_champion"}
    champion = pointer["version"]
    result = analyse(registry.profile(champion), sink.read_since(hours))
    result.update({"champion": champion, "window_hours": hours,
                   "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
    md = to_markdown(result, champion, hours)
    store = open_storage(reports_uri)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H%M")
    store.write_bytes(f"monitoring/{stamp}.json", json.dumps(result, indent=2).encode())
    store.write_bytes("monitoring/latest.md", md.encode())
    severity = "WARNING" if result.get("alerts") else "INFO"
    log(severity, "credit-risk monitoring", status=result["status"], alerts=result.get("alerts", []),
        champion=champion, n=result.get("n"), report=store.uri(f"monitoring/{stamp}.json"))
    return result
