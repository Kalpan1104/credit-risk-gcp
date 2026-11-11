"""
Where scored applications are logged, and where monitoring reads them back.

    jsonl:logs/predictions.jsonl            local file (development, tests)
    bigquery:PROJECT.DATASET.TABLE          BigQuery (production)
"""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path


def _now():
    return datetime.now(timezone.utc)


class JsonlSink:
    def __init__(self, path):
        self.path = Path(path)

    def write(self, rows):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def read_since(self, hours):
        if not self.path.exists():
            return []
        cutoff = _now() - timedelta(hours=hours)
        rows = [json.loads(line) for line in self.path.read_text().splitlines() if line.strip()]
        return [r for r in rows if datetime.fromisoformat(r["ts"]) >= cutoff]


class BigQuerySink:
    def __init__(self, table, client=None):
        if client is None:
            from google.cloud import bigquery
            client = bigquery.Client()
        self.client = client
        self.table = table

    def write(self, rows):
        payload = [{**r, "features": json.dumps(r["features"]), "reasons": r.get("reasons", [])} for r in rows]
        errors = self.client.insert_rows_json(self.table, payload)
        if errors:
            raise RuntimeError(f"BigQuery insert failed: {errors[:3]}")

    def read_since(self, hours):
        hours = int(hours)  # integer only, so it is safe to put in the SQL text
        sql = (f"SELECT request_id, ts, model_version, pd, score, decision, features "
               f"FROM `{self.table}` WHERE ts >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {hours} HOUR)")
        out = []
        for row in self.client.query(sql).result():
            r = dict(row.items())
            r["features"] = json.loads(r["features"]) if isinstance(r["features"], str) else r["features"]
            r["ts"] = r["ts"].isoformat() if hasattr(r["ts"], "isoformat") else r["ts"]
            out.append(r)
        return out


def open_sink(spec, client=None):
    kind, _, target = spec.partition(":")
    if kind == "jsonl":
        return JsonlSink(target)
    if kind == "bigquery":
        return BigQuerySink(target, client=client)
    raise ValueError(f"unknown sink '{spec}' (use jsonl:PATH or bigquery:PROJECT.DATASET.TABLE)")
