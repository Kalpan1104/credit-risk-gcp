"""
A model registry on top of storage (a local folder or gs://bucket/registry).

    versions/v1/model.joblib          the bundle (pipeline + metadata)
    versions/v1/metrics.json          holdout metrics + gate inputs
    versions/v1/reference_profile.json
    versions/v1/model_card.md
    champion.json                     {"version": "v3", "previous": "v2", ...}  <- what production serves

Production reads only champion.json, so promoting or rolling back is one small
write, with no redeploy of the function.
"""
import io
import json
from datetime import datetime, timezone

import joblib

from credit_risk.storage import open_storage


class Registry:
    def __init__(self, uri, client=None):
        self.store = open_storage(uri, client=client)
        self.uri = uri

    # ----- versions -----
    def versions(self):
        names = {k.split("/")[1] for k in self.store.list("versions/") if k.count("/") >= 2}
        return sorted(names, key=lambda v: int(v[1:]))

    def next_version(self):
        existing = self.versions()
        return f"v{int(existing[-1][1:]) + 1 if existing else 1}"

    def save_version(self, version, bundle, metrics, profile, card):
        buf = io.BytesIO()
        joblib.dump(bundle, buf)
        base = f"versions/{version}"
        self.store.write_bytes(f"{base}/model.joblib", buf.getvalue())
        self.write_json(f"{base}/metrics.json", metrics)
        self.write_json(f"{base}/reference_profile.json", profile)
        self.store.write_bytes(f"{base}/model_card.md", card.encode())
        return self.store.uri(base)

    def load_bundle(self, version):
        return joblib.load(io.BytesIO(self.store.read_bytes(f"versions/{version}/model.joblib")))

    def metrics(self, version):
        return self.read_json(f"versions/{version}/metrics.json")

    def profile(self, version):
        return self.read_json(f"versions/{version}/reference_profile.json")

    # ----- champion pointer -----
    def champion(self):
        return self.read_json("champion.json") if self.store.exists("champion.json") else None

    def set_champion(self, version, previous, reason):
        pointer = {"version": version, "previous": previous, "reason": reason,
                   "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        self.write_json("champion.json", pointer)
        history = self.read_json("history.json") if self.store.exists("history.json") else []
        history.append(pointer)
        self.write_json("history.json", history)
        return pointer

    # ----- helpers -----
    def write_json(self, key, obj):
        self.store.write_bytes(key, json.dumps(obj, indent=2, default=str).encode())

    def read_json(self, key):
        return json.loads(self.store.read_bytes(key))
