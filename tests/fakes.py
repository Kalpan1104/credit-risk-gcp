"""In-memory stand-ins for the Cloud Storage and BigQuery clients (no credentials needed)."""
from datetime import datetime, timezone


class FakeBlob:
    def __init__(self, store, name):
        self.store, self.name = store, name

    def upload_from_string(self, data):
        self.store[self.name] = data if isinstance(data, bytes) else data.encode()

    def download_as_bytes(self):
        if self.name not in self.store:
            raise FileNotFoundError(self.name)
        return self.store[self.name]

    def exists(self):
        return self.name in self.store


class FakeBucket:
    def __init__(self, store):
        self.store = store

    def blob(self, name):
        return FakeBlob(self.store, name)


class FakeGCSClient:
    def __init__(self):
        self.buckets = {}

    def bucket(self, name):
        return FakeBucket(self.buckets.setdefault(name, {}))

    def list_blobs(self, bucket, prefix=""):
        return [FakeBlob(self.buckets.get(bucket, {}), n) for n in sorted(self.buckets.get(bucket, {})) if n.startswith(prefix)]


class _Row:
    def __init__(self, d):
        self.d = d

    def items(self):
        return self.d.items()


class _Job:
    def __init__(self, rows):
        self.rows = rows

    def result(self):
        return [_Row(r) for r in self.rows]


class FakeBigQueryClient:
    def __init__(self):
        self.rows, self.queries = [], []

    def insert_rows_json(self, table, rows):
        self.table = table
        for r in rows:
            self.rows.append({**r, "ts": datetime.fromisoformat(r["ts"]).astimezone(timezone.utc)})
        return []

    def query(self, sql):
        self.queries.append(sql)
        return _Job(self.rows)
