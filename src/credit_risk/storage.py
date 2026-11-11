"""
One tiny interface over a local folder or a Cloud Storage bucket, so the same
registry/report code runs on a laptop, in GitHub Actions and inside a Cloud Run function.

    open_storage("local_registry")          -> LocalStorage
    open_storage("gs://my-bucket/registry") -> GCSStorage
"""
from pathlib import Path


class LocalStorage:
    def __init__(self, root):
        self.root = Path(root)

    def write_bytes(self, key, data):
        path = self.root / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def read_bytes(self, key):
        return (self.root / key).read_bytes()

    def exists(self, key):
        return (self.root / key).exists()

    def list(self, prefix=""):
        base = self.root / prefix
        if not base.exists():
            return []
        return sorted(str(p.relative_to(self.root)) for p in base.rglob("*") if p.is_file())

    def uri(self, key):
        return str(self.root / key)


class GCSStorage:
    def __init__(self, bucket, prefix="", client=None):
        if client is None:
            from google.cloud import storage
            client = storage.Client()
        self.client = client
        self.bucket = client.bucket(bucket)
        self.bucket_name = bucket
        self.prefix = prefix.strip("/")

    def _key(self, key):
        return f"{self.prefix}/{key}" if self.prefix else key

    def write_bytes(self, key, data):
        self.bucket.blob(self._key(key)).upload_from_string(data)

    def read_bytes(self, key):
        return self.bucket.blob(self._key(key)).download_as_bytes()

    def exists(self, key):
        return self.bucket.blob(self._key(key)).exists()

    def list(self, prefix=""):
        full = self._key(prefix)
        cut = len(self.prefix) + 1 if self.prefix else 0
        return sorted(b.name[cut:] for b in self.client.list_blobs(self.bucket_name, prefix=full))

    def uri(self, key):
        return f"gs://{self.bucket_name}/{self._key(key)}"


def open_storage(uri, client=None):
    if uri.startswith("gs://"):
        bucket, _, prefix = uri[len("gs://"):].partition("/")
        return GCSStorage(bucket, prefix, client=client)
    return LocalStorage(uri)
