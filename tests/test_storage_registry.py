import json

from credit_risk.registry import Registry
from credit_risk.storage import GCSStorage, LocalStorage, open_storage
from tests.fakes import FakeGCSClient


def test_open_storage_picks_backend(tmp_path):
    assert isinstance(open_storage(str(tmp_path)), LocalStorage)
    s = open_storage("gs://my-bucket/registry", client=FakeGCSClient())
    assert isinstance(s, GCSStorage) and s.uri("a.json") == "gs://my-bucket/registry/a.json"


def test_gcs_roundtrip_and_listing():
    client = FakeGCSClient()
    s = open_storage("gs://b/reg", client=client)
    s.write_bytes("versions/v1/m.json", b"{}")
    s.write_bytes("versions/v2/m.json", b"{}")
    open_storage("gs://b/other", client=client).write_bytes("x", b"1")
    assert s.list("versions/") == ["versions/v1/m.json", "versions/v2/m.json"]
    assert s.read_bytes("versions/v1/m.json") == b"{}" and s.exists("versions/v2/m.json")
    assert not s.exists("nope")


def _save(reg, version, bundle):
    reg.save_version(version, bundle, {"auc": 0.8}, {"p": 1}, "# card")


def test_registry_versions_and_champion_on_both_backends(tmp_path, bundle):
    for reg in (Registry(str(tmp_path / "reg")), Registry("gs://bkt/registry", client=FakeGCSClient())):
        assert reg.versions() == [] and reg.next_version() == "v1" and reg.champion() is None
        for v in ("v1", "v2", "v10"):
            _save(reg, v, bundle)
        assert reg.versions() == ["v1", "v2", "v10"] and reg.next_version() == "v11"
        reg.set_champion("v2", None, "test")
        reg.set_champion("v10", "v2", "test")
        assert reg.champion()["version"] == "v10" and reg.champion()["previous"] == "v2"
        assert len(reg.read_json("history.json")) == 2
        assert reg.load_bundle("v2")["sklearn_version"] == bundle["sklearn_version"]
        assert json.loads(json.dumps(reg.metrics("v1"))) == {"auc": 0.8}
