from loralib.utils import JsonCache, cached


def test_json_cache_round_trips_nested_values(tmp_path):
    path = tmp_path / "cache.json"
    cache = JsonCache(path)
    cache["model"]["norms"]["layer"] = 1.25
    cache.save(discard=True)

    restored = JsonCache(path)

    assert restored["model"]["norms"]["layer"] == 1.25
    restored.clear()


def test_cached_initializes_plain_dict_and_caches_none():
    class Subject:
        def __init__(self):
            self._cache = {}
            self.calls = 0

        @cached("values")
        def lookup(self, key):
            self.calls += 1
            return None

    subject = Subject()

    assert subject.lookup("missing") is None
    assert subject.lookup("missing") is None
    assert subject.calls == 1
    assert subject._cache == {"values": {"missing": None}}
