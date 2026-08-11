from __future__ import annotations

from cutriton.cache import JitDecisionCache


def test_cache_round_trip_and_negative_decisions(tmp_path):
    cache = JitDecisionCache(tmp_path, max_bytes=10_000_000)
    cache.put("shape-a", "enabled", {"triton_ms": 1.0})
    cache.put("shape-b", "failed", {"reason": "compile"})
    assert cache.get("shape-a").payload["triton_ms"] == 1.0
    assert cache.get("shape-b").status == "failed"
    assert cache.count() == 2


def test_corrupt_database_is_recovered(tmp_path):
    path = tmp_path / "decisions.sqlite3"
    path.write_bytes(b"not sqlite")
    cache = JitDecisionCache(tmp_path, max_bytes=10_000_000)
    assert cache.count() == 0
    assert list(tmp_path.glob("decisions.corrupt-*"))
