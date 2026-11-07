import threading

import numpy as np
import pytest

from cacheguard.embeddings import normalize_rows
from cacheguard.partitions import partition_key
from cacheguard.policy import GlobalThreshold
from cacheguard.store import CacheStore


def unit(*values):
    return normalize_rows(np.array([values], dtype=np.float32))[0]


def make_store(clock=None, **kwargs):
    kwargs.setdefault("policy", GlobalThreshold(0.9))
    if clock is not None:
        kwargs["clock"] = clock
    return CacheStore(dim=2, **kwargs)


# --- lookup ---------------------------------------------------------------------

def test_empty_partition_misses():
    lookup = make_store().lookup("p", unit(1, 0))
    assert not lookup.hit and lookup.similarity is None and lookup.threshold is None


def test_hit_and_near_miss_report_nearest_query_and_threshold():
    store = make_store()
    store.add("p", "how to cancel an order", unit(1, 0), {"a": 1})
    hit = store.lookup("p", unit(1, 0.2))
    assert hit.hit and hit.entry.response == {"a": 1} and hit.threshold == 0.9
    miss = store.lookup("p", unit(1, 1))
    assert not miss.hit
    assert miss.nearest_query == "how to cancel an order"
    assert miss.similarity == pytest.approx(0.7071, abs=1e-3)


def test_partitions_never_share_answers():
    store = make_store()
    store.add("tenant-a", "q", unit(1, 0), "secret answer for A")
    assert not store.lookup("tenant-b", unit(1, 0)).hit
    assert store.lookup("tenant-a", unit(1, 0)).hit


def test_hits_are_counted_on_the_entry():
    store = make_store()
    entry = store.add("p", "q", unit(1, 0), "a")
    store.lookup("p", unit(1, 0))
    store.lookup("p", unit(1, 0))
    assert store.get(entry.entry_id).hit_count == 2


def test_rejects_wrong_dimension_and_unserializable_response():
    store = make_store()
    with pytest.raises(ValueError):
        store.add("p", "q", np.ones(3, dtype=np.float32), "a")
    with pytest.raises(TypeError):
        store.add("p", "q", unit(1, 0), object())
    assert len(store) == 0


# --- expiry, eviction, invalidation -----------------------------------------------

def test_expired_entries_stop_matching_and_are_removed(clock):
    store = make_store(clock, default_ttl_seconds=60)
    store.add("p", "old", unit(1, 0), "a")
    store.add("p", "forever", unit(0, 1), "b", ttl_seconds=10**9)
    clock.advance(61)
    assert not store.lookup("p", unit(1, 0)).hit
    assert len(store) == 1
    assert store.stats()["expirations"] == 1


def test_expired_nearest_entry_falls_back_to_next_valid_one(clock):
    store = make_store(clock, policy=GlobalThreshold(0.5))
    store.add("p", "expires", unit(1, 0), "a", ttl_seconds=5)
    store.add("p", "stays", unit(1, 0.3), "b")
    clock.advance(10)
    lookup = store.lookup("p", unit(1, 0))
    assert lookup.hit and lookup.entry.query == "stays"


def test_purge_expired(clock):
    store = make_store(clock, default_ttl_seconds=5)
    for i in range(3):
        store.add("p", f"q{i}", unit(1, i), i)
    clock.advance(6)
    assert store.purge_expired() == 3 and len(store) == 0


def test_lru_eviction_keeps_recently_used_entries():
    store = make_store(max_entries=2)
    first = store.add("p", "first", unit(1, 0), 1)
    second = store.add("p", "second", unit(0, 1), 2)
    store.lookup("p", unit(1, 0))  # touch "first"
    store.add("p", "third", unit(-1, 0), 3)
    assert store.get(first.entry_id) is not None
    assert store.get(second.entry_id) is None
    assert store.stats()["evictions"] == 1


def test_invalidate_by_tag_entry_and_partition():
    store = make_store()
    a = store.add("p1", "a", unit(1, 0), 1, tags=["source:7", "namespace:x"])
    store.add("p1", "b", unit(0, 1), 2, tags=["source:8"])
    store.add("p2", "c", unit(1, 1), 3, tags=["source:7"])
    assert store.invalidate(tag="source:7") == 2
    assert store.get(a.entry_id) is None
    assert store.invalidate(tag="source:7") == 0
    assert store.invalidate(partition="p1") == 1
    assert len(store) == 0
    assert store.invalidate(entry_id=999) == 0
    with pytest.raises(ValueError):
        store.invalidate(tag="x", partition="p")


def test_clear_resets_entries_and_counters():
    store = make_store()
    store.add("p", "q", unit(1, 0), 1)
    store.lookup("p", unit(1, 0))
    store.clear()
    stats = store.stats()
    assert stats["total_entries"] == 0 and stats["hits"] == 0 and stats["partitions"] == {}


def test_stats_are_broken_down_by_partition():
    store = make_store()
    store.add("a", "q", unit(1, 0), 1)
    store.lookup("a", unit(1, 0))
    store.lookup("b", unit(1, 0))
    stats = store.stats()
    assert stats["hits"] == 1 and stats["misses"] == 1
    assert stats["partitions"]["a"]["entries"] == 1 and stats["partitions"]["a"]["hits"] == 1
    assert stats["partitions"]["b"]["misses"] == 1
    assert stats["policy"] == {"type": "global", "threshold": 0.9}


# --- persistence ------------------------------------------------------------------

def test_restart_restores_entries_hits_and_ids(tmp_path, clock):
    db = tmp_path / "cache.sqlite3"
    store = make_store(clock, db_path=db)
    kept = store.add("p", "kept", unit(1, 0), {"answer": "x"}, tags=["source:1"])
    store.add("p", "short-lived", unit(0, 1), "y", ttl_seconds=5)
    store.lookup("p", unit(1, 0))
    store.close()

    clock.advance(10)
    restored = make_store(clock, db_path=db)
    assert len(restored) == 1
    lookup = restored.lookup("p", unit(1, 0))
    assert lookup.hit and lookup.entry.response == {"answer": "x"}
    assert restored.get(kept.entry_id).hit_count == 2
    assert restored.invalidate(tag="source:1") == 1
    assert restored.add("p", "new", unit(1, 1), 1).entry_id > kept.entry_id


def test_restart_applies_lru_order_and_capacity(tmp_path, clock):
    db = tmp_path / "cache.sqlite3"
    store = make_store(clock, db_path=db)
    old = store.add("p", "old", unit(1, 0), 1)
    clock.advance(1)
    store.add("p", "new", unit(0, 1), 2)
    clock.advance(1)
    store.lookup("p", unit(1, 0))  # "old" becomes most recently used
    store.close()

    restored = make_store(clock, db_path=db, max_entries=1)
    assert [e.query for e in (restored.get(old.entry_id),) if e] == ["old"]
    assert len(restored) == 1


# --- concurrency and keys ------------------------------------------------------------

def test_concurrent_lookups_and_adds_are_consistent():
    store = CacheStore(dim=8, policy=GlobalThreshold(0.99), max_entries=300)
    errors = []

    def worker(seed):
        rng = np.random.default_rng(seed)
        try:
            for _ in range(200):
                v = rng.normal(size=8).astype(np.float32)
                partition = f"p{seed % 2}"
                if not store.lookup(partition, v).hit:
                    store.add(partition, "q", v, seed, tags=[f"t{seed}"])
                if rng.random() < 0.05:
                    store.invalidate(tag=f"t{seed}")
        except Exception as exc:  # pragma: no cover - surfaced below
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(s,)) for s in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    stats = store.stats()
    assert stats["hits"] + stats["misses"] == 800
    assert len(store) <= 300
    assert sum(p["entries"] for p in stats["partitions"].values()) == len(store)


def test_partition_key_is_exact_and_order_independent():
    assert partition_key("acme", "groq") == "acme|groq"
    k1 = partition_key("acme", "groq", model="m", temperature=0)
    assert k1 == partition_key("acme", "groq", temperature=0, model="m")
    assert k1 != partition_key("acme", "groq", model="m", temperature=0.7)
    assert k1 != partition_key("other", "groq", model="m", temperature=0)
