"""Production cache store.

Compared with the baseline ``SemanticCache`` (one Python list, linear scan):

* **Partitions.** Entries live in separate indexes keyed by partition (e.g.
  tenant + model + system prompt). A lookup never crosses partitions, so one
  tenant can never receive another tenant's cached answer.
* **FAISS lookup.** Each partition is an exact inner-product index with ID
  mapping, which supports deletion (needed for eviction and invalidation).
* **Expiry and eviction.** Optional TTL per entry; least-recently-used eviction
  once ``max_entries`` is reached.
* **Invalidation by tag.** Entries carry tags such as ``source:123``; when a
  source changes, every answer built from it can be dropped at once.
* **Persistence.** Entries are written through to SQLite and the indexes are
  rebuilt from it on startup, so a restart keeps the cache warm.

All public methods are thread-safe (one lock; FAISS search is fast enough
that finer-grained locking is not worth the complexity at this scale).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import faiss
import numpy as np

from cacheguard.embeddings import normalize_rows
from cacheguard.policy import GlobalThreshold, ThresholdPolicy

_SEARCH_DEPTH = 8  # candidates fetched per lookup, so lazily-expired entries can be skipped


@dataclass
class StoredEntry:
    entry_id: int
    partition: str
    query: str
    response: Any  # JSON-serializable
    cluster: int | None
    tags: tuple[str, ...]
    created_at: float
    expires_at: float | None
    last_hit_at: float | None = None
    hit_count: int = 0

    def expired(self, now: float) -> bool:
        return self.expires_at is not None and now >= self.expires_at


@dataclass(frozen=True)
class StoreLookup:
    hit: bool
    entry: StoredEntry | None  # the matched entry on a hit, else None
    similarity: float | None  # nearest valid entry in the partition, hit or not
    threshold: float | None  # threshold that was applied to the nearest entry
    nearest_query: str | None = None  # for explaining near-misses


@dataclass
class _Counters:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    invalidations: int = 0
    decisions: dict[str, int] = field(default_factory=dict)  # why lookups ended as they did

    def merge(self, other: _Counters) -> None:
        for name in ("hits", "misses", "evictions", "expirations", "invalidations"):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        for reason, count in other.decisions.items():
            self.decisions[reason] = self.decisions.get(reason, 0) + count

    def as_dict(self) -> dict:
        lookups = self.hits + self.misses
        return {**vars(self), "decisions": dict(self.decisions), "hit_rate": self.hits / lookups if lookups else 0.0}


@dataclass
class _Partition:
    index: faiss.IndexIDMap2
    ids: set[int] = field(default_factory=set)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    entry_id     INTEGER PRIMARY KEY,
    partition    TEXT    NOT NULL,
    query        TEXT    NOT NULL,
    embedding    BLOB    NOT NULL,
    response     TEXT    NOT NULL,
    cluster      INTEGER,
    tags         TEXT    NOT NULL,
    created_at   REAL    NOT NULL,
    expires_at   REAL,
    last_hit_at  REAL,
    hit_count    INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS entries_partition ON entries (partition);
"""


class CacheStore:
    def __init__(
        self,
        dim: int,
        policy: ThresholdPolicy | None = None,
        db_path: str | Path = ":memory:",
        default_ttl_seconds: float | None = None,
        max_entries: int | None = None,
        clock: Callable[[], float] = time.time,
    ):
        if dim <= 0:
            raise ValueError("dim must be positive")
        if max_entries is not None and max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self.dim = dim
        self.policy: ThresholdPolicy = policy or GlobalThreshold()
        self.default_ttl_seconds = default_ttl_seconds
        self.max_entries = max_entries
        self._clock = clock
        self._lock = threading.RLock()
        self._entries: dict[int, StoredEntry] = {}
        self._partitions: dict[str, _Partition] = {}
        self._lru: OrderedDict[int, None] = OrderedDict()  # oldest first
        self._by_tag: dict[str, set[int]] = defaultdict(set)
        self._counters: dict[str, _Counters] = defaultdict(_Counters)
        self._next_id = 1

        if str(db_path) != ":memory:":
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(db_path), check_same_thread=False)
        self._db.executescript(_SCHEMA)
        if str(db_path) != ":memory:":
            self._db.execute("PRAGMA journal_mode=WAL")
            self._db.execute("PRAGMA synchronous=NORMAL")
        self._restore()

    # ------------------------------------------------------------------ lookups

    def lookup(self, partition: str, embedding: np.ndarray) -> StoreLookup:
        """Nearest entry + threshold policy in one call (no external verifier)."""
        query = self._prepare(embedding)
        with self._lock:
            nearest = self._nearest_valid(partition, query, self._clock())
            if nearest is None:
                self.record(partition, None, hit=False)
                return StoreLookup(False, None, None, None)
            entry, similarity = nearest
            threshold = self.policy.threshold(partition, entry.cluster)
            hit = similarity >= threshold
            self.record(partition, entry if hit else None, hit=hit, reason="threshold")
            return StoreLookup(hit, entry if hit else None, similarity, threshold, entry.query)

    def nearest(self, partition: str, embedding: np.ndarray) -> tuple[StoredEntry, float] | None:
        """Nearest unexpired entry in the partition, without recording anything.

        Used with ``record`` when the hit decision is made outside the store
        (the guard), so a slow verifier never runs while holding the store lock.
        """
        query = self._prepare(embedding)
        with self._lock:
            return self._nearest_valid(partition, query, self._clock())

    def record(self, partition: str, entry: StoredEntry | None, hit: bool, reason: str | None = None) -> None:
        """Count a lookup outcome; on a hit, refresh the entry's recency and hit count."""
        with self._lock:
            counters = self._counters[partition]
            if reason is not None:
                counters.decisions[reason] = counters.decisions.get(reason, 0) + 1
            if not hit:
                counters.misses += 1
                return
            counters.hits += 1
            if entry is None or entry.entry_id not in self._entries:
                return  # evicted or invalidated while the decision was being made
            now = self._clock()
            entry.hit_count += 1
            entry.last_hit_at = now
            self._lru.move_to_end(entry.entry_id)
            self._db.execute(
                "UPDATE entries SET hit_count = ?, last_hit_at = ? WHERE entry_id = ?",
                (entry.hit_count, now, entry.entry_id),
            )
            self._db.commit()

    def _nearest_valid(self, partition: str, query: np.ndarray, now: float) -> tuple[StoredEntry, float] | None:
        part = self._partitions.get(partition)
        while part is not None and part.index.ntotal > 0:
            k = min(_SEARCH_DEPTH, part.index.ntotal)
            scores, ids = part.index.search(query.reshape(1, -1), k)
            expired = []
            for score, entry_id in zip(scores[0], ids[0]):
                if entry_id < 0:
                    continue
                entry = self._entries[int(entry_id)]
                if entry.expired(now):
                    expired.append(entry.entry_id)
                    continue
                self._remove_ids(expired, reason="expirations")
                return entry, float(score)
            if not expired:
                return None
            self._remove_ids(expired, reason="expirations")  # all candidates expired; search again
            part = self._partitions.get(partition)
        return None

    # ------------------------------------------------------------------ writes

    def add(
        self,
        partition: str,
        query: str,
        embedding: np.ndarray,
        response: Any,
        cluster: int | None = None,
        tags: Iterable[str] = (),
        ttl_seconds: float | None = None,
    ) -> StoredEntry:
        vector = self._prepare(embedding)
        ttl = self.default_ttl_seconds if ttl_seconds is None else ttl_seconds
        response_json = json.dumps(response)  # fail before touching state if not serializable
        with self._lock:
            now = self._clock()
            entry = StoredEntry(
                entry_id=self._next_id,
                partition=partition,
                query=query,
                response=response,
                cluster=cluster,
                tags=tuple(dict.fromkeys(tags)),
                created_at=now,
                expires_at=None if ttl is None else now + ttl,
            )
            self._next_id += 1
            self._db.execute(
                "INSERT INTO entries (entry_id, partition, query, embedding, response, cluster, tags, created_at, expires_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (entry.entry_id, partition, query, vector.tobytes(), response_json, cluster,
                 json.dumps(entry.tags), entry.created_at, entry.expires_at),
            )
            self._index_entry(entry, vector)
            self._evict_if_needed()
            self._db.commit()
            return entry

    def invalidate(self, *, entry_id: int | None = None, tag: str | None = None, partition: str | None = None) -> int:
        """Remove entries matching exactly one selector. Returns how many were removed."""
        if sum(x is not None for x in (entry_id, tag, partition)) != 1:
            raise ValueError("pass exactly one of entry_id, tag, partition")
        with self._lock:
            if entry_id is not None:
                ids = [entry_id] if entry_id in self._entries else []
            elif tag is not None:
                ids = list(self._by_tag.get(tag, ()))
            else:
                part = self._partitions.get(partition)
                ids = list(part.ids) if part else []
            self._remove_ids(ids, reason="invalidations")
            self._db.commit()
            return len(ids)

    def purge_expired(self) -> int:
        with self._lock:
            now = self._clock()
            ids = [e.entry_id for e in self._entries.values() if e.expired(now)]
            self._remove_ids(ids, reason="expirations")
            self._db.commit()
            return len(ids)

    def clear(self) -> None:
        with self._lock:
            self._db.execute("DELETE FROM entries")
            self._db.commit()
            self._entries.clear()
            self._partitions.clear()
            self._lru.clear()
            self._by_tag.clear()
            self._counters.clear()

    # ------------------------------------------------------------------ introspection

    def __len__(self) -> int:
        return len(self._entries)

    def get(self, entry_id: int) -> StoredEntry | None:
        return self._entries.get(entry_id)

    def stats(self) -> dict:
        with self._lock:
            totals = _Counters()
            partitions = {}
            for name in sorted(set(self._partitions) | set(self._counters)):
                c = self._counters[name]
                totals.merge(c)
                part = self._partitions.get(name)
                partitions[name] = {"entries": len(part.ids) if part else 0, **c.as_dict()}
            policy = getattr(self.policy, "describe", lambda: {"type": type(self.policy).__name__})()
            return {
                "total_entries": len(self._entries),
                "max_entries": self.max_entries,
                "default_ttl_seconds": self.default_ttl_seconds,
                "policy": policy,
                **totals.as_dict(),
                "partitions": partitions,
            }

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------------ internals

    def _prepare(self, embedding: np.ndarray) -> np.ndarray:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.dim:
            raise ValueError(f"embedding has dim {vector.shape[0]}, store expects {self.dim}")
        return np.ascontiguousarray(normalize_rows(vector.reshape(1, -1))[0])

    def _index_entry(self, entry: StoredEntry, vector: np.ndarray) -> None:
        # The vector lives only in FAISS (and SQLite); entries do not keep a copy.
        part = self._partitions.get(entry.partition)
        if part is None:
            part = self._partitions[entry.partition] = _Partition(faiss.IndexIDMap2(faiss.IndexFlatIP(self.dim)))
        part.index.add_with_ids(vector.reshape(1, -1), np.array([entry.entry_id], dtype=np.int64))
        part.ids.add(entry.entry_id)
        self._entries[entry.entry_id] = entry
        self._lru[entry.entry_id] = None
        for tag in entry.tags:
            self._by_tag[tag].add(entry.entry_id)

    def _remove_ids(self, ids: Iterable[int], reason: str) -> None:
        by_partition: dict[str, list[int]] = defaultdict(list)
        for entry_id in ids:
            entry = self._entries.pop(entry_id, None)
            if entry is None:
                continue
            by_partition[entry.partition].append(entry_id)
            self._lru.pop(entry_id, None)
            for tag in entry.tags:
                tagged = self._by_tag.get(tag)
                if tagged is not None:
                    tagged.discard(entry_id)
                    if not tagged:
                        del self._by_tag[tag]
            counters = self._counters[entry.partition]
            setattr(counters, reason, getattr(counters, reason) + 1)
        for name, removed in by_partition.items():
            part = self._partitions[name]
            part.index.remove_ids(np.array(removed, dtype=np.int64))
            part.ids.difference_update(removed)
            if not part.ids:
                del self._partitions[name]
            self._db.executemany("DELETE FROM entries WHERE entry_id = ?", [(i,) for i in removed])

    def _evict_if_needed(self) -> None:
        if self.max_entries is None or len(self._entries) <= self.max_entries:
            return
        excess = len(self._entries) - self.max_entries
        victims = [entry_id for entry_id, _ in zip(self._lru, range(excess))]
        self._remove_ids(victims, reason="evictions")

    def _restore(self) -> None:
        rows = self._db.execute(
            "SELECT entry_id, partition, query, embedding, response, cluster, tags, created_at, expires_at,"
            " last_hit_at, hit_count FROM entries ORDER BY COALESCE(last_hit_at, created_at), entry_id"
        ).fetchall()
        now = self._clock()
        stale = []
        for row in rows:
            embedding = np.frombuffer(row[3], dtype=np.float32).copy()
            entry = StoredEntry(
                entry_id=row[0], partition=row[1], query=row[2],
                response=json.loads(row[4]), cluster=row[5], tags=tuple(json.loads(row[6])),
                created_at=row[7], expires_at=row[8], last_hit_at=row[9], hit_count=row[10],
            )
            if embedding.shape[0] != self.dim or entry.expired(now):
                stale.append((entry.entry_id,))
                continue
            self._index_entry(entry, embedding)  # rows arrive oldest-use first, so LRU order is rebuilt
        if stale:
            self._db.executemany("DELETE FROM entries WHERE entry_id = ?", stale)
            self._db.commit()
        if rows:
            self._next_id = max(r[0] for r in rows) + 1
        self._evict_if_needed()
        self._db.commit()
