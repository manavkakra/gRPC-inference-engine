from __future__ import annotations

import json
import logging
import threading
import time
from collections import OrderedDict
from dataclasses import asdict
from typing import Optional

from .rolling_engine import FeatureSnapshot, RollingFeatureEngine

logger = logging.getLogger(__name__)


class LRUCache:
    """Thread-safe LRU cache backed by an OrderedDict."""

    def __init__(self, capacity: int = 10_000):
        self._cap = capacity
        self._store: OrderedDict[str, FeatureSnapshot] = OrderedDict()
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def get(self, key: str) -> Optional[FeatureSnapshot]:
        with self._lock:
            if key not in self._store:
                self._misses += 1
                return None
            self._store.move_to_end(key)
            self._hits += 1
            return self._store[key]

    def set(self, key: str, value: FeatureSnapshot) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            if len(self._store) > self._cap:
                self._store.popitem(last=False)

    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def size(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0


class RedisStore:
    """
    Thin wrapper around redis-py with JSON serialisation.
    Falls back to a no-op if Redis is unreachable (dev / unit-test mode).
    """

    _TTL_SECONDS = 300

    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0):
        self._available = False
        try:
            import redis

            self._client = redis.Redis(
                host=host,
                port=port,
                db=db,
                socket_connect_timeout=2,
                socket_timeout=0.1,
                decode_responses=True,
            )
            self._client.ping()
            self._available = True
            logger.info("Redis connected at %s:%d", host, port)
        except Exception as exc:
            logger.warning("Redis unavailable (%s) — running L1-only mode.", exc)
            self._client = None

    @property
    def available(self) -> bool:
        return self._available

    def _key(self, entity_id: str) -> str:
        return f"fs:v1:{entity_id }"

    def get(self, entity_id: str) -> Optional[FeatureSnapshot]:
        if not self._available:
            return None
        try:
            raw = self._client.get(self._key(entity_id))
            if raw is None:
                return None
            data = json.loads(raw)
            return FeatureSnapshot(**data)
        except Exception as exc:
            logger.debug("Redis GET error: %s", exc)
            return None

    def set(self, snap: FeatureSnapshot) -> None:
        if not self._available:
            return
        try:
            payload = json.dumps(snap.to_dict())
            self._client.setex(self._key(snap.entity_id), self._TTL_SECONDS, payload)
        except Exception as exc:
            logger.debug("Redis SET error: %s", exc)

    def pipeline_set(self, snaps: list[FeatureSnapshot]) -> None:
        """Batch-write multiple snapshots in a single Redis pipeline."""
        if not self._available or not snaps:
            return
        try:
            pipe = self._client.pipeline(transaction=False)
            for snap in snaps:
                pipe.setex(self._key(snap.entity_id), self._TTL_SECONDS, json.dumps(snap.to_dict()))
            pipe.execute()
        except Exception as exc:
            logger.debug("Redis pipeline error: %s", exc)


class FeatureStore:
    """
    Unified interface for feature ingestion and retrieval.

    Usage:
        store = FeatureStore()

        # On every new transaction event:
        store.ingest(entity_id="user_123", amount=42.5, lat=40.7, lon=-74.0,
                     merchant="coffee_shop", merchant_category="restaurant")

        # At inference time:
        features = store.get_features("user_123", current_amount=42.5, ...)
        model_input = features.to_model_array()
    """

    def __init__(
        self,
        redis_host: str = "localhost",
        redis_port: int = 6379,
        l1_capacity: int = 10_000,
        buffer_capacity: int = 1_000,
        max_entities: int = 100_000,
    ):
        self._engine = RollingFeatureEngine(
            buffer_capacity=buffer_capacity,
            max_entities=max_entities,
        )
        self._l1 = LRUCache(capacity=l1_capacity)
        self._l2 = RedisStore(host=redis_host, port=redis_port)

        self._write_count = 0
        self._start_time = time.time()
        self._write_lock = threading.Lock()

    def ingest(
        self,
        entity_id: str,
        amount: float,
        lat: float = 0.0,
        lon: float = 0.0,
        merchant: str = "unknown",
        merchant_category: str = "other",
        ts: Optional[float] = None,
    ) -> None:
        """Ingest a new event and update both cache tiers."""
        self._engine.ingest(entity_id, amount, lat, lon, merchant, ts)

        snap = self._engine.compute_features(entity_id, amount, lat, lon, merchant_category)
        self._l1.set(entity_id, snap)
        self._l2.set(snap)
        with self._write_lock:
            self._write_count += 1

    def get_features(
        self,
        entity_id: str,
        current_amount: float = 0.0,
        current_lat: float = 0.0,
        current_lon: float = 0.0,
        merchant_category: str = "other",
        max_age_seconds: float = 5.0,
    ) -> FeatureSnapshot:
        """
        Retrieve the latest FeatureSnapshot for entity_id.

        Cache hierarchy:
          L1 (memory) → L2 (Redis) → live compute (fallback)
        """
        now = time.time()
        import dataclasses

        snap = self._l1.get(entity_id)
        if snap is None:
            snap = self._l2.get(entity_id)
            if snap is not None:
                self._l1.set(entity_id, snap)

        if snap is not None:
            return dataclasses.replace(
                snap,
                current_amount=current_amount,
                merchant_category=merchant_category,
                computed_at=now
            )

        snap = self._engine.compute_features(
            entity_id, current_amount, current_lat, current_lon, merchant_category
        )
        self._l1.set(entity_id, snap)
        self._l2.set(snap)
        return snap

    def clear_cache(self) -> None:
        """Clear the L1 in-memory cache."""
        self._l1.clear()

    def health(self) -> dict:
        engine_stats = self._engine.stats()
        return {
            "uptime_seconds": int(time.time() - self._start_time),
            "writes": self._write_count,
            "l1_size": self._l1.size(),
            "l1_cache_hit_rate": round(self._l1.hit_rate(), 4),
            "l2_available": self._l2.available,
            **engine_stats,
        }
