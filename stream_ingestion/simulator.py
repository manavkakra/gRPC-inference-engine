from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import time
import uuid
import sys
import os
from dataclasses import asdict, dataclass
from typing import AsyncIterator, Callable, Optional

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

logger = logging.getLogger(__name__)


MERCHANT_CATEGORIES = [
    "grocery",
    "restaurant",
    "gas_station",
    "online",
    "retail",
    "travel",
    "atm",
    "other",
]

CITY_CENTROIDS = {
    "NYC": (40.7128, -74.0060),
    "LA": (34.0522, -118.2437),
    "Chicago": (41.8781, -87.6298),
    "Houston": (29.7604, -95.3698),
    "Phoenix": (33.4484, -112.0740),
}


@dataclass
class Transaction:
    transaction_id: str
    entity_id: str
    amount: float
    merchant_id: str
    merchant_category: str
    latitude: float
    longitude: float
    timestamp_ms: int
    is_fraud: bool = False
    fraud_type: str = "none"


class EntityProfile:
    """Behavioural profile for a simulated user/account."""

    def __init__(self, entity_id: str, profile_type: str = "normal"):
        self.entity_id = entity_id
        self.profile_type = profile_type

        city_name, (clat, clon) = random.choice(list(CITY_CENTROIDS.items()))
        self.home_city = city_name
        self.home_lat = clat
        self.home_lon = clon

        if profile_type == "normal":
            self.mean_amount = random.uniform(20, 150)
            self.std_amount = self.mean_amount * 0.3
            self.txn_rate_hz = random.uniform(0.01, 0.1)
            self.preferred_cats = random.sample(MERCHANT_CATEGORIES, k=3)
        elif profile_type == "churner":
            self.mean_amount = random.uniform(20, 100)
            self.std_amount = self.mean_amount * 0.3
            self.txn_rate_hz = random.uniform(0.5, 2.0)
            self.preferred_cats = MERCHANT_CATEGORIES
        else:
            self.mean_amount = random.uniform(500, 3000)
            self.std_amount = self.mean_amount * 0.5
            self.txn_rate_hz = random.uniform(0.2, 1.0)
            self.preferred_cats = ["online", "atm", "travel"]

    def generate_transaction(self) -> Transaction:
        now_ms = int(time.time() * 1000)
        is_fraud = self.profile_type != "normal"

        amount = max(1.0, np.random.normal(self.mean_amount, self.std_amount))
        if self.profile_type == "fraudster" and random.random() < 0.3:
            amount *= random.uniform(3, 10)

        if self.profile_type == "fraudster" and random.random() < 0.6:
            lat = self.home_lat + random.uniform(-20, 20)
            lon = self.home_lon + random.uniform(-20, 20)
        else:
            lat = self.home_lat + np.random.normal(0, 0.05)
            lon = self.home_lon + np.random.normal(0, 0.05)

        category = random.choice(self.preferred_cats)
        merchant = f"merch_{category }_{random .randint (1 ,200 ):04d}"

        return Transaction(
            transaction_id=str(uuid.uuid4()),
            entity_id=self.entity_id,
            amount=round(amount, 2),
            merchant_id=merchant,
            merchant_category=category,
            latitude=round(lat, 6),
            longitude=round(lon, 6),
            timestamp_ms=now_ms,
            is_fraud=is_fraud,
            fraud_type=self.profile_type if is_fraud else "none",
        )


class KafkaProducer:
    """Async Kafka producer; gracefully degrades when Kafka is unavailable."""

    TOPIC = "transactions"

    def __init__(self, bootstrap: str = "127.0.0.1:9092"):
        self._available = False
        self._producer = None
        self._bootstrap = bootstrap
        self._sent_count = 0

    async def start(self) -> None:
        try:
            from aiokafka import AIOKafkaProducer

            self._producer = AIOKafkaProducer(
                bootstrap_servers=self._bootstrap,
                value_serializer=lambda v: json.dumps(v).encode(),
                linger_ms=5,
                acks=1,
            )
            await self._producer.start()
            self._available = True
            logger.info("Kafka producer connected to %s", self._bootstrap)
        except Exception as exc:
            logger.warning("Kafka unavailable (%s) — in-memory mode.", exc)

    async def stop(self) -> None:
        if self._producer:
            await self._producer.stop()

    async def send(self, txn: Transaction) -> None:
        if not self._available:
            return
        await self._producer.send(
            self.TOPIC,
            value=asdict(txn),
            key=txn.entity_id.encode(),
        )
        self._sent_count += 1

    @property
    def sent(self) -> int:
        return self._sent_count


class TransactionSimulator:
    """
    Drives multiple entity profiles and emits transactions at configurable rates.

    Architecture:
      • N_NORMAL  normal profiles   → ~80% of traffic
      • N_CHURN   churner profiles  → ~10% of traffic (high velocity)
      • N_FRAUD   fraudster profiles → ~10% of traffic (high amounts)
      • asyncio.gather() runs all entity coroutines concurrently
      • on_transaction callback feeds FeatureStore + Kafka
    """

    def __init__(
        self,
        n_normal: int = 400,
        n_churner: int = 50,
        n_fraudster: int = 50,
        target_rps: int = 1000,
        kafka_bootstrap: str = "127.0.0.1:9092",
        on_transaction: Optional[Callable[[Transaction], None]] = None,
        grpc_target: Optional[str] = None,
    ):
        self._on_transaction = on_transaction
        self._target_rps = target_rps
        self._running = False
        self._grpc_target = grpc_target
        self._grpc_channel = None
        self._grpc_stub = None
        self._grpc_frauds = 0

        self._profiles: list[EntityProfile] = []
        for i in range(n_normal):
            self._profiles.append(EntityProfile(f"user_n_{i :05d}", "normal"))
        for i in range(n_churner):
            self._profiles.append(EntityProfile(f"user_c_{i :05d}", "churner"))
        for i in range(n_fraudster):
            self._profiles.append(EntityProfile(f"user_f_{i :05d}", "fraudster"))

        self._kafka = KafkaProducer(kafka_bootstrap)

        self._emitted = 0
        self._start_time = 0.0
        self._fraud_emitted = 0

    async def run(self, duration_seconds: float = 0.0) -> None:
        """
        Run the simulator.
        duration_seconds=0 → run until cancelled.
        """
        if self._grpc_target:
            import grpc
            import inference_pb2_grpc
            
            self._grpc_channel = grpc.aio.insecure_channel(self._grpc_target)
            self._grpc_stub = inference_pb2_grpc.InferenceServiceStub(self._grpc_channel)
            logger.info("gRPC integration enabled, target: %s", self._grpc_target)

        await self._kafka.start()
        self._running = True
        self._start_time = time.time()
        logger.info(
            "Simulator started: %d entities, target %d RPS",
            len(self._profiles),
            self._target_rps,
        )

        tasks = [self._entity_loop(p) for p in self._profiles]
        if duration_seconds > 0:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks), timeout=duration_seconds)
            except asyncio.TimeoutError:
                pass
        else:
            await asyncio.gather(*tasks)

        self._running = False
        await self._kafka.stop()
        if self._grpc_channel:
            await self._grpc_channel.close()
        self._print_summary()

    async def _entity_loop(self, profile: EntityProfile) -> None:
        """Coroutine that continuously emits transactions for one entity."""
        while self._running:

            interval = random.expovariate(profile.txn_rate_hz)
            await asyncio.sleep(interval)

            txn = profile.generate_transaction()
            await self._emit(txn)

    async def _emit(self, txn: Transaction) -> None:
        """Send transaction to Kafka and fire the callback."""
        self._emitted += 1
        if txn.is_fraud:
            self._fraud_emitted += 1

        if self._on_transaction:

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._on_transaction, txn)

        await self._kafka.send(txn)
        
        if self._grpc_stub:
            # Wait 2.0s to ensure Feature Store has ingested this transaction into Redis
            # (Increased from 50ms because local CPUs cause a Kafka backlog)
            async def delayed_grpc():
                await asyncio.sleep(2.0)
                await self._send_grpc(txn)
            asyncio.create_task(delayed_grpc())

        if self._emitted % 5000 == 0:
            elapsed = time.time() - self._start_time
            rps = self._emitted / max(elapsed, 0.001)
            logger.info(
                "Emitted %d events | %.0f RPS | fraud rate %.1f%%",
                self._emitted,
                rps,
                100.0 * self._fraud_emitted / max(self._emitted, 1),
            )

    async def _send_grpc(self, txn: Transaction) -> None:
        import inference_pb2
        req = inference_pb2.InferenceRequest(
            request_id=str(uuid.uuid4()),
            transaction=inference_pb2.Transaction(
                transaction_id=txn.transaction_id,
                entity_id=txn.entity_id,
                amount=txn.amount,
                merchant_category=txn.merchant_category,
                latitude=txn.latitude,
                longitude=txn.longitude,
                timestamp_ms=int(txn.timestamp_ms),
            ),
        )
        try:
            resp = await self._grpc_stub.Predict(req, timeout=1.0)
            if resp.is_fraud:
                self._grpc_frauds += 1
        except Exception:
            pass

    def _print_summary(self) -> None:
        elapsed = time.time() - self._start_time
        rps = self._emitted / max(elapsed, 0.001)
        print(
            f"\n{'─'*50 }\n"
            f"  Simulator completed\n"
            f"  Duration  : {elapsed :.1f}s\n"
            f"  Emitted   : {self ._emitted :,} events\n"
            f"  Throughput: {rps :,.0f} RPS\n"
            f"  Sim Fraud : {100 *self ._fraud_emitted /max (self ._emitted ,1 ):.1f}%\n"
            f"  gRPC Found: {100 *self ._grpc_frauds /max (self ._emitted ,1 ):.1f}%\n"
            f"  Kafka sent: {self ._kafka .sent :,}\n"
            f"{'─'*50 }"
        )

    @property
    def emitted(self) -> int:
        return self._emitted

    def current_rps(self) -> float:
        elapsed = time.time() - self._start_time
        return self._emitted / max(elapsed, 0.001)


class TransactionConsumer:
    """
    Reads from the Kafka 'transactions' topic and yields Transaction objects.
    Used by the feature store server to stay up-to-date across restarts.
    """

    def __init__(
        self,
        bootstrap: str = "127.0.0.1:9092",
        group_id: str = "feature-store-consumer",
        auto_offset: str = "latest",
    ):
        self._bootstrap = bootstrap
        self._group_id = group_id
        self._auto_offset = auto_offset
        self._consumer = None
        self._available = False

    async def start(self) -> None:
        try:
            from aiokafka import AIOKafkaConsumer

            self._consumer = AIOKafkaConsumer(
                KafkaProducer.TOPIC,
                bootstrap_servers=self._bootstrap,
                group_id=self._group_id,
                auto_offset_reset=self._auto_offset,
                enable_auto_commit=True,
                value_deserializer=lambda b: json.loads(b.decode()),
            )
            await self._consumer.start()
            self._available = True
            logger.info("Kafka consumer started (group=%s)", self._group_id)
        except Exception as exc:
            logger.warning("Kafka consumer unavailable: %s", exc)

    async def stop(self) -> None:
        if self._consumer:
            await self._consumer.stop()

    async def __aiter__(self) -> AsyncIterator[Transaction]:
        if not self._available:
            return
        async for msg in self._consumer:
            try:
                data = msg.value
                yield Transaction(**data)
            except Exception as exc:
                logger.debug("Malformed message: %s", exc)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Transaction stream simulator")
    parser.add_argument("--normal", type=int, default=400, help="Normal user count")
    parser.add_argument("--churners", type=int, default=50, help="Churner count")
    parser.add_argument("--fraudsters", type=int, default=50, help="Fraudster count")
    parser.add_argument(
        "--duration", type=float, default=60.0, help="Runtime in seconds (0=forever)"
    )
    parser.add_argument("--kafka", type=str, default="127.0.0.1:9092")
    parser.add_argument("--grpc", action="store_true", help="Mirror traffic to Inference Engine")
    parser.add_argument("--grpc-target", type=str, default="127.0.0.1:50051")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    sim = TransactionSimulator(
        n_normal=args.normal,
        n_churner=args.churners,
        n_fraudster=args.fraudsters,
        kafka_bootstrap=args.kafka,
        grpc_target=args.grpc_target if args.grpc else None,
    )
    asyncio.run(sim.run(duration_seconds=args.duration))
