from __future__ import annotations

import asyncio
import logging
import random
import time
import uuid
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from queue import Empty, Queue
from typing import AsyncIterator, Iterator, List, Optional

import grpc

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    decision: str
    confidence: float
    inference_latency_us: int
    feature_latency_us: int
    model_version: str
    features: Optional[dict] = None
    explanations: Optional[dict] = None


@dataclass
class HealthResult:
    healthy: bool
    version: str
    uptime_seconds: int
    requests_served: int
    cache_hit_rate: float


_CHANNEL_OPTIONS = [
    ("grpc.keepalive_time_ms", 10_000),
    ("grpc.keepalive_timeout_ms", 5_000),
    ("grpc.keepalive_permit_without_calls", 1),
    ("grpc.http2.max_pings_without_data", 0),
    ("grpc.max_receive_message_length", 16 * 1024 * 1024),
    ("grpc.max_send_message_length", 16 * 1024 * 1024),
    ("grpc.enable_retries", 1),
    (
        "grpc.service_config",
        '{"retryPolicy": {'
        '"maxAttempts": 3,'
        '"initialBackoff": "0.1s",'
        '"maxBackoff": "1s",'
        '"backoffMultiplier": 2,'
        '"retryableStatusCodes": ["UNAVAILABLE", "DEADLINE_EXCEEDED"]'
        "}}",
    ),
]


def _make_channel(target: str) -> grpc.Channel:
    return grpc.insecure_channel(target, options=_CHANNEL_OPTIONS)


class InferenceClient:
    """
    Thread-safe synchronous gRPC client with connection pooling.

    Args:
        target:       "host:port" of the gRPC server
        pool_size:    Number of gRPC channels to maintain
        timeout_ms:   Default per-call deadline in milliseconds
    """

    def __init__(
        self,
        target: str = "127.0.0.1:50051",
        pool_size: int = 4,
        timeout_ms: int = 5_000,
    ):
        self._target = target
        self._timeout_ms = timeout_ms
        self._pool: Queue[grpc.Channel] = Queue()

        try:
            import inference_pb2
            import inference_pb2_grpc

            self._pb2 = inference_pb2
            self._pb2_grpc = inference_pb2_grpc
        except ImportError as exc:
            raise RuntimeError(
                "Generated gRPC stubs not found. " "Run `python scripts/compile_proto.py` first."
            ) from exc

        for _ in range(pool_size):
            ch = _make_channel(target)
            stub = inference_pb2_grpc.InferenceServiceStub(ch)
            self._pool.put((ch, stub))

        logger.info("InferenceClient pool(%d) → %s", pool_size, target)

    @contextmanager
    def _stub(self):
        ch, stub = self._pool.get()
        try:
            yield stub
        finally:
            self._pool.put((ch, stub))

    def _deadline(self) -> float:
        return time.monotonic() + self._timeout_ms / 1000.0

    def predict(
        self,
        entity_id: str,
        amount: float,
        merchant_category: str = "other",
        lat: float = 0.0,
        lon: float = 0.0,
        transaction_id: Optional[str] = None,
        include_features: bool = False,
        explain: bool = False,
    ) -> PredictionResult:
        txn_id = transaction_id or str(uuid.uuid4())
        req = self._pb2.InferenceRequest(
            request_id=txn_id,
            include_features=include_features,
            explain=explain,
            transaction=self._pb2.Transaction(
                transaction_id=txn_id,
                entity_id=entity_id,
                amount=amount,
                merchant_category=merchant_category,
                latitude=lat,
                longitude=lon,
                timestamp_ms=int(time.time() * 1000),
            ),
        )
        with self._stub() as stub:
            resp = stub.Predict(req, timeout=self._timeout_ms / 1000.0)

        return PredictionResult(
            transaction_id=resp.transaction_id,
            fraud_probability=resp.fraud_probability,
            is_fraud=resp.is_fraud,
            decision=resp.decision,
            confidence=resp.confidence,
            inference_latency_us=resp.inference_latency_us,
            feature_latency_us=resp.feature_latency_us,
            model_version=resp.model_version,
        )

    def batch_predict(self, requests: list[dict]) -> list[PredictionResult]:
        """
        Submit a batch of requests. Each dict should have keys matching
        predict()'s keyword arguments.
        """
        pb_requests = []
        for r in requests:
            txn_id = r.get("transaction_id") or str(uuid.uuid4())
            pb_requests.append(
                self._pb2.InferenceRequest(
                    request_id=txn_id,
                    transaction=self._pb2.Transaction(
                        transaction_id=txn_id,
                        entity_id=r["entity_id"],
                        amount=r["amount"],
                        merchant_category=r.get("merchant_category", "other"),
                        latitude=r.get("lat", 0.0),
                        longitude=r.get("lon", 0.0),
                        timestamp_ms=int(time.time() * 1000),
                    ),
                )
            )

        batch_req = self._pb2.BatchInferenceRequest(
            requests=pb_requests,
            timeout_ms=self._timeout_ms,
        )
        with self._stub() as stub:
            batch_resp = stub.BatchPredict(batch_req, timeout=self._timeout_ms / 1000.0)

        return [
            PredictionResult(
                transaction_id=r.transaction_id,
                fraud_probability=r.fraud_probability,
                is_fraud=r.is_fraud,
                decision=r.decision,
                confidence=r.confidence,
                inference_latency_us=r.inference_latency_us,
                feature_latency_us=r.feature_latency_us,
                model_version=r.model_version,
            )
            for r in batch_resp.responses
        ]

    def stream_predict(
        self,
        transactions: Iterator[dict],
    ) -> Iterator[PredictionResult]:
        """
        Send a stream of transactions and yield inference results as they arrive.
        Implements bidirectional gRPC streaming.
        """

        def _txn_iter():
            for r in transactions:
                yield self._pb2.Transaction(
                    transaction_id=r.get("transaction_id", str(uuid.uuid4())),
                    entity_id=r["entity_id"],
                    amount=r["amount"],
                    merchant_category=r.get("merchant_category", "other"),
                    latitude=r.get("lat", 0.0),
                    longitude=r.get("lon", 0.0),
                    timestamp_ms=int(time.time() * 1000),
                )

        with self._stub() as stub:
            for resp in stub.StreamPredict(_txn_iter()):
                yield PredictionResult(
                    transaction_id=resp.transaction_id,
                    fraud_probability=resp.fraud_probability,
                    is_fraud=resp.is_fraud,
                    decision=resp.decision,
                    confidence=resp.confidence,
                    inference_latency_us=resp.inference_latency_us,
                    feature_latency_us=resp.feature_latency_us,
                    model_version=resp.model_version,
                )

    def health(self) -> HealthResult:
        with self._stub() as stub:
            resp = stub.Health(self._pb2.HealthRequest(), timeout=2.0)
        return HealthResult(
            healthy=resp.healthy,
            version=resp.version,
            uptime_seconds=resp.uptime_seconds,
            requests_served=resp.requests_served,
            cache_hit_rate=resp.cache_hit_rate,
        )

    def close(self) -> None:
        while not self._pool.empty():
            ch, _ = self._pool.get_nowait()
            ch.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class AsyncInferenceClient:
    """
    Async gRPC client using aio channel (grpc.aio).
    One channel is sufficient for async — the HTTP/2 transport multiplexes.
    """

    def __init__(self, target: str = "127.0.0.1:50051", timeout_ms: int = 5_000):
        self._target = target
        self._timeout_ms = timeout_ms
        self._channel = None
        self._stub = None

        try:
            import inference_pb2
            import inference_pb2_grpc

            self._pb2 = inference_pb2
            self._pb2_grpc = inference_pb2_grpc
        except ImportError as exc:
            raise RuntimeError("Compile proto stubs first.") from exc

    async def connect(self) -> None:
        self._channel = grpc.aio.insecure_channel(self._target, options=_CHANNEL_OPTIONS)
        self._stub = self._pb2_grpc.InferenceServiceStub(self._channel)
        logger.info("AsyncInferenceClient connected → %s", self._target)

    async def close(self) -> None:
        if self._channel:
            await self._channel.close()

    async def predict(
        self,
        entity_id: str,
        amount: float,
        merchant_category: str = "other",
        lat: float = 0.0,
        lon: float = 0.0,
        transaction_id: Optional[str] = None,
    ) -> PredictionResult:
        txn_id = transaction_id or str(uuid.uuid4())
        req = self._pb2.InferenceRequest(
            request_id=txn_id,
            transaction=self._pb2.Transaction(
                transaction_id=txn_id,
                entity_id=entity_id,
                amount=amount,
                merchant_category=merchant_category,
                latitude=lat,
                longitude=lon,
                timestamp_ms=int(time.time() * 1000),
            ),
        )
        resp = await self._stub.Predict(req, timeout=self._timeout_ms / 1000.0)
        return PredictionResult(
            transaction_id=resp.transaction_id,
            fraud_probability=resp.fraud_probability,
            is_fraud=resp.is_fraud,
            decision=resp.decision,
            confidence=resp.confidence,
            inference_latency_us=resp.inference_latency_us,
            feature_latency_us=resp.feature_latency_us,
            model_version=resp.model_version,
        )

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, *_):
        await self.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Quick inference client test")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--entity", default="test_user_001")
    parser.add_argument("--amount", type=float, default=250.0)
    parser.add_argument("--n", type=int, default=10, help="Number of requests")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    target = f"{args .host }:{args .port }"
    print(f"\nConnecting to {target } …")

    with InferenceClient(target) as client:

        try:
            h = client.health()
            print(
                f"Server healthy: {h .healthy }  model v{h .version }  "
                f"uptime {h .uptime_seconds }s  "
                f"cache_hit_rate {h .cache_hit_rate :.2%}"
            )
        except Exception as e:
            print(f"Health check failed: {e }")

        print(f"\nSending {args .n } predictions …")
        latencies = []
        for i in range(args.n):
            try:
                r = client.predict(
                    entity_id=args.entity,
                    amount=args.amount + i * 10,
                    merchant_category=random.choice(["online", "grocery", "atm"]),
                )
                latencies.append(r.inference_latency_us / 1000)
                print(
                    f"  [{i +1 :2d}] {r .decision :7s}  "
                    f"p(fraud)={r .fraud_probability :.3f}  "
                    f"latency={r .inference_latency_us /1000 :.1f}ms"
                )
            except grpc.RpcError as e:
                print(f"  [{i +1 :2d}] RPC error: {e .code ()}")

        if latencies:
            from statistics import mean

            print(f"\n  mean latency: {mean (latencies ):.2f}ms  " f"max: {max (latencies ):.2f}ms")
