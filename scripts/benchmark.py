from __future__ import annotations

import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from statistics import mean, stdev
from typing import Callable

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from feature_store.rolling_engine import FeatureSnapshot
from feature_store.store import FeatureStore


def percentile(data: list[float], p: float) -> float:
    arr = sorted(data)
    idx = int(len(arr) * p / 100)
    return arr[min(idx, len(arr) - 1)]


def print_latency_report(name: str, latencies_us: list[float]) -> None:
    lat_ms = [x / 1000 for x in latencies_us]
    print(f"\n{'─'*50 }")
    print(f"  {name }")
    print(f"  n={len (lat_ms ):,}")
    print(f"  p50  : {percentile (lat_ms ,50 ):.3f} ms")
    print(f"  p90  : {percentile (lat_ms ,90 ):.3f} ms")
    print(f"  p95  : {percentile (lat_ms ,95 ):.3f} ms")
    print(f"  p99  : {percentile (lat_ms ,99 ):.3f} ms")
    print(f"  mean : {mean (lat_ms ):.3f} ms")
    print(f"  max  : {max (lat_ms ):.3f} ms")


def bench_ingestion(store: FeatureStore, n: int = 10_000) -> None:
    print(f"\n[1/3] Ingestion throughput benchmark (n={n :,}) …")

    entity_ids = [f"bench_user_{i :05d}" for i in range(100)]

    t0 = time.perf_counter()
    for i in range(n):
        store.ingest(
            entity_id=random.choice(entity_ids),
            amount=random.uniform(5, 500),
            lat=random.uniform(35, 45),
            lon=random.uniform(-100, -70),
            merchant=f"merch_{random .randint (0 ,50 )}",
            merchant_category=random.choice(["grocery", "restaurant", "online", "atm"]),
        )
    elapsed = time.perf_counter() - t0
    rps = n / elapsed

    print(f"  Ingested {n :,} events in {elapsed :.2f}s")
    print(f"  Throughput: {rps :,.0f} events/sec")
    print(f"  Avg per event: {elapsed *1000 /n :.3f} ms")


def bench_feature_read(store: FeatureStore, n: int = 5_000) -> None:
    print(f"\n[2/3] Feature read latency benchmark (n={n :,}) …")

    entity_ids = [f"bench_user_{i :05d}" for i in range(50)]
    for eid in entity_ids:
        for _ in range(10):
            store.ingest(eid, random.uniform(10, 200), 40.7, -74.0, "test_merch")

    latencies_us = []
    for _ in range(n):
        eid = random.choice(entity_ids)
        t0 = time.perf_counter()
        _ = store.get_features(eid, current_amount=random.uniform(10, 200))
        latencies_us.append((time.perf_counter() - t0) * 1_000_000)

    print_latency_report("Feature retrieval (L1 cache)", latencies_us)

    store.clear_cache()
    latencies_l2 = []
    for _ in range(min(n, 1000)):
        eid = random.choice(entity_ids)
        t0 = time.perf_counter()
        _ = store.get_features(eid, current_amount=random.uniform(10, 200))
        latencies_l2.append((time.perf_counter() - t0) * 1_000_000)

    label = (
        "Feature retrieval (L2 Redis)"
        if store._l2.available
        else "Feature retrieval (live compute)"
    )
    print_latency_report(label, latencies_l2)


def bench_inference_inprocess(store: FeatureStore, n: int = 5_000) -> None:
    print(f"\n[3/3] In-process inference latency benchmark (n={n :,}) …")

    model_path = "models/fraud_model.pkl"
    if not os.path.exists(model_path):
        print("  ⚠ Model not found — run `python scripts/train_model.py` first.")
        return

    from inference_engine.server import FraudModel, InferenceServiceCore

    model = FraudModel(model_path)
    core = InferenceServiceCore(store, model)

    entity_ids = [f"bench_user_{i :05d}" for i in range(50)]

    for _ in range(50):
        core.predict_single(
            entity_id=random.choice(entity_ids),
            transaction_id="warmup",
            amount=100.0,
            lat=40.7,
            lon=-74.0,
            merchant_category="grocery",
        )

    latencies_us = []
    for _ in range(n):
        t0 = time.perf_counter()
        core.predict_single(
            entity_id=random.choice(entity_ids),
            transaction_id=f"txn_{random .randint (0 ,1_000_000 )}",
            amount=random.uniform(5, 1000),
            lat=random.uniform(35, 45),
            lon=random.uniform(-100, -70),
            merchant_category=random.choice(["grocery", "restaurant", "online", "atm"]),
        )
        latencies_us.append((time.perf_counter() - t0) * 1_000_000)

    print_latency_report("End-to-end inference (in-process)", latencies_us)

    print("\n  Concurrent throughput (16 threads) …")
    latencies_conc = []
    lock = __import__("threading").Lock()

    def _predict(_):
        t0 = time.perf_counter()
        core.predict_single(
            entity_id=random.choice(entity_ids),
            transaction_id="bench",
            amount=random.uniform(5, 1000),
            lat=40.7,
            lon=-74.0,
        )
        with lock:
            latencies_conc.append((time.perf_counter() - t0) * 1_000_000)

    t_conc = time.perf_counter()
    with ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(_predict, range(n)))
    elapsed_conc = time.perf_counter() - t_conc

    print_latency_report("End-to-end inference (16 threads concurrent)", latencies_conc)
    print(f"  Concurrent RPS: {n /elapsed_conc :,.0f}")


def bench_grpc(host: str = "localhost", port: int = 50051, n: int = 1_000) -> None:
    print(f"\n[gRPC] Round-trip benchmark → {host }:{port }  (n={n :,}) …")
    try:
        import grpc

        import inference_pb2
        import inference_pb2_grpc
    except ImportError:
        print("  ⚠ gRPC stubs not compiled. Run `python scripts/compile_proto.py`.")
        return

    channel = grpc.insecure_channel(
        f"{host }:{port }",
        options=[
            ("grpc.keepalive_time_ms", 5000),
            ("grpc.keepalive_timeout_ms", 2000),
            ("grpc.max_receive_message_length", 16 * 1024 * 1024),
        ],
    )
    stub = inference_pb2_grpc.InferenceServiceStub(channel)

    try:
        stub.Health(inference_pb2.HealthRequest())
    except Exception as exc:
        print(f"  ✗ Cannot reach server: {exc }")
        return

    latencies_us = []
    for i in range(n):
        req = inference_pb2.InferenceRequest(
            request_id=str(i),
            transaction=inference_pb2.Transaction(
                transaction_id=f"grpc_bench_{i }",
                entity_id=f"bench_user_{i %50 :05d}",
                amount=random.uniform(5, 1000),
                merchant_category="online",
                latitude=40.7 + random.gauss(0, 0.1),
                longitude=-74.0 + random.gauss(0, 0.1),
                timestamp_ms=int(time.time() * 1000),
            ),
        )
        t0 = time.perf_counter()
        stub.Predict(req)
        latencies_us.append((time.perf_counter() - t0) * 1_000_000)

    print_latency_report("gRPC round-trip (single thread)", latencies_us)
    channel.close()


def print_summary(store: FeatureStore) -> None:
    health = store.health()
    print(f"\n{'═'*50 }")
    print("  FEATURE STORE HEALTH")
    print(f"{'═'*50 }")
    for k, v in health.items():
        print(f"  {k :<30} {v }")
    print(f"{'═'*50 }\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Feature store & inference benchmark")
    parser.add_argument("--n", type=int, default=10_000, help="Sample count")
    parser.add_argument("--grpc", action="store_true", help="Also benchmark gRPC endpoint")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=50051)
    args = parser.parse_args()

    store = FeatureStore()

    bench_ingestion(store, n=args.n)
    bench_feature_read(store, n=min(args.n, 5_000))
    bench_inference_inprocess(store, n=min(args.n, 5_000))

    if args.grpc:
        bench_grpc(args.host, args.port, n=min(args.n, 1_000))

    print_summary(store)
