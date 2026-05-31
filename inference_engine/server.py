from __future__ import annotations

import asyncio
import logging
import os
import pickle
import signal
import threading
import time
import uuid
from concurrent import futures
from typing import AsyncIterator, Iterator, Optional

import grpc
import numpy as np

try:
    from prometheus_client import Counter, Gauge, Histogram
    from prometheus_client import start_http_server as prom_start

    PROM_AVAILABLE = True
except ImportError:
    PROM_AVAILABLE = False

logger = logging.getLogger(__name__)


if PROM_AVAILABLE:
    INFERENCE_LATENCY = Histogram(
        "inference_latency_seconds",
        "End-to-end gRPC inference latency",
        buckets=[0.001, 0.002, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5],
    )
    FEATURE_LATENCY = Histogram(
        "feature_latency_seconds",
        "Feature retrieval latency",
        buckets=[0.0001, 0.0005, 0.001, 0.002, 0.005, 0.01],
    )
    REQUESTS_TOTAL = Counter("inference_requests_total", "Total inference requests", ["decision"])
    ERRORS_TOTAL = Counter("inference_errors_total", "Total inference errors")
    FRAUD_SCORE_HIST = Histogram(
        "fraud_score",
        "Distribution of fraud scores",
        buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    )
    ACTIVE_REQUESTS = Gauge("active_requests", "In-flight gRPC requests")


class FraudModel:
    """Thin wrapper around the pickled model artifact."""

    DECISION_THRESHOLDS = {
        "approve": 0.2,
        "review": 0.5,
    }

    def __init__(self, model_path: str = "models/fraud_model.pkl"):
        self._path = model_path
        self._load()

    def _load(self) -> None:
        with open(self._path, "rb") as f:
            payload = pickle.load(f)
        self._model = payload["model"]
        self._scaler = payload.get("scaler")
        self._feature_names = payload["feature_names"]
        self._model_type = payload["model_type"]
        self._version = payload["version"]
        self._threshold = payload.get("threshold", 0.5)
        logger.info("Loaded model v%s (%s)", self._version, self._model_type)

    def reload(self) -> None:
        """Hot-reload the model (call on SIGHUP)."""
        self._load()
        logger.info("Model reloaded from %s", self._path)

    def predict(self, feature_array: np.ndarray) -> tuple[float, str, float]:
        """
        Returns (fraud_probability, decision, confidence).
        decision ∈ {APPROVE, REVIEW, DECLINE}
        """
        if self._scaler is not None and self._model_type != "xgboost":
            feature_array = self._scaler.transform(feature_array.reshape(1, -1))

        prob = float(self._model.predict_proba(feature_array.reshape(1, -1))[0, 1])

        if prob < self.DECISION_THRESHOLDS["approve"]:
            decision = "APPROVE"
            confidence = 1.0 - prob
        elif prob < self.DECISION_THRESHOLDS["review"]:
            decision = "REVIEW"
            confidence = 0.5
        else:
            decision = "DECLINE"
            confidence = prob

        return prob, decision, confidence

    def explain(self, feature_array: np.ndarray) -> dict[str, float]:
        """Return feature importances (XGBoost only; others return empty)."""
        if self._model_type != "xgboost":
            return {}
        importances = self._model.feature_importances_
        return {
            name: float(imp * abs(val))
            for name, imp, val in zip(self._feature_names, importances, feature_array)
        }

    @property
    def version(self) -> str:
        return str(self._version)


class InferenceServiceCore:
    """
    Business logic for inference, decoupled from gRPC transport.
    Can be tested independently and called from HTTP adapters too.
    """

    def __init__(self, feature_store, model: FraudModel):
        self._store = feature_store
        self._model = model
        self._request_count = 0
        self._count_lock = threading.Lock()

    def predict_single(
        self,
        entity_id: str,
        transaction_id: str,
        amount: float,
        lat: float = 0.0,
        lon: float = 0.0,
        merchant_category: str = "other",
        include_features: bool = False,
        explain: bool = False,
    ) -> dict:
        """Perform a single inference and return a result dict."""
        t_start = time.perf_counter()

        t_feat_start = time.perf_counter()
        snap = self._store.get_features(
            entity_id=entity_id,
            current_amount=amount,
            current_lat=lat,
            current_lon=lon,
            merchant_category=merchant_category,
        )
        feat_latency_us = int((time.perf_counter() - t_feat_start) * 1_000_000)

        feature_array = snap.to_model_array()
        fraud_prob, decision, confidence = self._model.predict(feature_array)

        explanations = {}
        if explain:
            explanations = self._model.explain(feature_array)

        total_latency_us = int((time.perf_counter() - t_start) * 1_000_000)
        with self._count_lock:
            self._request_count += 1

        if PROM_AVAILABLE:
            INFERENCE_LATENCY.observe(total_latency_us / 1_000_000)
            FEATURE_LATENCY.observe(feat_latency_us / 1_000_000)
            REQUESTS_TOTAL.labels(decision=decision).inc()
            FRAUD_SCORE_HIST.observe(fraud_prob)

        return {
            "transaction_id": transaction_id,
            "fraud_probability": fraud_prob,
            "is_fraud": fraud_prob > self._model._threshold,
            "decision": decision,
            "confidence": confidence,
            "inference_latency_us": total_latency_us,
            "feature_latency_us": feat_latency_us,
            "model_version": self._model.version,
            "features": snap.to_dict() if include_features else None,
            "explanations": explanations,
        }

    def batch_predict(self, requests: list[dict]) -> list[dict]:
        """Run multiple predictions (can be parallelised with ThreadPoolExecutor)."""
        return [self.predict_single(**r) for r in requests]

    @property
    def request_count(self) -> int:
        return self._request_count


def build_grpc_server(
    feature_store,
    model: FraudModel,
    host: str = "[::]",
    port: int = 50051,
    max_workers: int = 16,
) -> Optional[grpc.Server]:
    """
    Build and return a gRPC server.
    Returns None if generated stubs are not available (run compile_proto.py first).
    """
    try:

        import inference_pb2
        import inference_pb2_grpc

        core = InferenceServiceCore(feature_store, model)

        class InferenceServicer(inference_pb2_grpc.InferenceServiceServicer):

            def Predict(self, request, context):
                if PROM_AVAILABLE:
                    ACTIVE_REQUESTS.inc()
                try:
                    result = core.predict_single(
                        entity_id=request.transaction.entity_id,
                        transaction_id=request.transaction.transaction_id,
                        amount=request.transaction.amount,
                        lat=request.transaction.latitude,
                        lon=request.transaction.longitude,
                        merchant_category=request.transaction.merchant_category,
                        include_features=request.include_features,
                        explain=request.explain,
                    )
                    resp = inference_pb2.InferenceResponse(
                        request_id=request.request_id,
                        transaction_id=result["transaction_id"],
                        fraud_probability=result["fraud_probability"],
                        is_fraud=result["is_fraud"],
                        confidence=result["confidence"],
                        decision=result["decision"],
                        inference_latency_us=result["inference_latency_us"],
                        feature_latency_us=result["feature_latency_us"],
                        model_version=result["model_version"],
                    )
                    if result.get("features"):
                        feat_dict = result["features"].copy()
                        feat_dict["computed_at_ms"] = int(feat_dict.pop("computed_at", 0) * 1000)
                        resp.features.CopyFrom(inference_pb2.FeatureVector(**feat_dict))
                    if result.get("explanations"):
                        resp.explanations.update(result["explanations"])
                    return resp
                except Exception as exc:
                    if PROM_AVAILABLE:
                        ERRORS_TOTAL.inc()
                    context.abort(grpc.StatusCode.INTERNAL, str(exc))
                finally:
                    if PROM_AVAILABLE:
                        ACTIVE_REQUESTS.dec()

            def Health(self, request, context):
                h = feature_store.health()
                return inference_pb2.HealthResponse(
                    healthy=True,
                    version=model.version,
                    uptime_seconds=h.get("uptime_seconds", 0),
                    requests_served=core.request_count,
                    cache_hit_rate=h.get("l1_cache_hit_rate", 0.0),
                )

            def BatchPredict(self, request, context):
                requests = [
                    dict(
                        entity_id=r.transaction.entity_id,
                        transaction_id=r.transaction.transaction_id,
                        amount=r.transaction.amount,
                        lat=r.transaction.latitude,
                        lon=r.transaction.longitude,
                        merchant_category=r.transaction.merchant_category,
                        include_features=r.include_features,
                        explain=r.explain,
                    )
                    for r in request.requests
                ]
                t0 = time.perf_counter()
                results = core.batch_predict(requests)
                total = int((time.perf_counter() - t0) * 1_000_000)

                responses = []
                for r in results:
                    resp = inference_pb2.InferenceResponse(
                        transaction_id=r["transaction_id"],
                        fraud_probability=r["fraud_probability"],
                        is_fraud=r["is_fraud"],
                        decision=r["decision"],
                        confidence=r["confidence"],
                        inference_latency_us=r["inference_latency_us"],
                        feature_latency_us=r.get("feature_latency_us", 0),
                        model_version=r.get("model_version", ""),
                    )
                    if r.get("features"):
                        feat_dict = r["features"].copy()
                        feat_dict["computed_at_ms"] = int(feat_dict.pop("computed_at", 0) * 1000)
                        resp.features.CopyFrom(inference_pb2.FeatureVector(**feat_dict))
                    if r.get("explanations"):
                        resp.explanations.update(r["explanations"])
                    responses.append(resp)

                return inference_pb2.BatchInferenceResponse(
                    responses=responses,
                    total_latency_us=total,
                    succeeded=len(results),
                    failed=0,
                )

            def StreamPredict(self, request_iterator, context):
                for transaction in request_iterator:
                    if PROM_AVAILABLE:
                        ACTIVE_REQUESTS.inc()
                    try:
                        result = core.predict_single(
                            entity_id=transaction.entity_id,
                            transaction_id=transaction.transaction_id,
                            amount=transaction.amount,
                            lat=transaction.latitude,
                            lon=transaction.longitude,
                            merchant_category=transaction.merchant_category,
                            include_features=False,
                            explain=False,
                        )
                        yield inference_pb2.InferenceResponse(
                            transaction_id=result["transaction_id"],
                            fraud_probability=result["fraud_probability"],
                            is_fraud=result["is_fraud"],
                            confidence=result["confidence"],
                            decision=result["decision"],
                            inference_latency_us=result["inference_latency_us"],
                            feature_latency_us=result["feature_latency_us"],
                            model_version=result["model_version"],
                        )
                    except Exception as exc:
                        if PROM_AVAILABLE:
                            ERRORS_TOTAL.inc()
                        logger.error("StreamPredict error: %s", exc)
                    finally:
                        if PROM_AVAILABLE:
                            ACTIVE_REQUESTS.dec()

            def GetFeatures(self, request, context):
                try:
                    snap = feature_store.get_features(request.entity_id)
                    feat_dict = snap.to_dict()
                    feat_dict["computed_at_ms"] = int(feat_dict.pop("computed_at", 0) * 1000)
                    if request.fields:
                        feat_dict = {
                            k: v
                            for k, v in feat_dict.items()
                            if k in request.fields or k == "entity_id"
                        }
                    return inference_pb2.FeatureVector(**feat_dict)
                except Exception as exc:
                    context.abort(grpc.StatusCode.INTERNAL, str(exc))

        server = grpc.server(
            futures.ThreadPoolExecutor(max_workers=max_workers),
            options=[
                ("grpc.max_send_message_length", 16 * 1024 * 1024),
                ("grpc.max_receive_message_length", 16 * 1024 * 1024),
                ("grpc.keepalive_time_ms", 10_000),
                ("grpc.keepalive_timeout_ms", 5_000),
                ("grpc.keepalive_permit_without_calls", True),
                ("grpc.http2.max_pings_without_data", 0),
            ],
        )
        inference_pb2_grpc.add_InferenceServiceServicer_to_server(InferenceServicer(), server)
        server.add_insecure_port(f"{host }:{port }")
        return server

    except ImportError:
        logger.warning(
            "Generated gRPC stubs not found. "
            "Run `python scripts/compile_proto.py` to generate them, "
            "then restart the server."
        )
        return None


def main() -> None:
    import sys

    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

    from feature_store.store import FeatureStore

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    model_path = os.environ.get("MODEL_PATH", "models/fraud_model.pkl")
    grpc_port = int(os.environ.get("GRPC_PORT", 50051))
    prom_port = int(os.environ.get("PROM_PORT", 8000))

    if not os.path.exists(model_path):
        logger.error(
            "Model not found at %s — run `python scripts/train_model.py` first.", model_path
        )
        return

    store = FeatureStore(
        redis_host=os.environ.get("REDIS_HOST", "127.0.0.1"),
        redis_port=int(os.environ.get("REDIS_PORT", 6379)),
    )
    model = FraudModel(model_path)
    server = build_grpc_server(store, model, port=grpc_port)

    if server is None:
        logger.error("Could not start gRPC server (see warnings above).")
        return

    if PROM_AVAILABLE:
        prom_start(prom_port)
        logger.info("Prometheus metrics at http://127.0.0.1:%d/metrics", prom_port)

    def _reload_model(*_):
        model.reload()

    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, _reload_model)

    server.start()
    logger.info("gRPC inference server listening on port %d", grpc_port)
    logger.info("Send SIGHUP to PID %d to hot-reload the model.", os.getpid())

    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("Shutting down gRPC server …")
        server.stop(grace=5)


if __name__ == "__main__":
    main()
