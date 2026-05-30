from __future__ import annotations

import asyncio
import json
import os
import random
import sys
import time
from collections import deque
from typing import Deque

from aiohttp import web

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class MetricBuffer:
    """Circular buffer of time-series data points."""

    def __init__(self, maxlen: int = 60):
        self.times: Deque[str] = deque(maxlen=maxlen)
        self.values: Deque[float] = deque(maxlen=maxlen)

    def push(self, value: float) -> None:
        self.times.append(time.strftime("%H:%M:%S"))
        self.values.append(round(value, 3))

    def to_dict(self) -> dict:
        return {"labels": list(self.times), "data": list(self.values)}


class DashboardState:
    def __init__(self):
        self.ingestion_rps = MetricBuffer()
        self.inference_p50 = MetricBuffer()
        self.inference_p99 = MetricBuffer()
        self.cache_hit_rate = MetricBuffer()
        self.fraud_rate = MetricBuffer()
        self.entity_count = MetricBuffer()
        self.feature_p50 = MetricBuffer()
        self.decisions = {"APPROVE": 0, "REVIEW": 0, "DECLINE": 0}
        self._tick = 0
        self._total_requests = 0

    def tick(self) -> None:
        """Advance simulation by one second."""
        t = self._tick
        self._tick += 1

        base_rps = 6500 + 1500 * abs(math.sin(t / 20))
        noise = lambda s: random.gauss(0, s)

        rps = max(0, base_rps + noise(400))
        self.ingestion_rps.push(rps)

        p50 = 0.145 + noise(0.015) + (5 if random.random() < 0.03 else 0)
        p99 = 0.233 + noise(0.04) + (20 if random.random() < 0.02 else 0)
        self.inference_p50.push(max(0.05, p50))
        self.inference_p99.push(max(0.1, p99))

        fp50 = 0.001 + abs(noise(0.0005))
        self.feature_p50.push(fp50)

        chr_pct = min(100, max(90, 99.7 + noise(0.3)))
        self.cache_hit_rate.push(chr_pct)

        fr = max(0, min(25, 10.0 + noise(1.5)))
        self.fraud_rate.push(fr)

        ec = 100 + t * 2 + int(noise(5))
        self.entity_count.push(max(0, ec))

        n_req = int(rps)
        self._total_requests += n_req
        fr_frac = fr / 100
        self.decisions["APPROVE"] += int(n_req * (1 - fr_frac) * 0.85)
        self.decisions["REVIEW"] += int(n_req * (1 - fr_frac) * 0.15)
        self.decisions["DECLINE"] += int(n_req * fr_frac)

    def to_json(self) -> dict:
        return {
            "ingestion_rps": self.ingestion_rps.to_dict(),
            "inference_p50": self.inference_p50.to_dict(),
            "inference_p99": self.inference_p99.to_dict(),
            "feature_p50": self.feature_p50.to_dict(),
            "cache_hit_rate": self.cache_hit_rate.to_dict(),
            "fraud_rate": self.fraud_rate.to_dict(),
            "entity_count": self.entity_count.to_dict(),
            "decisions": self.decisions,
            "total_requests": self._total_requests,
            "uptime_seconds": self._tick,
        }


import math

state = DashboardState()


async def handle_metrics(request: web.Request) -> web.Response:
    return web.json_response(state.to_json())


async def handle_index(request: web.Request) -> web.Response:
    html_path = os.path.join(os.path.dirname(__file__), "index.html")
    with open(html_path) as f:
        return web.Response(text=f.read(), content_type="text/html")


async def metric_ticker(app: web.Application) -> None:
    """Background task that advances simulated metrics every second."""
    while True:
        state.tick()
        await asyncio.sleep(1)


async def on_startup(app: web.Application) -> None:
    app["ticker"] = asyncio.create_task(metric_ticker(app))


async def on_cleanup(app: web.Application) -> None:
    app["ticker"].cancel()
    try:
        await app["ticker"]
    except asyncio.CancelledError:
        pass


def make_app() -> web.Application:
    app = web.Application()
    app.router.add_get("/", handle_index)
    app.router.add_get("/metrics", handle_metrics)
    app.on_startup.append(on_startup)
    app.on_cleanup.append(on_cleanup)
    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8888)
    args = parser.parse_args()

    print(f"\n  Dashboard → http://localhost:{args .port }\n")
    web.run_app(make_app(), port=args.port, print=lambda *_: None)
