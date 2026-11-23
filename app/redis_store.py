import json
from typing import Any
import numpy as np
from upstash_redis.asyncio import Redis
from app.config import settings


WINDOW_KEY = "pulseguard:telemetry_window"
STATS_KEY = "pulseguard:counters"
MAX_WINDOW_SIZE = 200


class RedisTelemetryStore:
    def __init__(self) -> None:
        self.client = Redis(
            url=settings.upstash_redis_rest_url,
            token=settings.upstash_redis_rest_token,
        )

    async def push_event(self, event: dict[str, Any]) -> None:
        pipe = self.client.pipeline()
        pipe.lpush(WINDOW_KEY, json.dumps(event))
        pipe.ltrim(WINDOW_KEY, 0, MAX_WINDOW_SIZE - 1)
        pipe.hincrby(STATS_KEY, "total_events", 1)
        if event.get("is_anomaly"):
            pipe.hincrby(STATS_KEY, "total_anomalies", 1)
        await pipe.exec()

    async def get_window(self, limit: int = MAX_WINDOW_SIZE) -> list[dict[str, Any]]:
        raw_items = await self.client.lrange(WINDOW_KEY, 0, limit - 1)
        events = []
        for item in raw_items:
            if isinstance(item, str):
                events.append(json.loads(item))
            elif isinstance(item, dict):
                events.append(item)
        # Reverse so oldest -> newest for time-series charting
        return list(reversed(events))

    async def get_summary_metrics(self) -> dict[str, Any]:
        window = await self.get_window()
        counters = await self.client.hgetall(STATS_KEY) or {}
        total_events = int(counters.get("total_events", 0))
        total_anomalies = int(counters.get("total_anomalies", 0))

        if not window:
            return {
                "total_events": total_events,
                "total_anomalies": total_anomalies,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "error_rate_pct": 0.0,
            }

        latencies = [float(e["latency_ms"]) for e in window]
        errors = sum(1 for e in window if int(e["status_code"]) >= 400)

        return {
            "total_events": total_events,
            "total_anomalies": total_anomalies,
            "avg_latency_ms": round(float(np.mean(latencies)), 2),
            "p95_latency_ms": round(float(np.percentile(latencies, 95)), 2),
            "error_rate_pct": round((errors / len(window)) * 100, 2),
        }

    async def reset_window(self) -> None:
        await self.client.delete(WINDOW_KEY)