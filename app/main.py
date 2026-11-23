import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal
import numpy as np
from fastapi import BackgroundTasks, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from app.alerts import dispatch_webhook_alert
from app.database import AsyncSessionLocal, IncidentLog, engine, get_recent_incidents, init_db
from app.ml_engine import TelemetryMLEngine
from app.redis_store import RedisTelemetryStore


ml_engine: TelemetryMLEngine | None = None
redis_store: RedisTelemetryStore | None = None


class ConnectionManager:
    def __init__(self) -> None:
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict) -> None:
        dead: list[WebSocket] = []
        for ws in self.active_connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


ws_manager = ConnectionManager()


class TelemetryIn(BaseModel):
    service_name: str = Field(default="payment-gateway", max_length=80)
    endpoint: str = Field(default="/api/v1/checkout", max_length=120)
    latency_ms: float = Field(..., gt=0, le=60000)
    status_code: int = Field(default=200, ge=100, le=599)
    payload_kb: float = Field(default=12.5, ge=0.1, le=50000)
    cpu_usage_pct: float = Field(default=40.0, ge=0.0, le=100.0)


class SimulateRequest(BaseModel):
    mode: Literal["normal", "anomaly", "drift", "reset"] = "normal"
    count: int = Field(default=1, ge=1, le=30)


async def process_telemetry(payload: TelemetryIn, background_tasks: BackgroundTasks | None = None) -> dict:
    assert ml_engine is not None and redis_store is not None

    prediction = ml_engine.predict_anomaly(
        latency_ms=payload.latency_ms,
        status_code=payload.status_code,
        payload_kb=payload.payload_kb,
        cpu_usage_pct=payload.cpu_usage_pct,
    )

    event_record = {
        "timestamp": datetime.now(timezone.utc).strftime("%H:%M:%S"),
        "service_name": payload.service_name,
        "endpoint": payload.endpoint,
        "latency_ms": round(payload.latency_ms, 2),
        "status_code": payload.status_code,
        "payload_kb": round(payload.payload_kb, 2),
        "cpu_usage_pct": round(payload.cpu_usage_pct, 2),
        "is_anomaly": prediction["is_anomaly"],
        "risk_score": prediction["risk_score"],
        "reason": prediction["reason"],
    }

    await redis_store.push_event(event_record)

    window = await redis_store.get_window()
    feature_matrix = [
        [e["latency_ms"], e["status_code"], e["payload_kb"], e["cpu_usage_pct"]]
        for e in window
    ]
    drift_report = ml_engine.compute_drift(feature_matrix)

    # Persist anomalies or severe drift events to Neon PostgreSQL
    if prediction["is_anomaly"]:
        async with AsyncSessionLocal() as session:
            session.add(
                IncidentLog(
                    service_name=payload.service_name,
                    endpoint=payload.endpoint,
                    incident_type="ANOMALY",
                    latency_ms=round(payload.latency_ms, 2),
                    status_code=payload.status_code,
                    risk_score=prediction["risk_score"],
                    reason=prediction["reason"],
                )
            )
            await session.commit()

        if background_tasks:
            background_tasks.add_task(
                dispatch_webhook_alert,
                f"Anomaly in {payload.service_name}",
                f"Endpoint `{payload.endpoint}` | Latency: `{payload.latency_ms:.1f}ms` | Status: `{payload.status_code}` | Reason: {prediction['reason']}",
            )

    result = {
        "event": event_record,
        "drift": drift_report,
    }
    await ws_manager.broadcast(result)
    return result


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_engine, redis_store
    ml_engine = TelemetryMLEngine()
    redis_store = RedisTelemetryStore()
    await init_db()

    # Seed baseline window if Redis is empty so dashboard starts populated
    existing = await redis_store.get_window(limit=5)
    if not existing:
        rng = np.random.default_rng(42)
        for _ in range(25):
            sample = TelemetryIn(
                service_name="payment-gateway",
                endpoint="/api/v1/checkout",
                latency_ms=float(np.clip(rng.lognormal(4.65, 0.25), 40, 260)),
                status_code=200,
                payload_kb=float(np.clip(rng.gamma(3.0, 6.0), 2, 45)),
                cpu_usage_pct=float(np.clip(rng.normal(42, 8), 18, 65)),
            )
            await process_telemetry(sample)

    yield
    await engine.dispose()


app = FastAPI(
    title="PulseGuard AI — Real-Time Log & API Anomaly Detector",
    description="Async FastAPI telemetry pipeline with Isolation Forest anomaly detection, Redis sliding windows, KS-test drift monitoring, and PostgreSQL persistence.",
    version="1.0.0",
    lifespan=lifespan,
)

STATIC_DIR = Path("static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def serve_dashboard():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": ml_engine is not None}


@app.post("/api/v1/ingest")
async def ingest_telemetry(payload: TelemetryIn, background_tasks: BackgroundTasks):
    return await process_telemetry(payload, background_tasks)


@app.get("/api/v1/overview")
async def get_overview():
    assert ml_engine is not None and redis_store is not None
    metrics = await redis_store.get_summary_metrics()
    window = await redis_store.get_window()
    feature_matrix = [
        [e["latency_ms"], e["status_code"], e["payload_kb"], e["cpu_usage_pct"]]
        for e in window
    ]
    drift = ml_engine.compute_drift(feature_matrix)
    incidents = await get_recent_incidents(limit=12)

    return {
        "metrics": metrics,
        "drift": drift,
        "window": window[-60:],
        "incidents": incidents,
    }


@app.post("/api/v1/simulate")
async def simulate_traffic(req: SimulateRequest, background_tasks: BackgroundTasks):
    assert redis_store is not None
    rng = np.random.default_rng()

    if req.mode == "reset":
        await redis_store.reset_window()
        for _ in range(25):
            sample = TelemetryIn(
                service_name="payment-gateway",
                endpoint="/api/v1/checkout",
                latency_ms=float(np.clip(rng.lognormal(4.65, 0.25), 40, 260)),
                status_code=200,
                payload_kb=float(np.clip(rng.gamma(3.0, 6.0), 2, 45)),
                cpu_usage_pct=float(np.clip(rng.normal(42, 8), 18, 65)),
            )
            await process_telemetry(sample)
        return {"status": "reset_complete"}

    last_res = None
    for _ in range(req.count):
        if req.mode == "normal":
            sample = TelemetryIn(
                service_name="payment-gateway",
                endpoint="/api/v1/checkout",
                latency_ms=float(np.clip(rng.lognormal(4.65, 0.25), 45, 280)),
                status_code=200,
                payload_kb=float(np.clip(rng.gamma(3.0, 6.0), 2, 50)),
                cpu_usage_pct=float(np.clip(rng.normal(43, 8), 20, 68)),
            )
        elif req.mode == "anomaly":
            sample = TelemetryIn(
                service_name="payment-gateway",
                endpoint="/api/v1/settle",
                latency_ms=float(rng.uniform(780, 1850)),
                status_code=int(rng.choice([500, 502, 503, 200], p=[0.4, 0.2, 0.2, 0.2])),
                payload_kb=float(rng.uniform(140, 320)),
                cpu_usage_pct=float(rng.uniform(86, 98)),
            )
        else:  # "drift" -> gradual distribution shift (moderate higher latency & payload without 5xx crashes)
            sample = TelemetryIn(
                service_name="recommendation-api",
                endpoint="/api/v1/rank",
                latency_ms=float(rng.uniform(260, 340)),
                status_code=200,
                payload_kb=float(rng.uniform(65, 110)),
                cpu_usage_pct=float(rng.uniform(66, 78)),
            )
        last_res = await process_telemetry(sample, background_tasks)
        await asyncio.sleep(0.03)

    return {"status": "simulated", "mode": req.mode, "count": req.count, "latest": last_res}


@app.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)