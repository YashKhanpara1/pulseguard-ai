# PulseGuard AI — Real-Time Log & API Anomaly Detector with Drift Monitoring

**Live Dashboard:** https://pulseguard-ai-b7sw.onrender.com  
**Interactive API Docs (Swagger):** https://pulseguard-ai-b7sw.onrender.com/docs  

PulseGuard AI is an asynchronous, real-time telemetry observability and machine learning monitoring engine built with **FastAPI**, **Scikit-Learn**, **SciPy**, **Upstash Redis**, and **Neon PostgreSQL**. It ingests live microservice API metrics (`latency_ms`, `status_code`, `payload_kb`, `cpu_usage_pct`), detects sudden point-level anomalies using an **Isolation Forest** model, identifies gradual data distribution shifts using two-sample **Kolmogorov–Smirnov (KS) statistical tests**, and streams live metrics and incident alerts to an interactive dashboard via **WebSockets**.

---

## Why PulseGuard AI?

Traditional rule-based API monitoring only triggers alerts after static thresholds are breached (e.g., `HTTP 500` spikes), missing two critical production failure modes:
1. **Multivariate Point Anomalies:** Subtle combinations of elevated latency, abnormal payload sizes, and CPU saturation that precede service crashes.
2. **Silent Data & Performance Drift:** Gradual latency creep (e.g., `100 ms` drifting to `300 ms`) where individual requests still return `HTTP 200` and stay near single-point thresholds, yet the overall traffic distribution has degraded.

PulseGuard AI solves both problems simultaneously by combining unsupervised ML outlier scoring with statistical distribution testing over a sliding memory window.

---

## System Architecture

```text
Microservice Telemetry (POST /api/v1/ingest)
                    │
                    ▼
   ┌─────────────────────────────────┐
   │   FastAPI Async Ingestion API   │
   └────────────────┬────────────────┘
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
┌───────────────┐       ┌───────────────┐
│ Isolation     │       │ Upstash Redis │
│ Forest Model  │       │ Sliding Window│
│ (Point Score) │       │ (Last N Events│
└───────┬───────┘       └───────┬───────┘
        │                       │
        │                       ▼
        │               ┌───────────────┐
        │               │ SciPy KS-Test │
        │               │ Drift Engine  │
        │               └───────┬───────┘
        ▼                       │
┌───────────────────┐           │
│ Neon PostgreSQL   │           │
│ IncidentAuditLog  │           │
│ (Anomalies Only)  │           │
└─────────┬─────────┘           │
          │                     │
          └──────────┬──────────┘
                     ▼
        ┌─────────────────────────┐
        │ WebSocket / Polling UI  │
        │ (Live Chart & Alerts)   │
        └─────────────────────────┘
