from pathlib import Path
from typing import Any
import joblib
import numpy as np
from scipy.stats import ks_2samp


BUNDLE_PATH = Path("artifacts/model_bundle.joblib")


class TelemetryMLEngine:
    def __init__(self, bundle_path: Path = BUNDLE_PATH) -> None:
        if not bundle_path.exists():
            raise FileNotFoundError(
                f"Model bundle not found at {bundle_path}. Run 'python scripts/train_model.py' first."
            )
        bundle = joblib.load(bundle_path)
        self.model = bundle["model"]
        self.reference_data: np.ndarray = bundle["reference_data"]
        self.feature_names: list[str] = bundle["feature_names"]

    def predict_anomaly(
        self,
        latency_ms: float,
        status_code: int,
        payload_kb: float,
        cpu_usage_pct: float,
    ) -> dict[str, Any]:
        features = np.array([[latency_ms, float(status_code), payload_kb, cpu_usage_pct]])
        pred = int(self.model.predict(features)[0])  # -1 = anomaly, 1 = normal
        raw_score = float(self.model.decision_function(features)[0])

        # Convert decision_function into a 0.0 to 1.0 risk score (higher = more anomalous)
        risk_score = float(np.clip(0.5 - raw_score, 0.0, 1.0))

        reasons: list[str] = []
        if status_code >= 500:
            reasons.append(f"HTTP {status_code} server failure")
        if latency_ms > 350:
            reasons.append(f"High latency ({latency_ms:.1f} ms)")
        if cpu_usage_pct > 80:
            reasons.append(f"CPU saturation ({cpu_usage_pct:.1f}%)")
        if payload_kb > 120:
            reasons.append(f"Oversized payload ({payload_kb:.1f} KB)")

        # Hard guardrail: any 5xx or extreme latency is always flagged
        is_anomaly = (pred == -1) or (status_code >= 500) or (latency_ms > 600)
        if is_anomaly and not reasons:
            reasons.append("Multivariate distribution outlier")

        return {
            "is_anomaly": is_anomaly,
            "risk_score": round(risk_score, 4),
            "reason": ", ".join(reasons) if is_anomaly else "Normal telemetry",
        }

    def compute_drift(self, live_rows: list[list[float]]) -> dict[str, Any]:
        """Compare live sliding window against training baseline using KS-test."""
        if len(live_rows) < 20:
            return {
                "drift_detected": False,
                "drift_score": 0.0,
                "drifted_features": [],
                "sample_count": len(live_rows),
                "status": "Warming up (need >= 20 events)",
            }

        live_arr = np.array(live_rows, dtype=float)
        # Check continuous features: latency_ms (0), payload_kb (2), cpu_usage_pct (3)
        monitored_indices = [0, 2, 3]
        ks_stats: list[float] = []
        drifted_features: list[str] = []

        for idx in monitored_indices:
            stat, p_value = ks_2samp(self.reference_data[:, idx], live_arr[:, idx])
            ks_stats.append(float(stat))
            if stat > 0.35 and p_value < 0.01:
                drifted_features.append(self.feature_names[idx])

        max_ks = round(max(ks_stats), 4)
        drift_detected = len(drifted_features) > 0

        return {
            "drift_detected": drift_detected,
            "drift_score": max_ks,
            "drifted_features": drifted_features,
            "sample_count": len(live_rows),
            "status": f"DRIFT ALERT ({', '.join(drifted_features)})" if drift_detected else "Distribution Stable",
        }