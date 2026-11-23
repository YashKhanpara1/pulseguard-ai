from pathlib import Path
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest


def generate_baseline_telemetry(n_samples: int = 5000, seed: int = 42) -> np.ndarray:
    """Generate realistic healthy production API & server telemetry."""
    rng = np.random.default_rng(seed)

    # Healthy API latencies centered around ~110ms (log-normal right-skewed)
    latency_ms = rng.lognormal(mean=4.65, sigma=0.28, size=n_samples)
    latency_ms = np.clip(latency_ms, 25.0, 450.0)

    # Mostly HTTP 200 (98.5%), occasional 4xx (1.5%)
    status_code = rng.choice([200, 201, 400, 404], size=n_samples, p=[0.90, 0.085, 0.01, 0.005])

    # Payload size in KB (normal API requests ~2KB to 45KB)
    payload_kb = rng.gamma(shape=3.0, scale=6.5, size=n_samples)
    payload_kb = np.clip(payload_kb, 0.5, 120.0)

    # Server CPU utilization % (healthy baseline ~25% to 60%)
    cpu_usage_pct = rng.normal(loc=42.0, scale=9.5, size=n_samples)
    cpu_usage_pct = np.clip(cpu_usage_pct, 10.0, 82.0)

    return np.column_stack([latency_ms, status_code, payload_kb, cpu_usage_pct])


def main() -> None:
    print("Generating baseline telemetry dataset...")
    X_train = generate_baseline_telemetry(n_samples=5000)

    print("Training IsolationForest anomaly detector...")
    model = IsolationForest(
        n_estimators=150,
        contamination=0.03,
        max_samples=256,
        random_state=42,
    )
    model.fit(X_train)

    # Keep a 500-sample reference slice for fast KS-test statistical drift checks
    reference_sample = X_train[:500]

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    bundle_path = artifacts_dir / "model_bundle.joblib"

    joblib.dump(
        {
            "model": model,
            "reference_data": reference_sample,
            "feature_names": ["latency_ms", "status_code", "payload_kb", "cpu_usage_pct"],
        },
        bundle_path,
    )
    print(f"Saved trained model bundle to {bundle_path.resolve()}")


if __name__ == "__main__":
    main()