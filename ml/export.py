import os
import json
import joblib
from datetime import datetime, timezone
from typing import Dict, Any


GO_FIELD_MAPPING = {
    "score_cpu": "WeightCPU",
    "score_memory": "WeightMemory",
    "score_benefit": "WeightBenefit",
    "score_idle": "WeightIdle",
    "score_replica": "WeightReplica",
    "score_priority": "WeightPriority",
    "score_state": "WeightState",
    "score_pdb": "WeightPDB",
    "score_checkpoint": "WeightCheckpoint",
}


def export_artifacts(
    output_dir: str,
    weights: Dict[str, float],
    linear_weights: Dict[str, float],
    xgb_weights: Dict[str, float],
    raw_coefficients: Dict[str, float],
    metrics: Dict[str, Any],
    ridge_model,
    xgb_model,
    sample_size: int,
):
    """
    Saves JSON artifacts, serialized models, and prints the 9-decimal Go struct snippet.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Prepare weights JSON with 9 decimals
    weights_artifact = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": "Google Borg 2019 Traces (derrickmwiti/google-2019-cluster-sample)",
            "sample_size": sample_size,
            "model_type": "Ensemble Bounded Ridge NNLS + XGBoost Feature Importance",
            "target_formulation": "Multi-Criteria Net Reclaim Utility (Formulation B)",
            "precision_decimals": 9,
            "metrics": metrics,
        },
        "weights": {GO_FIELD_MAPPING.get(k, k): v for k, v in weights.items()},
        "linear_weights": {GO_FIELD_MAPPING.get(k, k): v for k, v in linear_weights.items()},
        "xgboost_weights": {GO_FIELD_MAPPING.get(k, k): v for k, v in xgb_weights.items()},
        "raw_coefficients": {GO_FIELD_MAPPING.get(k, k): round(v, 9) for k, v in raw_coefficients.items()},
        "validation": {
            "sum": round(sum(weights.values()), 9),
            "factors_count": len(weights),
        },
    }

    weights_file = os.path.join(output_dir, "trained_weights.json")
    with open(weights_file, "w") as f:
        json.dump(weights_artifact, f, indent=2)
    print(f"[ml.export] Saved weights artifact to: {weights_file}")

    # 2. Save metrics JSON
    metrics_file = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[ml.export] Saved training metrics to: {metrics_file}")

    # 3. Save serialized models
    ridge_file = os.path.join(output_dir, "ridge_model.joblib")
    joblib.dump(ridge_model, ridge_file)
    print(f"[ml.export] Saved Ridge model to: {ridge_file}")

    xgb_file = os.path.join(output_dir, "xgboost_model.json")
    try:
        xgb_model.save_model(xgb_file)
        print(f"[ml.export] Saved XGBoost model to: {xgb_file}")
    except Exception as e:
        print(f"[ml.export] Warning: Could not save native XGBoost JSON: {e}")

    # 4. Generate Go Code Snippet (9 Decimals)
    print("\n" + "=" * 65)
    print("      GENERATED GO STRUCT CODE (for pkg/decision/policy.go)")
    print("      Exact 9-Decimal Precision (Sum == 1.000000000)")
    print("=" * 65)
    print("// Data-backed weights derived via Bounded NNLS + XGBoost on Google Borg traces:")
    for feat, go_field in GO_FIELD_MAPPING.items():
        w = weights.get(feat, 0.0)
        print(f"\t\t{go_field:17s}: {w:.9f},")
    print("=" * 65 + "\n")
