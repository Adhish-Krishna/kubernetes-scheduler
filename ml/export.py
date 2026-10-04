import os
import json
from datetime import datetime, timezone
from typing import Dict, Any


GO_FIELD_MAPPING = {
    "score_cpu": "WeightCPU",
    "score_memory": "WeightMemory",
    "score_benefit": "WeightBenefit",
    "score_priority": "WeightPriority",
    "score_state": "WeightState",
    "score_replica": "WeightReplica",
}


def export_artifacts(
    output_dir: str,
    weights: Dict[str, float],
    entropy_details: Dict[str, Any],
    thresholds: Dict[str, float],
    cluster_info: Dict[str, Any],
    sample_size: int,
):
    """
    Saves JSON artifacts and prints the derived Go struct snippet.
    """
    os.makedirs(output_dir, exist_ok=True)

    weights_mapped = {GO_FIELD_MAPPING.get(k, k): v for k, v in weights.items()}

    weights_artifact = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "dataset": "Google Borg 2019 Traces (derrickmwiti/google-2019-cluster-sample)",
            "sample_size": sample_size,
            "weight_derivation_method": "Shannon's Entropy Weight Method (EWM)",
            "threshold_derivation_method": "Unsupervised K-Means Clustering (K=3)",
            "factors_count": len(weights),
        },
        "weights": weights_mapped,
        "thresholds": thresholds,
        "cluster_distribution": cluster_info["distribution"],
        "cluster_centroids": cluster_info["centroids"],
        "entropy_metrics": {
            GO_FIELD_MAPPING.get(k, k): {
                "entropy": entropy_details["entropy"][k],
                "diversity_utility": entropy_details["diversity"][k],
                "weight": weights[k],
                "percentage": f"{weights[k] * 100:.2f}%",
            }
            for k in weights
        },
        "validation": {
            "sum": round(sum(weights.values()), 6),
            "is_valid": abs(sum(weights.values()) - 1.0) < 1e-5,
        },
    }

    weights_file = os.path.join(output_dir, "trained_weights.json")
    with open(weights_file, "w", encoding="utf-8") as f:
        json.dump(weights_artifact, f, indent=2)
    print(f"[ml.export] Saved weights artifact to: {weights_file}")

    print("\n" + "=" * 65)
    print("      OBJECTIVE WEIGHTS FOR DECISION ENGINE (6 FACTORS)")
    print("=" * 65)
    for feat, go_field in GO_FIELD_MAPPING.items():
        w = weights.get(feat, 0.0)
        print(f"  {go_field:16s}: {w:.6f}  ({w * 100:5.2f}%)")
    print("-" * 65)
    print(f"  {'SUM':16s}: {sum(weights.values()):.6f}  (100.00%)")
    print("=" * 65)

    print("\n" + "=" * 65)
    print("      DATA-DRIVEN DECISION THRESHOLDS (K-MEANS K=3)")
    print("=" * 65)
    print(f"  SoftReclaimScoreThreshold : {thresholds['SoftReclaimScoreThreshold']:.4f}")
    print(f"  FullReclaimScoreThreshold : {thresholds['FullReclaimScoreThreshold']:.4f}")
    print("  Population Distribution:")
    for tier, data in cluster_info["distribution"].items():
        print(f"    • {tier:14s}: {data['count']:10,d} ({data['percentage']:5.2f}%)")
    print("=" * 65 + "\n")
