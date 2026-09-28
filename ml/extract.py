import os
import numpy as np
import pandas as pd
from typing import Tuple, Optional


FEATURE_NAMES = [
    "score_cpu",
    "score_memory",
    "score_benefit",
    "score_idle",
    "score_replica",
    "score_priority",
    "score_state",
    "score_pdb",
    "score_checkpoint",
]


def resolve_dataset_path(custom_path: Optional[str] = None) -> str:
    """Finds the dataset locally or downloads it via kagglehub if absent."""
    if custom_path and os.path.exists(custom_path):
        return custom_path

    # Check local repository dataset folder
    candidates = [
        "dataset/borg_traces_data.csv",
        "../dataset/borg_traces_data.csv",
        os.path.join(os.path.dirname(__file__), "..", "dataset", "borg_traces_data.csv"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)

    # Check kagglehub cache directory
    cache_path = os.path.expanduser("~/.cache/kagglehub/datasets/derrickmwiti/google-2019-cluster-sample/versions/1/borg_traces_data.csv")
    if os.path.exists(cache_path):
        return cache_path

    # Fallback to automated kagglehub download
    print("[ml.extract] Local dataset not found. Downloading via kagglehub...")
    import kagglehub
    path = kagglehub.dataset_download("derrickmwiti/google-2019-cluster-sample")
    target = os.path.join(path, "borg_traces_data.csv")
    if os.path.exists(target):
        return target
    raise FileNotFoundError(f"Could not find borg_traces_data.csv in {path}")


def extract_features_and_target(
    dataset_path: str,
    sample_size: int = 0,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Extracts features and target from Borg dataset.
    If sample_size <= 0, processes the ENTIRE dataset (all 1.32M rows).
    
    Returns:
        X: DataFrame of 9 normalized scoring features
        y: Series of continuous Net_Utility targets
        raw_meta: DataFrame with original priorities, events, and failed flags
    """
    usecols = [
        "priority",
        "scheduling_class",
        "collection_id",
        "instance_index",
        "resource_request",
        "average_usage",
        "start_time",
        "end_time",
        "vertical_scaling",
        "constraint",
        "failed",
        "event",
    ]

    if sample_size > 0:
        print(f"[ml.extract] Loading sample of {sample_size} records from {dataset_path}...")
        df = pd.read_csv(dataset_path, usecols=usecols, nrows=sample_size * 2)
    else:
        print(f"[ml.extract] Loading ALL rows from {dataset_path} (full dataset)...")
        df = pd.read_csv(dataset_path, usecols=usecols)

    df = df.dropna(subset=["resource_request", "average_usage", "priority", "failed"])
    
    if sample_size > 0 and len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=random_seed).reset_index(drop=True)

    total_rows = len(df)
    print(f"[ml.extract] Processing {total_rows:,} records into 9 decision factors (vectorized)...")

    # Fast Vectorized Regex Extraction for JSON strings (100x faster than literal_eval)
    req_cpu = df["resource_request"].str.extract(r"'cpus':\s*([0-9.eE+-]+)")[0].astype(float).fillna(0.0).values
    req_mem = df["resource_request"].str.extract(r"'memory':\s*([0-9.eE+-]+)")[0].astype(float).fillna(0.0).values

    use_cpu = df["average_usage"].str.extract(r"'cpus':\s*([0-9.eE+-]+)")[0].astype(float).fillna(0.0).values
    use_mem = df["average_usage"].str.extract(r"'memory':\s*([0-9.eE+-]+)")[0].astype(float).fillna(0.0).values

    # 1. Factor 1: CPU Waste Score [0, 1]
    cpu_ratio = np.where(req_cpu > 0, use_cpu / np.maximum(req_cpu, 1e-9), 1.0)
    score_cpu = np.clip(1.0 - cpu_ratio, 0.0, 1.0)

    # 2. Factor 2: Memory Waste Score [0, 1]
    mem_ratio = np.where(req_mem > 0, use_mem / np.maximum(req_mem, 1e-9), 1.0)
    score_mem = np.clip(1.0 - mem_ratio, 0.0, 1.0)

    # 3. Factor 3: Reclamation Benefit Score [0, 1]
    reclaim_cpu = np.maximum(0.0, req_cpu - use_cpu)
    reclaim_mem = np.maximum(0.0, req_mem - use_mem)
    p95_cpu = np.percentile(reclaim_cpu, 95)
    p95_mem = np.percentile(reclaim_mem, 95)
    max_cpu = p95_cpu if p95_cpu > 0 else 1.0
    max_mem = p95_mem if p95_mem > 0 else 1.0
    score_benefit = np.clip(0.5 * (reclaim_cpu / max_cpu) + 0.5 * (reclaim_mem / max_mem), 0.0, 1.0)

    # 4. Factor 4: Idle Duration Score [0, 1]
    duration_sec = np.maximum(0.0, (df["end_time"].values - df["start_time"].values) / 1e6)
    score_idle = np.clip(duration_sec / 3600.0, 0.0, 1.0)
    # Higher idle score if the task exhibits low CPU usage
    score_idle = np.where(score_cpu > 0.4, score_idle, score_idle * 0.2)

    # 5. Factor 5: Priority Score [0, 1]
    # Lower priority -> safer -> higher score (Borg max priority = 450)
    priority_val = pd.to_numeric(df["priority"], errors="coerce").fillna(200).values
    score_priority = np.clip(1.0 - (priority_val / 450.0), 0.0, 1.0)

    # 6. Factor 6: Replica Redundancy Score [0, 1]
    # Redundancy mapped from collection instance counts
    collection_counts = df["collection_id"].map(df["collection_id"].value_counts()).fillna(1).values
    score_replica = np.select(
        [collection_counts >= 4, collection_counts == 3, collection_counts == 2],
        [1.0, 0.8, 0.5],
        default=0.0,
    )

    # 7. Factor 7: Workload Statefulness Score [0, 1]
    # Borg scheduling_class: 0=batch (stateless) -> 1.0, 3=latency-critical -> 0.1
    sched_class = pd.to_numeric(df["scheduling_class"], errors="coerce").fillna(1).values
    score_state = np.select(
        [sched_class == 0, sched_class == 1, sched_class == 2],
        [1.0, 0.7, 0.4],
        default=0.1,
    )

    # 8. Factor 8: PDB / Constraint Headroom Score [0, 1]
    has_constraints = df["constraint"].apply(lambda c: 0.0 if (c and str(c) not in ("[]", "nan")) else 1.0).values
    score_pdb = has_constraints

    # 9. Factor 9: Checkpoint Compatibility Score [0, 1]
    vert_scaling = pd.to_numeric(df["vertical_scaling"], errors="coerce").fillna(0.0).values
    score_checkpoint = np.where(vert_scaling > 0.0, 1.0, 0.5)

    # Construct Features DataFrame
    X = pd.DataFrame(
        {
            "score_cpu": score_cpu,
            "score_memory": score_mem,
            "score_benefit": score_benefit,
            "score_idle": score_idle,
            "score_replica": score_replica,
            "score_priority": score_priority,
            "score_state": score_state,
            "score_pdb": score_pdb,
            "score_checkpoint": score_checkpoint,
        }
    )

    # Formulation B: Comprehensive Multi-Criteria Net Reclaim Utility Target
    # 1. Resource Headroom Saved (Proportional Waste + Absolute Benefit)
    gain = 0.6 * ((score_cpu + score_mem) / 2.0) + 0.4 * score_benefit

    # 2. Operational Safety & Non-Disruption
    safety = (
        0.25 * score_priority
        + 0.20 * score_replica
        + 0.20 * score_idle
        + 0.15 * score_state
        + 0.10 * score_pdb
        + 0.10 * score_checkpoint
    )

    # 3. Overall Net Utility penalized by failure
    failed_val = pd.to_numeric(df["failed"], errors="coerce").fillna(0).values
    net_utility = gain * safety * (1.0 - failed_val)
    y = pd.Series(net_utility, name="net_utility")

    raw_meta = df[["priority", "scheduling_class", "event", "failed"]].copy()

    print(f"[ml.extract] Feature matrix created: shape {X.shape}, target mean: {y.mean():.4f}")
    return X, y, raw_meta
