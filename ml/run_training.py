import os
import argparse
import time

from extract import resolve_dataset_path, extract_features
from train import compute_correlation, compute_entropy_weights, compute_kmeans_thresholds
from export import export_artifacts


def main():
    parser = argparse.ArgumentParser(description="Adaptive Scheduler ML Weight & Threshold Optimization Pipeline")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Number of records to sample (0 = process ALL 1.32M rows in dataset)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Train on all rows in dataset (equivalent to --sample-size 0)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Path to borg_traces_data.csv (default: auto-detected)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "artifacts"),
        help="Directory to save weights, charts, and report artifacts",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()

    effective_sample_size = 0 if args.all else args.sample_size

    start_time = time.time()
    print("=" * 70)
    print("  ADAPTIVE K8S SCHEDULER: WEIGHT & THRESHOLD OPTIMIZATION PIPELINE")
    print(f"  Dataset Scope: {'ALL ROWS (1.32M)' if effective_sample_size <= 0 else f'{effective_sample_size:,} rows'}")
    print(f"  Stage 2: Correlation Analysis + Shannon's Entropy Weight Method (EWM)")
    print(f"  Stage 3: Unsupervised K-Means Clustering (K=3) for Decision Ranges")
    print(f"  Factors: score_cpu, score_memory, score_benefit, score_priority, score_state, score_replica")
    print("=" * 70)

    # 1. Resolve dataset
    dataset_file = resolve_dataset_path(args.data_path)
    print(f"[pipeline] Using dataset file: {dataset_file}")

    # 2. Extract 6 physical features
    X, raw_meta = extract_features(
        dataset_path=dataset_file,
        sample_size=effective_sample_size,
        random_seed=args.seed,
    )

    # 3. Correlation Analysis (Heatmap PNG + Text Report)
    print("\n[pipeline] Running Stage 2: Correlation Analysis...")
    corr_matrix = compute_correlation(X=X, output_dir=args.output_dir)

    # 4. Entropy Weight Method (Bar Chart PNG + Text Report)
    print("\n[pipeline] Running Stage 2: Entropy Weight Method (EWM)...")
    weights, entropy_details = compute_entropy_weights(X=X, output_dir=args.output_dir)

    # 5. K-Means Threshold Optimization (Cluster Histogram PNG + Text Report)
    print("\n[pipeline] Running Stage 3: K-Means Clustering for Threshold Ranges...")
    thresholds, cluster_info = compute_kmeans_thresholds(
        X=X,
        weights=weights,
        output_dir=args.output_dir,
        random_seed=args.seed,
    )

    # 6. Export Unified JSON Artifact
    print("\n[pipeline] Exporting artifacts...")
    export_artifacts(
        output_dir=args.output_dir,
        weights=weights,
        entropy_details=entropy_details,
        thresholds=thresholds,
        cluster_info=cluster_info,
        sample_size=len(X),
    )

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f"[pipeline] Pipeline executed successfully in {elapsed:.2f} seconds!")
    print(f"[pipeline] Visual heatmap:       {os.path.join(args.output_dir, 'correlation_heatmap.png')}")
    print(f"[pipeline] Correlation report:   {os.path.join(args.output_dir, 'correlation_report.txt')}")
    print(f"[pipeline] Visual weights chart: {os.path.join(args.output_dir, 'entropy_weights.png')}")
    print(f"[pipeline] Entropy report:       {os.path.join(args.output_dir, 'entropy_weights_report.txt')}")
    print(f"[pipeline] Thresholds plot:      {os.path.join(args.output_dir, 'kmeans_thresholds.png')}")
    print(f"[pipeline] Thresholds report:    {os.path.join(args.output_dir, 'thresholds_report.txt')}")
    print(f"[pipeline] Weights & Thresh JSON:{os.path.join(args.output_dir, 'trained_weights.json')}")
    print("=" * 70)


if __name__ == "__main__":
    main()
