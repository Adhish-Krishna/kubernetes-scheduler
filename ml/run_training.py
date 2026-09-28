import os
import argparse
import time
from sklearn.model_selection import train_test_split

from extract import resolve_dataset_path, extract_features_and_target
from train import train_models
from evaluate import evaluate_models
from export import export_artifacts


def main():
    parser = argparse.ArgumentParser(description="Adaptive Scheduler ML Weight Training Pipeline")
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
        help="Directory to save weights and model artifacts",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Proportion of dataset for testing (default: 0.2)",
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
    print("=" * 60)
    print("  ADAPTIVE K8S SCHEDULER: ML WEIGHT TRAINING PIPELINE")
    print(f"  Dataset Scope: {'ALL ROWS (1.32M)' if effective_sample_size <= 0 else f'{effective_sample_size:,} rows'}")
    print(f"  Target: Formulation B (Multi-Criteria Net Reclaim Utility)")
    print(f"  Primary Model: Ridge Regression (NNLS, non-negative)")
    print(f"  Benchmark: XGBoost Regressor")
    print("=" * 60)

    # 1. Resolve dataset
    dataset_file = resolve_dataset_path(args.data_path)
    print(f"[pipeline] Using dataset file: {dataset_file}")

    # 2. Extract features and target
    X, y, raw_meta = extract_features_and_target(
        dataset_path=dataset_file,
        sample_size=effective_sample_size,
        random_seed=args.seed,
    )

    # 3. Train/Test Split
    print(f"[pipeline] Splitting data: {1 - args.test_split:.0%} train, {args.test_split:.0%} test...")
    X_train, X_test, y_train, y_test, meta_train, meta_test = train_test_split(
        X, y, raw_meta, test_size=args.test_split, random_state=args.seed
    )

    # 4. Train Models
    ridge_model, xgb_model, blended_weights, linear_weights, xgb_weights, raw_coefs = train_models(
        X_train=X_train,
        y_train=y_train,
        random_seed=args.seed,
    )

    # 5. Evaluate Models
    metrics = evaluate_models(
        ridge_model=ridge_model,
        xgb_model=xgb_model,
        X_test=X_test,
        y_test=y_test,
        raw_meta_test=meta_test,
    )

    # 6. Export Artifacts
    export_artifacts(
        output_dir=args.output_dir,
        weights=blended_weights,
        linear_weights=linear_weights,
        xgb_weights=xgb_weights,
        raw_coefficients=raw_coefs,
        metrics=metrics,
        ridge_model=ridge_model,
        xgb_model=xgb_model,
        sample_size=len(X),
    )

    elapsed = time.time() - start_time
    print(f"[pipeline] Training pipeline completed successfully in {elapsed:.2f} seconds!")
    print(f"[pipeline] Trained weights saved to: {os.path.join(args.output_dir, 'trained_weights.json')}")


if __name__ == "__main__":
    main()
