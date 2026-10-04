import os
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans


def compute_correlation(X: pd.DataFrame, output_dir: str) -> pd.DataFrame:
    """
    Computes Pearson correlation matrix across the 6 factors,
    generates a visual heatmap PNG, and writes a readable text report.
    """
    os.makedirs(output_dir, exist_ok=True)
    corr = X.corr(method="pearson")

    # 1. Generate & Save Heatmap PNG
    plt.figure(figsize=(9, 7))
    sns.set_theme(style="white")
    ax = sns.heatmap(
        corr,
        annot=True,
        cmap="Blues",
        fmt=".3f",
        vmin=-0.2,
        vmax=1.0,
        linewidths=1.0,
        cbar_kws={"label": "Pearson Correlation Coefficient"},
    )
    plt.title("Correlation Matrix of 6 Decision Factors\n(Google Borg Production Traces)", fontsize=13, pad=15)
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "correlation_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    print(f"[ml.train] Saved correlation heatmap to: {heatmap_path}")

    # 2. Write Formatted Text Report
    report_path = os.path.join(output_dir, "correlation_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("        CORRELATION ANALYSIS REPORT: 6 DECISION FACTORS\n")
        f.write("             Dataset: Google Borg Production Traces\n")
        f.write("=" * 80 + "\n\n")
        f.write("1. PEARSON CORRELATION MATRIX:\n\n")
        f.write(corr.to_string())
        f.write("\n\n" + "-" * 80 + "\n")
        f.write("2. PAIRWISE MULTICOLLINEARITY ANALYSIS:\n\n")

        # Find any pairs with strong correlation (|r| > 0.70)
        strong_pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                col1, col2 = corr.columns[i], corr.columns[j]
                val = corr.iloc[i, j]
                if abs(val) > 0.70:
                    strong_pairs.append((col1, col2, val))

        if strong_pairs:
            f.write("High correlation pairs (|r| > 0.70):\n")
            for col1, col2, val in strong_pairs:
                f.write(f"  • {col1} <-> {col2}: r = {val:.4f}\n")
        else:
            f.write("No severe collinearity detected (|r| <= 0.70 for all distinct pairs).\n")
            f.write("All 6 factors capture sufficiently distinct operational signals.\n")

        f.write("\n" + "=" * 80 + "\n")

    print(f"[ml.train] Saved correlation text report to: {report_path}")
    return corr


def compute_entropy_weights(X: pd.DataFrame, output_dir: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Derives objective decision weights for the 6 factors using Shannon's Entropy Weight Method (EWM).
    Saves an Entropy Weights Bar Chart PNG and a detailed text report for faculty presentation.
    """
    os.makedirs(output_dir, exist_ok=True)
    m, n = X.shape
    feature_names = list(X.columns)

    print(f"[ml.train] Computing Entropy Weights over {m:,} records across {n} factors...")

    # Step 1: Probability Normalization p_ij
    eps = 1e-12
    X_vals = X.values.astype(float)
    col_sums = np.sum(X_vals + eps, axis=0)
    P = (X_vals + eps) / col_sums

    # Step 2: Shannon Entropy e_j = - (1 / ln(m)) * sum(p_ij * ln(p_ij))
    k = 1.0 / np.log(m)
    # Vectorized entropy calculation
    entropy_j = -k * np.sum(P * np.log(P), axis=0)
    # Clip numerical precision drift to strictly [0.0, 1.0]
    entropy_j = np.clip(entropy_j, 0.0, 1.0)

    # Step 3: Information Diversity / Utility degree d_j = 1 - e_j
    diversity_j = 1.0 - entropy_j

    # Step 4: Normalized Weights W_j = d_j / sum(d_j)
    sum_div = np.sum(diversity_j)
    if sum_div <= 0:
        # Fallback to equal weights if uniform distribution
        weights_raw = np.full(n, 1.0 / n)
    else:
        weights_raw = diversity_j / sum_div

    # Standardize weights format (round to 6 decimals and ensure exact sum = 1.0)
    weights_dict = {name: float(w) for name, w in zip(feature_names, weights_raw)}
    weights_rounded = {name: round(w, 6) for name, w in weights_dict.items()}
    diff = round(1.0 - sum(weights_rounded.values()), 6)
    largest_key = max(weights_rounded, key=weights_rounded.get)
    weights_rounded[largest_key] = round(weights_rounded[largest_key] + diff, 6)

    details = {
        "entropy": {name: float(e) for name, e in zip(feature_names, entropy_j)},
        "diversity": {name: float(d) for name, d in zip(feature_names, diversity_j)},
        "raw_weights": weights_dict,
        "final_weights": weights_rounded,
    }

    # 1. Generate & Save Bar Chart PNG
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Friendly labels
    label_map = {
        "score_cpu": "CPU Waste\n(Score_CPU)",
        "score_memory": "Memory Waste\n(Score_Memory)",
        "score_benefit": "Reclaimable Volume\n(Score_Benefit)",
        "score_priority": "Workload Inversion\n(Score_Priority)",
        "score_state": "Latency / State\n(Score_State)",
        "score_replica": "Redundancy\n(Score_Replica)",
    }
    plot_labels = [label_map.get(col, col) for col in feature_names]
    plot_weights = [weights_rounded[col] for col in feature_names]

    colors = sns.color_palette("viridis", len(feature_names))
    bars = plt.bar(plot_labels, plot_weights, color=colors, width=0.55, edgecolor="black", linewidth=0.8)

    # Add numeric percentage labels above each bar
    for bar in bars:
        h = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.005,
            f"{h * 100:.2f}%\n({h:.4f})",
            ha="center",
            va="bottom",
            fontsize=9.5,
            fontweight="bold",
        )

    plt.ylim(0, max(plot_weights) * 1.25)
    plt.ylabel("Objective Weight Value (Normalized to 1.0)", fontsize=11)
    plt.title(
        "Objective Factor Weights via Entropy Weight Method (EWM)\nDerived from Google Borg Production Traces",
        fontsize=13,
        pad=15,
    )
    plt.tight_layout()
    chart_path = os.path.join(output_dir, "entropy_weights.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[ml.train] Saved entropy weights bar chart to: {chart_path}")

    # 2. Write Comprehensive Text Report for Faculty
    report_path = os.path.join(output_dir, "entropy_weights_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 85 + "\n")
        f.write("        OBJECTIVE WEIGHT DERIVATION: ENTROPY WEIGHT METHOD (EWM)\n")
        f.write("           Dataset: Google Borg Production Traces (1.3M containers)\n")
        f.write("=" * 85 + "\n\n")
        f.write("1. METHODOLOGICAL FOUNDATION:\n")
        f.write("   - Origin: Claude Shannon's Information Entropy (1948) & TOPSIS Decision Science.\n")
        f.write("   - Principle: Features with higher information dispersion across cluster workloads\n")
        f.write("     carry greater discriminative contrast and naturally receive higher weights.\n")
        f.write("   - Eliminates subjective human bias and circular synthetic target formulas.\n\n")
        f.write("-" * 85 + "\n")
        f.write("2. STEP-BY-STEP CALCULATION RESULTS:\n\n")
        f.write(f"{'Factor Name':22s} | {'Entropy (e_j)':15s} | {'Diversity (d_j)':16s} | {'Final Weight':12s} | {'Share (%)':8s}\n")
        f.write("-" * 85 + "\n")
        for col in feature_names:
            e = details["entropy"][col]
            d = details["diversity"][col]
            w = weights_rounded[col]
            pct = w * 100
            f.write(f"{col:22s} | {e:15.8f} | {d:16.8f} | {w:12.6f} | {pct:6.2f}%\n")
        f.write("-" * 85 + "\n")
        f.write(f"{'SUM':22s} | {'-':15s} | {sum_div:16.8f} | {sum(weights_rounded.values()):12.6f} | 100.00%\n")
        f.write("=" * 85 + "\n\n")
        f.write("3. INTERPRETATION & DEFENSE FOR FACULTY:\n")
        f.write("   • Factors with lower entropy (e_j) exhibit higher variation and contrast across workloads.\n")
        f.write("   • Information utility (d_j = 1 - e_j) measures how useful the criterion is to rank candidates.\n")
        f.write("   • The final weights (W_j) sum strictly to 1.000000 and are 100% data-driven.\n")
        f.write("=" * 85 + "\n")

    print(f"[ml.train] Saved entropy weights text report to: {report_path}")
    return weights_rounded, details


def compute_kmeans_thresholds(
    X: pd.DataFrame,
    weights: Dict[str, float],
    output_dir: str,
    random_seed: int = 42,
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Computes composite reclamation scores S over all records,
    fits K-Means (K=3) to identify the 3 natural clusters (KEEP, SOFT_RECLAIM, FULL_RECLAIM),
    and derives the data-backed decision threshold boundaries.
    """
    os.makedirs(output_dir, exist_ok=True)
    m = len(X)
    print(f"[ml.train] Running K-Means clustering (K=3) over {m:,} workload composite scores...")

    # Step 1: Compute composite score S = sum(W_i * X_i)
    scores = np.zeros(m)
    for col, w in weights.items():
        if col in X.columns:
            scores += X[col].values * w

    scores = np.clip(scores, 0.0, 1.0)

    # Step 2: Fit 1D K-Means with K=3
    kmeans = KMeans(n_clusters=3, random_state=random_seed, n_init=10)
    kmeans.fit(scores.reshape(-1, 1))

    # Sort centroids so C0 < C1 < C2
    centroids = np.sort(kmeans.cluster_centers_.flatten())
    c0, c1, c2 = float(centroids[0]), float(centroids[1]), float(centroids[2])

    # Step 3: Derive Action Thresholds (midpoints between adjacent centroids)
    soft_threshold = round(float((c0 + c1) / 2.0), 4)
    full_threshold = round(float((c1 + c2) / 2.0), 4)

    # Compute population distribution across the 3 decision tiers
    count_keep = int(np.sum(scores < soft_threshold))
    count_soft = int(np.sum((scores >= soft_threshold) & (scores < full_threshold)))
    count_full = int(np.sum(scores >= full_threshold))

    pct_keep = (count_keep / m) * 100
    pct_soft = (count_soft / m) * 100
    pct_full = (count_full / m) * 100

    thresholds_dict = {
        "SoftReclaimScoreThreshold": soft_threshold,
        "FullReclaimScoreThreshold": full_threshold,
    }

    cluster_info = {
        "centroids": {"C0_Keep": c0, "C1_Soft": c1, "C2_Full": c2},
        "thresholds": thresholds_dict,
        "distribution": {
            "KEEP": {"count": count_keep, "percentage": round(pct_keep, 2)},
            "SOFT_RECLAIM": {"count": count_soft, "percentage": round(pct_soft, 2)},
            "FULL_RECLAIM": {"count": count_full, "percentage": round(pct_full, 2)},
        },
        "mean_score": float(np.mean(scores)),
        "std_score": float(np.std(scores)),
        "p25": float(np.percentile(scores, 25)),
        "p50": float(np.median(scores)),
        "p75": float(np.percentile(scores, 75)),
        "p90": float(np.percentile(scores, 90)),
    }

    # 1. Generate & Save K-Means Thresholds Histogram Plot
    plt.figure(figsize=(11, 6))
    sns.set_theme(style="whitegrid")

    # Histogram of scores
    counts, bins, _ = plt.hist(scores, bins=60, density=True, color="#4A90E2", alpha=0.65, edgecolor="white")

    # Shaded action regions
    max_y = max(counts) * 1.15 if len(counts) > 0 else 5.0
    plt.axvspan(0.0, soft_threshold, color="#6BAED6", alpha=0.18, label=f"KEEP Zone (< {soft_threshold:.2f}) [{pct_keep:.1f}%]")
    plt.axvspan(soft_threshold, full_threshold, color="#FD8D3C", alpha=0.18, label=f"SOFT_RECLAIM Zone ({soft_threshold:.2f} - {full_threshold:.2f}) [{pct_soft:.1f}%]")
    plt.axvspan(full_threshold, 1.0, color="#74C476", alpha=0.18, label=f"FULL_RECLAIM Zone (>= {full_threshold:.2f}) [{pct_full:.1f}%]")

    # Vertical threshold lines
    plt.axvline(soft_threshold, color="#D94801", linestyle="--", linewidth=2.0, label=f"Soft Threshold = {soft_threshold:.4f}")
    plt.axvline(full_threshold, color="#238B45", linestyle="--", linewidth=2.0, label=f"Full Threshold = {full_threshold:.4f}")

    # Mark centroids
    plt.scatter([c0, c1, c2], [max_y * 0.05, max_y * 0.05, max_y * 0.05], color=["#08519C", "#D94801", "#006D2C"], s=100, zorder=5, marker="D")
    plt.annotate(f"C0 (Keep)\n{c0:.3f}", (c0, max_y * 0.08), ha="center", fontsize=9, fontweight="bold", color="#08519C")
    plt.annotate(f"C1 (Soft)\n{c1:.3f}", (c1, max_y * 0.08), ha="center", fontsize=9, fontweight="bold", color="#D94801")
    plt.annotate(f"C2 (Full)\n{c2:.3f}", (c2, max_y * 0.08), ha="center", fontsize=9, fontweight="bold", color="#006D2C")

    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, max_y)
    plt.xlabel("Composite Reclamation Score S in [0.0, 1.0]", fontsize=11)
    plt.ylabel("Probability Density", fontsize=11)
    plt.title(
        f"K-Means Clustering (K=3) of Reclamation Scores on Google Borg Traces\nEmpirically Derived Thresholds: Soft = {soft_threshold:.4f} | Full = {full_threshold:.4f}",
        fontsize=12,
        pad=15,
    )
    plt.legend(loc="upper right", frameon=True, fontsize=9.5)
    plt.tight_layout()

    chart_path = os.path.join(output_dir, "kmeans_thresholds.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[ml.train] Saved K-Means thresholds plot to: {chart_path}")

    # 2. Write Comprehensive Threshold Report for Faculty
    report_path = os.path.join(output_dir, "thresholds_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 85 + "\n")
        f.write("       DATA-DRIVEN DECISION THRESHOLDS: K-MEANS CLUSTERING (K=3)\n")
        f.write("            Dataset: Google Borg Production Traces (1.3M containers)\n")
        f.write("=" * 85 + "\n\n")
        f.write("1. METHODOLOGICAL FOUNDATION:\n")
        f.write("   - Problem: Hardcoded thresholds (like 0.50 and 0.75) are arbitrary assumptions.\n")
        f.write("   - Solution: 1D Unsupervised K-Means clustering (K=3) groups workloads into 3 natural\n")
        f.write("     density tiers based on their objective reclamation composite scores.\n")
        f.write("   - Decision boundaries are calculated as exact midpoints between adjacent centroids:\n")
        f.write("       SoftReclaimThreshold = (C0 + C1) / 2\n")
        f.write("       FullReclaimThreshold = (C1 + C2) / 2\n\n")
        f.write("-" * 85 + "\n")
        f.write("2. K-MEANS CLUSTERING RESULTS:\n\n")
        f.write(f"   • Cluster 0 Centroid (Low Suitability / KEEP)        : {c0:.6f}\n")
        f.write(f"   • Cluster 1 Centroid (Moderate Suitability / SOFT)   : {c1:.6f}\n")
        f.write(f"   • Cluster 2 Centroid (High Suitability / FULL)       : {c2:.6f}\n\n")
        f.write("   DERIVED DECISION THRESHOLDS:\n")
        f.write(f"   --------------------------------------------------------\n")
        f.write(f"   • SoftReclaimScoreThreshold : {soft_threshold:.4f}  (Score >= {soft_threshold:.4f} triggers SOFT_RECLAIM)\n")
        f.write(f"   • FullReclaimScoreThreshold : {full_threshold:.4f}  (Score >= {full_threshold:.4f} triggers FULL_RECLAIM)\n")
        f.write(f"   --------------------------------------------------------\n\n")
        f.write("-" * 85 + "\n")
        f.write("3. EMPIRICAL WORKLOAD POPULATION DISTRIBUTION:\n\n")
        f.write(f"   {'Action Tier':16s} | {'Score Range':20s} | {'Workload Count':16s} | {'Percentage':10s}\n")
        f.write(f"   {'-'*16} | {'-'*20} | {'-'*16} | {'-'*10}\n")
        f.write(f"   {'KEEP':16s} | [0.0000, {soft_threshold:.4f})     | {count_keep:16,d} | {pct_keep:9.2f}%\n")
        f.write(f"   {'SOFT_RECLAIM':16s} | [{soft_threshold:.4f}, {full_threshold:.4f})     | {count_soft:16,d} | {pct_soft:9.2f}%\n")
        f.write(f"   {'FULL_RECLAIM':16s} | [{full_threshold:.4f}, 1.0000]     | {count_full:16,d} | {pct_full:9.2f}%\n")
        f.write(f"   {'-'*16} | {'-'*20} | {'-'*16} | {'-'*10}\n")
        f.write(f"   {'TOTAL':16s} | [0.0000, 1.0000]     | {m:16,d} | 100.00%\n\n")
        f.write("=" * 85 + "\n")

    print(f"[ml.train] Saved thresholds report to: {report_path}")
    return thresholds_dict, cluster_info
