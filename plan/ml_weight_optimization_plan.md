# ML Weight and Threshold Optimization Plan

This document outlines the strategy for utilizing the Google Borg production trace dataset (`dataset/borg_traces_data.csv`) to:
1. Derive objective scoring weights for **6 factors** using the **Entropy Weight Method (EWM)**.
2. Determine decision threshold ranges for **`KEEP`**, **`SOFT_RECLAIM`**, and **`FULL_RECLAIM`** using **K-Means Clustering**.

---

## 1. The 6 Scoring Factors (Mapped to Borg Dataset)

We focus on the 6 factors that map directly to columns in `dataset/borg_traces_data.csv`:

```text
Reclaim_Score = (W_CPU      * Score_CPU)
              + (W_Mem      * Score_Mem)
              + (W_Benefit  * Score_Benefit)
              + (W_Priority * Score_Priority)
              + (W_State    * Score_State)
              + (W_Replica  * Score_Replica)
```

### Dataset Column Mapping

| Factor Symbol | Factor Name | Raw Borg Column(s) | Extraction / Normalization Logic | Score Range | Description |
|---|---|---|---|---|---|
| `Score_CPU` | CPU Waste | `resource_request['cpus']`, `average_usage['cpus']` | `ratio = usage_cpu / max(req_cpu, 1e-9)`<br>`Score_CPU = clamp(1.0 - ratio, 0.0, 1.0)` | `[0.0, 1.0]` | Higher waste means more unutilized CPU headroom. |
| `Score_Mem` | Memory Waste | `resource_request['memory']`, `average_usage['memory']` | `ratio = usage_mem / max(req_mem, 1e-9)`<br>`Score_Mem = clamp(1.0 - ratio, 0.0, 1.0)` | `[0.0, 1.0]` | Higher waste means unallocated memory buffer. |
| `Score_Benefit` | Reclaimable Volume | `resource_request`, `average_usage` | `Diff_CPU = req_cpu - usage_cpu`<br>`Diff_Mem = req_mem - usage_mem`<br>`Score = 0.5*(Diff_CPU/P95_CPU) + 0.5*(Diff_Mem/P95_Mem)` | `[0.0, 1.0]` | Absolute CPU and Memory capacity recoverable. |
| `Score_Priority` | Workload Inversion | `priority` (0 to 450) | `Score_Priority = 1.0 - clamp(priority / 450.0, 0.0, 1.0)` | `[0.0, 1.0]` | Lower-priority batch jobs yield resources to high-priority services. |
| `Score_State` | Latency / Statefulness | `scheduling_class` (0 to 3) | Discrete mapping:<br>• Class 0 (Batch / non-prod) $\rightarrow 1.0$<br>• Class 1 (Internal service) $\rightarrow 0.7$<br>• Class 2 (Mid-tier service) $\rightarrow 0.4$<br>• Class 3 (User-facing latency-critical) $\rightarrow 0.1$ | `[0.0, 1.0]` | Stateless batch tasks tolerate disruption better than latency-critical services. |
| `Score_Replica` | Service Redundancy | `collection_id`, `instance_index` | Group by `collection_id`, count active instances:<br>• $\ge 4$ instances $\rightarrow 1.0$<br>• $3$ instances $\rightarrow 0.8$<br>• $2$ instances $\rightarrow 0.5$<br>• $\le 1$ instance $\rightarrow 0.0$ | `[0.0, 1.0]` | Higher redundancy means less risk to overall service availability. |

---

## 2. Weight Optimization: Correlation Analysis & Entropy Weight Method (EWM)

### Step 1: Correlation Analysis
* Compute the Pearson correlation matrix across the 6 extracted factors.
* Verify feature relationships and ensure no two factors are exact collinear duplicates.

### Step 2: Entropy Weight Method (EWM)
The Entropy Weight Method calculates the weight of each factor based on the information dispersion (entropy) across all rows in the dataset:

1. **Normalize Matrix ($m$ rows, $n=6$ columns):**
   $$p_{ij} = \frac{x_{ij} + \epsilon}{\sum_{i=1}^{m} (x_{ij} + \epsilon)}$$
   *(where $\epsilon = 10^{-12}$)*

2. **Compute Shannon's Information Entropy ($e_j$):**
   $$e_j = -\frac{1}{\ln(m)} \sum_{i=1}^{m} p_{ij} \ln(p_{ij})$$

3. **Compute Information Utility ($d_j$):**
   $$d_j = 1.0 - e_j$$
   Factors with greater variability across workloads have higher information utility ($d_j$).

4. **Calculate Final Normalized Weights ($W_j$):**
   $$W_j = \frac{d_j}{\sum_{k=1}^{6} d_k}$$
   Guarantees: $W_j \ge 0$ and $\sum_{j=1}^{6} W_j = 1.0$.

---

## 3. Threshold Optimization: K-Means Clustering for Decision Ranges

Instead of hardcoding static threshold cutoffs (like 0.50 and 0.75), we derive the boundaries using **unsupervised K-Means clustering** on the scores:

1. Compute the composite reclamation score $S_i$ for all workloads in the dataset using the derived EWM weights:
   $$S_i = \sum_{j=1}^{6} W_j \cdot x_{ij}$$

2. Fit **K-Means Clustering** with $K = 3$ clusters on the distribution of $S$:
   * **Cluster 0:** Low score centroid ($C_0$) $\rightarrow$ **`KEEP`**
   * **Cluster 1:** Medium score centroid ($C_1$) $\rightarrow$ **`SOFT_RECLAIM`**
   * **Cluster 2:** High score centroid ($C_2$) $\rightarrow$ **`FULL_RECLAIM`**

3. **Derive the Decision Ranges:**
   * Boundary between `KEEP` and `SOFT_RECLAIM`:
     $$\text{SoftReclaimScoreThreshold} = \frac{C_0 + C_1}{2}$$
   * Boundary between `SOFT_RECLAIM` and `FULL_RECLAIM`:
     $$\text{FullReclaimScoreThreshold} = \frac{C_1 + C_2}{2}$$

---

## 4. Implementation Workflow

```text
+-------------------------------------------------------------+
|                 dataset/borg_traces_data.csv                |
+-------------------------------------------------------------+
                               |
                               v  (ml/extract.py)
+-------------------------------------------------------------+
|  1. Extract the 6 features from Borg trace columns          |
|     [Score_CPU, Score_Mem, Benefit, Priority, State, Replica]|
+-------------------------------------------------------------+
                               |
                               v  (ml/train.py)
+-------------------------------------------------------------+
|  1. Run Correlation Analysis                                |
|  2. Compute Entropy Weights (W_CPU, W_Mem, ..., W_Replica)  |
|  3. Compute Composite Scores S for all rows                 |
|  4. Run K-Means (K=3) to derive Soft and Full Thresholds    |
+-------------------------------------------------------------+
                               |
                               v  (ml/export.py)
+-------------------------------------------------------------+
|  Export ml/artifacts/trained_weights.json                   |
|  - weights: {WeightCPU, WeightMemory, ..., WeightReplica}   |
|  - thresholds: {SoftThreshold, FullThreshold}               |
+-------------------------------------------------------------+
                               |
                               v
+-------------------------------------------------------------+
|  Go Engine Integration:                                     |
|  - pkg/decision/scoring.go (6-factor score computation)     |
|  - pkg/decision/policy.go (Defaults)                        |
|  - pkg/decision/weights.go (JSON loader)                    |
+-------------------------------------------------------------+
```

---

## 5. Deliverables

1. **`ml/extract.py`**: Extracts the 6 normalized features from the Borg traces.
2. **`ml/train.py`**: Computes correlation matrix, Shannon entropy weights, and K-Means cluster thresholds.
3. **`ml/export.py`**: Exports weights and thresholds to `ml/artifacts/trained_weights.json`.
4. **`adaptive-k8s-scheduler/pkg/decision/`**:
   * Update `scoring.go` and `policy.go` to use the 6 factors with the trained weights and thresholds.
   * Update `weights.go` to load the 6 weights and thresholds from JSON.
