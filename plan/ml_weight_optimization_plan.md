# ML-Driven Weight Optimization Plan

This document outlines the end-to-end strategy for utilizing the Google Borg production trace dataset (`dataset/borg_traces_data.csv`) to train machine learning models and derive optimal, data-backed scoring weights for the Reclamation Decision Engine in `pkg/decision/policy.go`.

---

## 1. Problem Statement

In the current implementation (`pkg/decision/policy.go`), the multi-criteria decision formula scores potential pods for resource reclamation using heuristic weights:

```text
Reclaim_Score = (W_CPU * Score_CPU)
              + (W_Mem * Score_Mem)
              + (W_Idle * Score_Idle)
              + (W_Benefit * Score_Benefit)
              + (W_Replica * Score_Replica)
              + (W_Priority * Score_Priority)
              + (W_PDB * Score_PDB)
              + (W_State * Score_State)
              + (W_Checkpoint * Score_Checkpoint)
```

The baseline initial weights were assumed as:
* `WeightCPU` = 0.20
* `WeightMemory` = 0.20
* `WeightIdle` = 0.15
* `WeightBenefit` = 0.15
* `WeightReplica` = 0.10
* `WeightPriority` = 0.05
* `WeightPDB` = 0.05
* `WeightState` = 0.05
* `WeightCheckpoint` = 0.05

While intuitive, these weights are manually assigned. By using production cluster traces from Google Borg (`dataset/borg_traces_data.csv`), we can replace assumed weights with statistically and empirically trained weights that maximize resource reclamation while minimizing workload failure and disruption.

---

## 2. Dataset Overview (`dataset/borg_traces_data.csv`)

The dataset located in the repository contains **1,324,696 rows** (approx. 328 MB) representing real-world containerized workloads on Google Borg compute clusters.

### Relevant Columns

| Dataset Column | Example Value | Description |
|---|---|---|
| `resource_request` | `{'cpus': 0.02066, 'memory': 0.01443}` | Normalized CPU and memory requested by the container |
| `average_usage` | `{'cpus': 0.00466, 'memory': 0.00592}` | Real mean CPU and memory consumed during execution |
| `maximum_usage` | `{'cpus': 0.01190, 'memory': 0.00593}` | Peak resource spikes observed |
| `priority` | `200` | Borg scheduling priority (0 = non-critical, 450 = infrastructure) |
| `scheduling_class` | `3` | Latency sensitivity (0 = batch, 3 = user-facing latency-critical) |
| `collection_id` | `94591244395` | Logical group ID (equivalent to a Kubernetes Deployment/ReplicaSet) |
| `instance_index` | `144` | Index of the task within the collection (replica instance) |
| `start_time`, `end_time` | `274800000000`, `275100000000` | Execution window timestamps (microseconds) |
| `event` | `FAIL`, `FINISH`, `EVICT`, `SCHEDULE` | Termination or lifecycle event |
| `failed` | `1` or `0` | Binary indicator whether the task failed |

---

## 3. Feature Mapping: Borg Traces to Decision Engine Factors

The table below maps each criteria from `pkg/decision/scoring.go` and `pkg/decision/policy.go` to the corresponding raw fields in `dataset/borg_traces_data.csv`, along with the extraction formula and normalization logic:

| Factor Symbol | Factor Name | Baseline Weight | Borg Trace Column(s) | Mathematical Extraction & Normalization Formula | Score Range | Engineering Rationale |
|---|---|---|---|---|---|---|
| `Score_CPU` | CPU Waste | 0.20 | `resource_request['cpus']`, `average_usage['cpus']` | `CPU_Waste = 1.0 - (usage_cpu / req_cpu)`<br>`Score_CPU = clamp(CPU_Waste, 0.0, 1.0)` | `[0.0, 1.0]` | Higher waste means CPU can be reclaimed with minimal risk of CPU starvation. |
| `Score_Mem` | Memory Waste | 0.20 | `resource_request['memory']`, `average_usage['memory']` | `Mem_Waste = 1.0 - (usage_mem / req_mem)`<br>`Score_Mem = clamp(Mem_Waste, 0.0, 1.0)` | `[0.0, 1.0]` | Large unallocated memory buffers represent trapped headroom that can be safely right-sized. |
| `Score_Benefit` | Reclamation Benefit | 0.15 | `resource_request`, `average_usage` | `Diff_CPU = req_cpu - usage_cpu`<br>`Diff_Mem = req_mem - usage_mem`<br>`Score = 0.5*(Diff_CPU / Max_CPU) + 0.5*(Diff_Mem / Max_Mem)` | `[0.0, 1.0]` | Reclaiming 2 cores / 4 GB provides more cluster-wide value than reclaiming 50 mCPU / 64 MB. |
| `Score_Idle` | Idle Duration | 0.15 | `start_time`, `end_time`, `average_usage['cpus']` | `Duration_Sec = (end_time - start_time) / 1e6`<br>`Score_Idle = clamp(Duration_Sec / 3600.0, 0.0, 1.0)` *(for idle tasks)* | `[0.0, 1.0]` | Tasks dormant for longer periods are statistically far less likely to experience sudden traffic bursts. |
| `Score_Replica` | Replica Redundancy | 0.10 | `collection_id`, `instance_index` | Group by `collection_id`, count active `instance_index`:<br>• `>= 4 instances` → `1.0`<br>• `3 instances` → `0.8`<br>• `2 instances` → `0.5`<br>• `<= 1 instance` → `0.0` | `[0.0, 1.0]` | Reclaiming an instance from a multi-replica service does not cause total service downtime. |
| `Score_Priority` | Priority Inversion | 0.05 | `priority` | `Score_Priority = 1.0 - clamp(priority / 450.0, 0.0, 1.0)` | `[0.0, 1.0]` | Lower-priority batch jobs are intended to yield resources to high-priority services. |
| `Score_State` | Workload Statefulness | 0.05 | `scheduling_class` | Discrete mapping based on Borg latency tier:<br>• Class 0 (Batch / non-prod) → `1.0`<br>• Class 1 (Internal service) → `0.7`<br>• Class 2 (Mid-tier service) → `0.4`<br>• Class 3 (User-facing latency-critical) → `0.1` | `[0.0, 1.0]` | Stateless batch tasks tolerate interruption and checkpointing significantly better than stateful DBs. |
| `Score_PDB` | Disruption Budget Headroom | 0.05 | `constraint`, `collections_events_type` | • No restrictive constraint → `1.0`<br>• Single allowed disruption → `0.7`<br>• Hard constraint (pinned/isolated) → `0.0` | `[0.0, 1.0]` | Ensures safety constraints are respected before selecting a candidate for eviction/checkpointing. |
| `Score_Checkpoint` | Checkpoint Compatibility | 0.05 | `vertical_scaling` | • Dynamic scaling enabled (`vertical_scaling > 0`) → `1.0`<br>• Static reservation → `0.5`<br>• Incompatible runtime / lock → `0.0` | `[0.0, 1.0]` | Workloads that support live checkpoint/restore (CRIU) or in-place resizing receive higher scores. |

---

## 4. Ground Truth Formulation (Target Variable)

Because the trace does not contain an explicit column labeled "reclamation score", we formulate the objective target based on observed production outcomes.

### Objective: Safe Reclamation Utility

A task represents an **ideal reclamation candidate** if:
1. It wastes substantial resources (high CPU and Memory waste).
2. It is not mission-critical (`priority` is low).
3. Reclaiming, throttling, or evicting it does not cause catastrophic failures (`failed == 0`).

---

### Selected Ground Truth: Formulation B (Continuous Net Reclaim Utility)

We select **Formulation B (Continuous Net Reclaim Utility)** as the primary target variable for model training.

#### Mathematical Definition:
```text
Net_Utility = [(CPU_Waste + Mem_Waste) / 2] * (1.0 - priority / 450.0) * (1.0 - failed)
```

#### Why Dividing by 450.0?
* In Google Borg production clusters, priority values range strictly from `0` (low-priority batch/offline processing) to `450` (system-critical infrastructure daemons).
* Dividing by `450.0` normalizes priority into `[0.0, 1.0]`, allowing it to act as an exact safety penalty:
  * **Batch pod (`priority = 0`):** `1.0 - (0 / 450) = 1.0` (zero penalty, full reclamation reward).
  * **Critical system pod (`priority = 450`):** `1.0 - (450 / 450) = 0.0` (utility drops to zero, blocking eviction).
* If a pod failed during the observation window (`failed == 1`), its net utility is immediately penalized to `0`.

#### Why Formulation B is Chosen Over Classification (Formulation A):
1. **Mathematical Consistency:** The Kubernetes decision engine computes a continuous score in `[0.0, 1.0]` for tiered decisions (Full Reclaim >= 0.75, Soft Reclaim >= 0.50, No Action < 0.50). A regression target models this continuous hierarchy directly.
2. **No Arbitrary Cutoffs:** Avoids artificial binary thresholds (such as declaring 40% waste "safe" and 39% "unsafe").
3. **Formulation A Retained for Validation:** We retain Formulation A (binary classification) as an independent sanity check to evaluate ROC-AUC and confirm that the learned weights effectively separate safe tasks from failing tasks.

---

## 5. Machine Learning Methodology: Ridge Regression (Selected Primary)

### Selected Primary Method: Ridge Regression with Non-Negative Least Squares (NNLS)

We train a regularized linear model with non-negative constraints to predict `Net_Utility`:

```text
Net_Utility = beta_1 * Score_CPU
            + beta_2 * Score_Mem
            + beta_3 * Score_Idle
            + beta_4 * Score_Benefit
            + beta_5 * Score_Replica
            + beta_6 * Score_Priority
            + beta_7 * Score_State
            + beta_8 * Score_PDB
            + beta_9 * Score_Checkpoint
```

### Key Technical Justifications for Ridge Regression:

1. **Exact 1-to-1 Equivalence with Go Engine:**
   The production Kubernetes scheduler in `engine.go` evaluates a linear addition:
   `totalScore = WeightCPU*scoreCPU + WeightMem*scoreMem + ...`
   Fitting a linear model ensures that the learned coefficients `beta_i` directly map to the weight multipliers in Go without mathematical translation errors.
2. **Handles Multicollinearity:**
   In container traces, CPU waste and Memory waste are strongly correlated. Ordinary Least Squares (OLS) can produce erratic, overfitted weights. Ridge Regression's L2 regularization penalty stabilizes coefficient estimation across correlated metrics.
3. **Guarantees Valid Weights ($W_i \ge 0$ and $\sum W_i = 1.00$):**
   Using non-negative bounds prevents negative weights (which would counterintuitively penalize high waste). Normalizing the coefficients yields:
   ```text
   Weight_i = beta_i / sum(beta_k for all k)
   ```
   Ensuring all weights sum to exactly `1.00`.
4. **Sub-Microsecond Latency in the Control Plane:**
   Complex models (like Neural Networks or Tree ensembles) would require embedding heavy C++ runtimes or Python microservices into the Kubernetes control plane, adding 20–50ms scheduling latency per pod. Deriving optimal linear weights offline allows the Go scheduler to execute decisions in pure native Go in `< 0.001 ms`.

### Benchmark Model: XGBoost (Comparative Baseline)
To demonstrate rigorous methodology for evaluation and project presentation:
* We also train an **XGBoost Regressor** on the exact same dataset.
* We compare the R-squared score of the Linear Ridge model against the non-linear XGBoost model.
* **Expected Outcome:** Demonstrating that the lightweight linear formula captures ~95%+ of the predictive power of a complex tree ensemble while maintaining microsecond evaluation speed.

---

## 6. Implementation Workflow

```text
+-------------------------------------------------------------+
|                 dataset/borg_traces_data.csv                |
|             (1,324,696 Google Borg Trace Records)           |
+-------------------------------------------------------------+
                              |
                              v  (scripts/train_weights.py)
+-------------------------------------------------------------+
|  1. Parse nested JSON fields (request, usage)               |
|  2. Compute 9 Decision Factors in [0.0, 1.0]                |
|  3. Compute Formulation B Target: Net_Utility               |
+-------------------------------------------------------------+
                              |
              +---------------+---------------+
              |                               |
              v (Primary)                     v (Benchmark)
+-----------------------------+ +-----------------------------+
|   Ridge Regression (NNLS)   | |      XGBoost Regressor      |
|  Learns linear coefficients | |   Evaluates non-linear upper|
|   beta_1 ... beta_9         | |   bound R^2 & SHAP values   |
+-----------------------------+ +-----------------------------+
              |                               |
              v                               v
+-----------------------------+ +-----------------------------+
|   Normalize: sum(W_i) = 1.0 | |    Compare R^2 Metrics      |
+-----------------------------+ +-----------------------------+
              |
              v
+-------------------------------------------------------------+
|               Generate Go Struct Code Snippet               |
+-------------------------------------------------------------+
                              |
                              v
+-------------------------------------------------------------+
|            Update pkg/decision/policy.go                    |
|       Validate with: go test -v ./pkg/decision/...          |
+-------------------------------------------------------------+
```

---

## 7. Deliverables & Next Steps

1. **Python Training Script (`scripts/train_weights.py`):**
   * Streams/samples 100,000 rows from `dataset/borg_traces_data.csv`.
   * Computes the 9 normalized features and the `Net_Utility` target.
   * Fits Ridge Regression with non-negative constraints.
   * Fits XGBoost as the non-linear baseline.
   * Outputs comparison metrics (R-squared, MAE) and prints the ready-to-paste Go struct.
2. **Go Code Update:**
   * Update `DefaultPolicy()` in `pkg/decision/policy.go` with the trained weights.
3. **Verification:**
   * Run the Go test suite to ensure all unit tests pass with the data-driven weights.
