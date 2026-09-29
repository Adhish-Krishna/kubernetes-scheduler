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
* We train an **XGBoost Regressor** on the exact same dataset as a high-capacity, non-linear baseline.
* We compare the predictive accuracy and feature importance rankings between the linear model and the tree ensemble to understand non-linear workload interactions.

---

### Ensemble Blending: 50/50 Hybrid Model (Final Weights)

To combine the mathematical guarantees of linear models with the non-linear safety awareness of gradient-boosted trees, the final decision weights are derived using an **Equal-Weighted Model Averaging Ensemble**:

#### 1. Mathematical Blending Concept:
For each decision factor $i$:
```text
Weight_Blended[i] = 0.5 * Weight_Linear[i] + 0.5 * Weight_XGBoost[i]
```

#### 2. Conceptual Roles of the Two Models:

| Feature Symbol | Factor Name | Role in Linear Model (Ridge NNLS) | Role in Tree Model (XGBoost) | Why Blending is Conceptually Superior |
|---|---|---|---|---|
| `score_cpu` | CPU Waste | Measures direct linear slope of CPU headroom saved. | Evaluates CPU waste in combination with workload priority. | Captures both direct core savings and operational safety. |
| `score_memory` | Memory Waste | Captures direct linear value of reclaimed RAM. | Identifies memory as the primary uncompressible constraint. | Balances high volume reclamation with cluster stability. |
| `score_benefit` | Reclaim Benefit | Prioritizes absolute core/memory volume recovered. | Evaluates volume threshold necessary to justify an action. | Prevents reclaiming trivial workloads while rewarding large gains. |
| `score_idle` | Idle Duration | Estimates average linear contribution of uptime. | Evaluates conditional dormancy (idle only matters if waste exists). | Tree model prevents premature reclamation of spiky containers. |
| `score_replica` | Replica Redundancy | Linear availability guardrail. | Evaluates redundancy as an availability prerequisite. | Blending ensures single-replica workloads remain protected. |
| `score_priority` | Priority Inversion | Continuous linear penalty for high priority. | Acts as a sharp step-function cutoff gate for critical pods. | Prevents eviction of system daemons while allowing batch yield. |
| `score_state` | Workload Statefulness | Linear risk adjustment for batch vs. services. | Splits decisions based on latency sensitivity tiers. | Strongly protects stateful, latency-sensitive services. |
| `score_pdb` | Disruption Budget | Linear disruption headroom credit. | Enforces non-disruption availability rules. | Guarantees compliance with Kubernetes disruption policies. |
| `score_checkpoint` | Checkpointability | Linear capability multiplier. | Action selector (full reclaim vs. soft reclaim fallback). | Weights workloads that support CRIU live state preservation. |

#### 3. Exact 9-Decimal Normalization & Delta Compensation Concept:
In computer arithmetic, floating-point rounding at 9 decimal places can introduce a microscopic drift (e.g., $0.999999999$ or $1.000000001$). 
To ensure strict mathematical consistency across the scheduler:
1. Each model's coefficients are normalized so their individual sums equal $1.0$.
2. The 50/50 arithmetic blend is rounded to 9 decimal places.
3. The remaining rounding difference ($\Delta = 1.0 - \sum \text{Weights}$) is added to the largest single weight.
4. **Result:** `sum(Weight_Blended) == 1.000000000` strictly, eliminating any floating-point drift.

#### 4. Theoretical & Statistical Justification for 50/50 Blending:
* **Bivariate Complementarity:** Linear models measure **first-order resource volume** (cores and gigabytes saved), whereas Tree models measure **higher-order safety gates** (idle duration and priority thresholds).
* **Elimination of Model Blind Spots:** Linear models can under-weight conditional metrics like idle duration, while tree models can under-weight continuous metrics like replica availability. Averaging both models eliminates individual blind spots without human bias.
* **Control Plane Performance:** Offline ensembling produces a fixed set of weights that compile into pure Go constants, delivering **sub-microsecond ($< 0.001\text{ ms}$)** evaluation speed in the Kubernetes control plane.

---

## 6. Implementation Workflow

```text
+-------------------------------------------------------------+
|                 dataset/borg_traces_data.csv                |
|             (Google Borg Production Trace Records)          |
+-------------------------------------------------------------+
                              |
                              v  (ml/extract.py)
+-------------------------------------------------------------+
|  1. Fast Vectorized Regex Extraction (JSON requests/usages) |
|  2. Compute 9 Decision Factors in [0.0, 1.0]                |
|  3. Compute Multi-Criteria Target: Net_Utility              |
+-------------------------------------------------------------+
                              |
              +---------------+---------------+
              |                               |
              v                               v
+-----------------------------+ +-----------------------------+
|   Bounded Ridge NNLS (50%)  | |    XGBoost Regressor (50%)  |
| Linear resource scaling     | | Non-linear safety thresholds|
| (First-order volume slope)  | | (Higher-order decision tree)|
+-----------------------------+ +-----------------------------+
              |                               |
              +---------------+---------------+
                              |
                              v  (ml/train.py)
+-------------------------------------------------------------+
|          50/50 Ensemble Model Averaging Blending            |
|       9-Decimal Normalization & Delta Compensation          |
|                 sum(W_i) = 1.000000000                      |
+-------------------------------------------------------------+
                              |
                              v  (ml/export.py)
+-------------------------------------------------------------+
|    Export ml/artifacts/trained_weights.json & Go Struct     |
+-------------------------------------------------------------+
                              |
              +---------------+---------------+
              |                               |
              v (Tier 1: Compiled)            v (Tier 2: Dynamic)
+-----------------------------+ +-----------------------------+
|  pkg/decision/policy.go     | |  pkg/decision/weights.go    |
|  Native Go Defaults (< 1us) | |  ConfigMap / JSON loader    |
+-----------------------------+ +-----------------------------+
```

---

## 7. Deliverables & Production Assets

1. **Python Training Pipeline (`ml/`):**
   * `ml/extract.py`: Fast vectorized parser for Borg traces into 9 normalized features.
   * `ml/train.py`: Fits Bounded NNLS + XGBoost and computes 9-decimal blended ensemble weights.
   * `ml/evaluate.py`: Generates R^2, MAE, and comparison metrics.
   * `ml/export.py`: Serializes models and exports `trained_weights.json`.
   * `ml/run_training.py`: Single-command master orchestrator (`python ml/run_training.py --all`).
2. **Weight Storage Artifacts (`ml/artifacts/`):**
   * `ml/artifacts/trained_weights.json`: Version-controlled JSON with metadata and 9-decimal weights.
   * `ml/artifacts/training_metrics.json`: R^2, MAE, and evaluation benchmarks.
   * `ml/artifacts/ridge_model.joblib`: Serialized Scikit-learn model.
   * `ml/artifacts/xgboost_model.json`: Serialized XGBoost tree ensemble.
3. **Scheduler Integration:**
   * `adaptive-k8s-scheduler/pkg/decision/policy.go`: Compiled 9-decimal native defaults in `DefaultPolicy()`.
   * `adaptive-k8s-scheduler/pkg/decision/weights.go`: Dynamic ConfigMap JSON loader with validation.
   * `adaptive-k8s-scheduler/pkg/decision/weights_test.go`: Unit tests confirming sum == 1.000000000 and JSON loading.
