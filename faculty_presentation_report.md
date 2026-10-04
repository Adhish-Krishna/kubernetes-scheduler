# Multi-Criteria Reclamation Weight Derivation Report
**Dataset:** Google Borg Production Traces (`dataset/borg_traces_data.csv` — 1.3M records)  
**Methodology:** Pearson Correlation Analysis + Shannon's Entropy Weight Method (EWM)

---

## 1. Correlation Analysis (Testing for Collinearity)

Before computing weights, we verify that the 6 factors capture distinct operational signals and are not duplicate collinear measurements.

![Correlation Heatmap](C:\Users\Darshan V G\.gemini\antigravity-ide\brain\7fce125e-2960-4e44-94d9-30634590e164\correlation_heatmap.png)

### Pearson Correlation Matrix Table

| Factor | `score_cpu` | `score_memory` | `score_benefit` | `score_priority` | `score_state` | `score_replica` |
|---|---|---|---|---|---|---|
| `score_cpu` | **1.0000** | 0.5524 | 0.2709 | 0.1992 | 0.0769 | 0.0063 |
| `score_memory` | 0.5524 | **1.0000** | 0.4039 | 0.3971 | 0.2889 | -0.0053 |
| `score_benefit` | 0.2709 | 0.4039 | **1.0000** | 0.0354 | -0.0513 | 0.0032 |
| `score_priority` | 0.1992 | 0.3971 | 0.0354 | **1.0000** | 0.5065 | -0.0049 |
| `score_state` | 0.0769 | 0.2889 | -0.0513 | 0.5065 | **1.0000** | 0.0016 |
| `score_replica` | 0.0063 | -0.0053 | 0.0032 | -0.0049 | 0.0016 | **1.0000** |

> [!NOTE]
> **Collinearity Finding:** Every distinct pair has $|r| \le 0.552$. No severe collinearity ($|r| > 0.70$) exists. All 6 factors provide distinct, non-redundant operational information.

---

## 2. Objective Weights via Entropy Weight Method (EWM)

Using Claude Shannon's Information Entropy (1948), the weight of each factor is determined strictly by its **information contrast / dispersion** across the cluster.

![Entropy Weights Bar Chart](C:\Users\Darshan V G\.gemini\antigravity-ide\brain\7fce125e-2960-4e44-94d9-30634590e164\entropy_weights.png)

### Step-by-Step EWM Calculation Table

| Factor Symbol | Factor Name | Information Entropy ($e_j$) | Diversity Degree ($d_j = 1 - e_j$) | Derived Weight ($W_j$) | Weight Share (%) |
|---|---|---|---|---|---|
| `Score_Benefit` | Reclaimable Volume | 0.956009 | 0.043991 | **0.409373** | **40.94%** |
| `Score_Memory` | Memory Waste | 0.975731 | 0.024269 | **0.225841** | **22.58%** |
| `Score_CPU` | CPU Waste | 0.978540 | 0.021460 | **0.199702** | **19.97%** |
| `Score_State` | Latency / Statefulness | 0.989421 | 0.010579 | **0.098443** | **9.84%** |
| `Score_Priority` | Workload Inversion | 0.993128 | 0.006872 | **0.063951** | **6.40%** |
| `Score_Replica` | Service Redundancy | 0.999711 | 0.000289 | **0.002690** | **0.27%** |
| **SUM** | — | — | **0.107460** | **1.000000** | **100.00%** |

---

## 3. Key Justifications for Faculty Review

1. **Why does `Score_Benefit` have the highest weight (40.94%)?**
   - In production clusters, containers have vastly different request sizes (some ask for 50 mCPU, others ask for 8 cores).
   - `Score_Benefit` has the highest variance (lowest entropy $e_j = 0.956$), meaning it provides the greatest contrast for identifying high-impact reclamation candidates.
2. **Why do `Score_CPU` and `Score_Memory` have ~20% and ~23%?**
   - Both show consistent variation between heavily over-provisioned batch jobs and tightly packed services.
3. **Mathematical Guarantees:**
   - All weights are non-negative ($W_j \ge 0$).
   - All weights sum strictly to $1.000000$ ($100.00\%$).
   - **Zero subjective guessing:** No synthetic formulas or human-invented target multipliers were used.
