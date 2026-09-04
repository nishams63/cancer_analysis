# Stage 2 Deep Learning: Independent Zero-Leakage Audit

> **MANDATORY NOTICE:** This model was developed using synthetic data and has not been clinically validated. Performance on this dataset does not establish clinical effectiveness, safety, or real-world patient benefit.
> $$\text{Synthetic data != Real patient evidence != Clinical validation}$$

## 1. Audit Summary

| Audit Domain | Verification Target | Observed Result | Status |
|---|---|---|:---:|
| **Patient Cohort Disjointness** | Train, Val, Test disjoint sets | Train=700, Val=150, Test=150 (Overlap=0) | **PASS** |
| **Visual Tile Patient Alignment** | 1,800 test tiles mapped to 150 test patients | 1,800 tiles, exactly 150 patients, 0 foreign tiles | **PASS** |
| **Temporal Patient Alignment** | 2,403 test observations mapped to 150 test patients | 2,403 rows, exactly 150 patients, 0 foreign rows | **PASS** |
| **Forecasting Horizon Boundary** | Historical inputs strictly $\le 90$ days | 1,088 historical visits, 1,315 future visits, 0 boundary violations | **PASS** |
| **ctDNA 30d Velocity Formulation** | Velocity $(t - (t-1))/\Delta t$ strictly backward-looking | Monitored across all sequences, 0 forward lookahead | **PASS** |
| **Scaler / Normalization Isolation** | Checkpoint parameters trained exclusively on Train | Exact match to train moments; 0 test exposure | **PASS** |
| **Supervised Target Isolation** | Targets excluded from model inputs | Extracted strictly as labels, excluded from sequence tensor | **PASS** |

## 2. Mathematical Proof of Feature Boundary Isolation

For any observation $x_{i,t}$ in the temporal input matrix $\mathbf{X}_i$ of patient $i$:
$$\forall t, \quad \text{days}_t \le 90 \iff \text{is\_input\_window}_t = 1$$
The ctDNA velocity feature is formulated as:
$$\text{ctDNA\_velocity\_30d}_t = \begin{cases} 0.0 & \text{if } t = 0 \\ \frac{\text{ctDNA}_t - \text{ctDNA}_{t-1}}{\text{days}_t - \text{days}_{t-1}} \times 30.0 & \text{if } t > 0 \end{cases}$$
Because $t \le 90$ and $t-1 < t \le 90$, the calculation uses exclusively historical or contemporaneous observations. No information from the future window ($t > 90$) enters the input tensor.
