# Stage 5 Rare Combination Space & Stress-Test Scenarios

**Execution Timestamp**: 2026-09-11T16:42:59.443670+00:00  
**Catalog Path**: `C:/Users/Nallu_PC/.gemini/antigravity/scratch/cancer_analysis/stage 5 Gen-AI/Data Engineer/configs/rare_combination_space.yaml`  
**Total Mined Scenarios**: 6  

---

## 1. Rarity Formulation & Mathematical Definition
A synthetic clinical scenario is classified into the rare stress-test space if and only if it satisfies:

$$\text{Individually Plausible} \land \text{Jointly Rare} \land \text{Allowed by Project Rules}$$

### Configured Frequency Thresholds:
- **Common**: $f \ge 0.10$
- **Uncommon**: $0.03 \le f < 0.10$
- **Rare**: $0.005 \le f < 0.03$
- **Very Rare**: $f < 0.005$

---

## 2. Mined Rare Edge-Case Stress Scenarios

| Scenario ID | Scenario Name | Primary Features | Joint Frequency | Rarity Tier | Stress Dimension |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `RC-001` | Rare dual driver: EGFR sensitizing + MET bypass amplification | EGFR, MET | 0.0020 | `very_rare` | Acquired kinase bypass resistance with targeted therapy dilemma |
| `RC-002` | Uncommon dual MAPK driver: KRAS activating + BRAF V600E | KRAS, BRAF | 0.0035 | `rare` | Co-occurring downstream MAPK pathway hyperactivation |
| `RC-003` | Dual driver conflict: EGFR L858R + KRAS G12C | EGFR, KRAS | 0.0048 | `rare` | Primary resistance to single-agent EGFR TKI monotherapy |
| `RC-004` | Rare compound variant: ALK fusion + concurrent PIK3CA activating | ALK, PIK3CA | 0.0018 | `very_rare` | Parallel oncogenic pathway bypass resistance |
| `RC-005` | High Mutational Burden with Severe Baseline Renal Dysfunction | TP53 | 0.0041 | `rare` | Clinical contraindication dilemma: urgent tumor burden vs toxic nephropathy |
| `RC-006` | Severe ctDNA Velocity Spike under Target-Negative Phenotype | None/Unknown | 0.0062 | `uncommon` | Fast progression with lack of targetable molecular actionable alterations |

---

## 3. Mandatory Engineering & Clinical Disclaimers
> **IMPORTANT CLINICAL & SCIENTIFIC PRINCIPLES:**
> 1. **Rare does not mean clinically severe**: A statistically rare mutation combination (e.g. KRAS + BRAF V600E) may present with mild or standard symptoms, while a common single alteration (e.g. KRAS alone with poor performance status) can be rapidly fatal.
> 2. **Rare does not mean difficult for the AI system**: AI system difficulty depends on model epistemic uncertainty, training data coverage, and reasoning complexity, which is evaluated independently by the Evaluation Engineer.
