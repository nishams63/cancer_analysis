"""
Visualization Module for Stage 4 EDA.
Renders 12 publication-quality statistical and clinical figures (300 DPI)
per Section 20 specifications.
"""

from pathlib import Path
from typing import Dict, Any, List
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


class FigureGenerator:
    """Generates 12 publication-quality PNG charts for the clinical data readiness audit."""

    def __init__(self, figures_dir: str):
        self.figures_dir = Path(figures_dir)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        # Apply standard clinical publication styling
        sns.set_theme(style="whitegrid", palette="muted")
        plt.rcParams.update({
            "font.family": "sans-serif",
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 10,
            "figure.titlesize": 14
        })

    def generate_all_figures(
        self,
        df: pd.DataFrame,
        token_arrays: Dict[str, np.ndarray],
        risk_data: Dict[str, Any],
        entity_density_data: Dict[str, Any],
        entity_retention_data: Dict[str, Any],
        vocab_data: Dict[str, Any],
        negation_data: Dict[str, Any],
        split_data: Dict[str, Any],
        stratified_risk_loss: Dict[str, Any]
    ) -> List[str]:
        """Generates and saves all 12 required figures."""
        saved_paths = []

        # 1. source_token_length.png
        p1 = self.figures_dir / "source_token_length.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(token_arrays["source"], bins=40, kde=True, color="#1f77b4", ax=ax)
        ax.set_title(f"Source Clinical Note Token Length (BPE) (N={len(token_arrays['source'])})")
        ax.set_xlabel("Token Count (cl100k_base)")
        ax.set_ylabel("Document Frequency")
        p95 = np.percentile(token_arrays["source"], 95)
        ax.axvline(p95, color="#d62728", linestyle="--", label=f"P95: {int(p95)} tokens")
        ax.legend()
        plt.tight_layout()
        plt.savefig(p1, dpi=300)
        plt.close(fig)
        saved_paths.append(str(p1))

        # 2. target_token_length.png
        p2 = self.figures_dir / "target_token_length.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.histplot(token_arrays["target"], bins=30, kde=True, color="#2ca02c", ax=ax)
        ax.set_title(f"Combined Target Token Length (Risk + Finding + Action) (N={len(token_arrays['target'])})")
        ax.set_xlabel("Target Tokens (cl100k_base)")
        ax.set_ylabel("Count")
        med = np.median(token_arrays["target"])
        ax.axvline(med, color="#ff7f0e", linestyle="--", label=f"Median: {int(med)} tokens")
        ax.legend()
        plt.tight_layout()
        plt.savefig(p2, dpi=300)
        plt.close(fig)
        saved_paths.append(str(p2))

        # 3. token_length_comparison.png
        p3 = self.figures_dir / "token_length_comparison.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        comp_df = pd.DataFrame({
            "Source Note": token_arrays["source"],
            "Target Output": token_arrays["target"],
            "Full Sequence": token_arrays["full_sequence"]
        })
        sns.boxplot(data=comp_df, palette="Blues", ax=ax)
        ax.set_title(f"Token Length Comparison Across Prompt Components (N={len(comp_df)})")
        ax.set_ylabel("Tokens (cl100k_base)")
        plt.tight_layout()
        plt.savefig(p3, dpi=300)
        plt.close(fig)
        saved_paths.append(str(p3))

        # 4. risk_distribution.png
        p4 = self.figures_dir / "risk_distribution.png"
        fig, ax = plt.subplots(figsize=(7, 5))
        counts = risk_data["counts"]
        tiers = list(counts.keys())
        vals = [counts[t] for t in tiers]
        colors = ["#2ca02c", "#ff7f0e", "#d62728"]
        bars = ax.bar(tiers, vals, color=colors, edgecolor="black", alpha=0.85)
        ax.set_title(f"Target Clinical Risk Tier Distribution (N={sum(vals)})")
        ax.set_ylabel("Number of Encounters")
        for bar in bars:
            yval = bar.get_height()
            pct = (yval / sum(vals)) * 100
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f"{yval}\n({pct:.1f}%)", ha="center", va="bottom", fontsize=10)
        ax.set_ylim(0, max(vals) * 1.18)
        plt.tight_layout()
        plt.savefig(p4, dpi=300)
        plt.close(fig)
        saved_paths.append(str(p4))

        # 5. entity_density_distribution.png
        p5 = self.figures_dir / "entity_density_distribution.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        dens_arrays = entity_density_data.get("density_arrays", {})
        dens_df = pd.DataFrame(dens_arrays)
        dens_df.rename(columns={"gene": "Gene", "drug": "Drug", "dosage": "Dosage", "adverse_event": "Adverse Event"}, inplace=True)
        sns.boxplot(data=dens_df, palette="Set2", ax=ax)
        ax.set_title("Entity Density per Clinical Note by Category")
        ax.set_ylabel("Count of Verified Entities")
        plt.tight_layout()
        plt.savefig(p5, dpi=300)
        plt.close(fig)
        saved_paths.append(str(p5))

        # 6. entity_density_by_risk.png
        p6 = self.figures_dir / "entity_density_by_risk.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        risk_tiers = ["Low", "Moderate", "High"]
        means = [stratified_risk_loss.get(t, {}).get("average_source_entities", 0.0) for t in risk_tiers]
        ax.bar(risk_tiers, means, color="#1f77b4", edgecolor="black", alpha=0.8)
        ax.set_title("Average Entity Density Stratified by Risk Tier")
        ax.set_ylabel("Average Source Entities per Note")
        for i, v in enumerate(means):
            ax.text(i, v + 0.1, f"{v:.2f}", ha="center", va="bottom")
        ax.set_ylim(0, max(means or [1]) * 1.2)
        plt.tight_layout()
        plt.savefig(p6, dpi=300)
        plt.close(fig)
        saved_paths.append(str(p6))

        # 7. entity_retention_by_type.png
        p7 = self.figures_dir / "entity_retention_by_type.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        cats = entity_retention_data.get("categories", {})
        cat_names = [c.capitalize().replace("_", " ") for c in cats.keys()]
        ret_rates = [cats[c]["retention_rate"] * 100.0 for c in cats.keys()]
        bars = ax.bar(cat_names, ret_rates, color="#9467bd", edgecolor="black", alpha=0.85)
        ax.set_title("Reference Entity Retention Rate in Generated Targets by Type")
        ax.set_ylabel("Retention Rate (%)")
        ax.set_ylim(0, 115)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha="center", va="bottom", fontsize=10)
        ax.axhline(95.0, color="orange", linestyle="--", label="Warning Threshold (95%)")
        ax.axhline(90.0, color="red", linestyle="--", label="Critical Threshold (90%)")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(p7, dpi=300)
        plt.close(fig)
        saved_paths.append(str(p7))

        # 8. entity_retention_by_risk.png
        p8 = self.figures_dir / "entity_retention_by_risk.png"
        fig, ax = plt.subplots(figsize=(8, 5))
        ret_by_risk = [stratified_risk_loss.get(t, {}).get("entity_retention_rate", 1.0) * 100.0 for t in risk_tiers]
        bars = ax.bar(risk_tiers, ret_by_risk, color="#e377c2", edgecolor="black", alpha=0.85)
        ax.set_title("Entity Retention Rate Stratified Across Risk Classes")
        ax.set_ylabel("Target Retention Rate (%)")
        ax.set_ylim(0, 115)
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha="center", va="bottom", fontsize=10)
        plt.tight_layout()
        plt.savefig(p8, dpi=300)
        plt.close(fig)
        saved_paths.append(str(p8))

        # 9. critical_term_frequency.png
        p9 = self.figures_dir / "critical_term_frequency.png"
        fig, ax = plt.subplots(figsize=(10, 6))
        terms_list = vocab_data.get("terms", [])[:15]  # top 15 terms
        term_labels = [t["term"] for t in terms_list]
        term_freqs = [t["frequency"] for t in terms_list]
        ax.barh(term_labels[::-1], term_freqs[::-1], color="#bcbd22", edgecolor="black", alpha=0.85)
        ax.set_title("Corpus Frequency of Curated Critical Clinical Terms")
        ax.set_xlabel("Occurrence Count in Clinical Notes")
        plt.tight_layout()
        plt.savefig(p9, dpi=300)
        plt.close(fig)
        saved_paths.append(str(p9))

        # 10. critical_term_fragmentation.png
        p10 = self.figures_dir / "critical_term_fragmentation.png"
        fig, ax = plt.subplots(figsize=(10, 6))
        frag_terms = sorted(vocab_data.get("terms", []), key=lambda x: x["fragmentation_ratio"], reverse=True)[:15]
        labels = [f"{t['term']} ({t['category']})" for t in frag_terms]
        ratios = [t["fragmentation_ratio"] for t in frag_terms]
        colors = ["#d62728" if r >= 3.0 else "#1f77b4" for r in ratios]
        ax.barh(labels[::-1], ratios[::-1], color=colors[::-1], edgecolor="black", alpha=0.85)
        ax.set_title("BPE Subword Fragmentation Ratio for Critical Terms")
        ax.set_xlabel("Tokens per Word (Fragmentation Ratio)")
        ax.axvline(3.0, color="red", linestyle="--", label="High Fragmentation Threshold (3.0)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(p10, dpi=300)
        plt.close(fig)
        saved_paths.append(str(p10))

        # 11. negation_analysis.png
        p11 = self.figures_dir / "negation_analysis.png"
        fig, ax = plt.subplots(figsize=(7, 5))
        pres = negation_data.get("correctly_preserved", 0)
        flips = negation_data.get("negation_flips", 0)
        neg_labels = ["Correctly Preserved", "Negation Flips"]
        neg_vals = [pres, flips]
        colors = ["#2ca02c", "#d62728"]
        bars = ax.bar(neg_labels, neg_vals, color=colors, edgecolor="black", alpha=0.85)
        ax.set_title(f"Clinical Negation Polarity Fidelity (Total={pres+flips})")
        ax.set_ylabel("Entity Occurrences")
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + (max(neg_vals)*0.02), f"{yval}", ha="center", va="bottom")
        max_val = max(neg_vals) if max(neg_vals) > 0 else 10
        ax.set_ylim(0, max_val * 1.18)
        plt.tight_layout()
        plt.savefig(p11, dpi=300)
        plt.close(fig)
        saved_paths.append(str(p11))

        # 12. split_distribution.png
        p12 = self.figures_dir / "split_distribution.png"
        fig, ax = plt.subplots(figsize=(7, 5))
        s_dist = split_data.get("split_distribution", {})
        splits = list(s_dist.keys())
        recs = [s_dist[s]["records"] for s in splits]
        colors = ["#17becf", "#9edae5", "#3182bd"]
        bars = ax.bar(splits, recs, color=colors, edgecolor="black", alpha=0.85)
        ax.set_title(f"Cohort Split Distribution (Total Records: {sum(recs)})")
        ax.set_ylabel("Record Count")
        for bar in bars:
            yval = bar.get_height()
            pct = (yval / sum(recs)) * 100.0
            ax.text(bar.get_x() + bar.get_width()/2.0, yval + 50, f"{yval}\n({pct:.1f}%)", ha="center", va="bottom")
        ax.set_ylim(0, max(recs) * 1.18)
        plt.tight_layout()
        plt.savefig(p12, dpi=300)
        plt.close(fig)
        saved_paths.append(str(p12))

        return saved_paths
