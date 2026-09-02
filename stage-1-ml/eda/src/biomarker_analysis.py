import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import math

from src.visualization import save_plot

def identify_biomarker_cols(df: pd.DataFrame) -> list:
    """Identifies potential biomarker/genomic columns."""
    keywords = ['mutation', 'gene', 'biomarker', 'ctdna', 'marker', 'genomic']
    biomarkers = []
    for col in df.columns:
        if any(kw in col.lower() for kw in keywords):
            biomarkers.append(col)
    return biomarkers

def analyze_biomarkers(df: pd.DataFrame, target: str):
    """Analyzes biomarkers and generates visualizations."""
    biomarker_cols = identify_biomarker_cols(df)
    if not biomarker_cols:
        return {}
        
    num_biomarkers = df[biomarker_cols].select_dtypes(include=[np.number]).columns.tolist()
    cat_biomarkers = df[biomarker_cols].select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    
    # Analyze categorical biomarkers
    if cat_biomarkers:
        cols_per_row = 2
        rows = math.ceil(len(cat_biomarkers) / cols_per_row)
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 5 * rows))
        if rows == 1 and cols_per_row == 1: axes = np.array([axes])
        axes = axes.flatten()
        
        for i, col in enumerate(cat_biomarkers):
            if df[col].nunique() < 20:
                crosstab = pd.crosstab(df[col], df[target], normalize='index') * 100
                cols = [c for c in ['Low', 'Moderate', 'High'] if c in crosstab.columns]
                crosstab = crosstab[cols]
                crosstab.plot(kind='bar', stacked=True, ax=axes[i], colormap='viridis')
                axes[i].set_title(f"{col} vs {target}")
                axes[i].set_ylabel("Percentage (%)")
            
        for i in range(len(cat_biomarkers), len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        save_plot(fig, 'biomarker_categorical_analysis.png')
        
    # Analyze numerical biomarkers
    if num_biomarkers:
        cols_per_row = 3
        rows = math.ceil(len(num_biomarkers) / cols_per_row)
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 4 * rows))
        if rows == 1 and cols_per_row == 1: axes = np.array([axes])
        axes = axes.flatten()
        
        for i, col in enumerate(num_biomarkers):
            sns.boxplot(data=df, x=target, y=col, ax=axes[i], order=['Low', 'Moderate', 'High'], palette='viridis')
            axes[i].set_title(f"{col} by {target}")
            
        for i in range(len(num_biomarkers), len(axes)):
            axes[i].axis('off')
            
        plt.tight_layout()
        save_plot(fig, 'biomarker_numerical_analysis.png')
        
    return {'numerical_biomarkers': num_biomarkers, 'categorical_biomarkers': cat_biomarkers}
