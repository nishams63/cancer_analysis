import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as plt_sns
import seaborn as sns
from pathlib import Path
import math

# Set style
sns.set_theme(style="whitegrid", palette="muted")

def save_plot(fig, filename: str):
    """Saves plot to visualizations directory."""
    out_dir = Path(__file__).resolve().parent.parent / 'visualizations'
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / filename, bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_target_distribution(df: pd.DataFrame, target: str):
    """Plots the distribution of the target variable."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(data=df, x=target, order=['Low', 'Moderate', 'High'], ax=ax, palette='viridis')
    ax.set_title("Toxicity Risk Distribution")
    ax.set_xlabel("Toxicity Risk")
    ax.set_ylabel("Count")
    
    # Add percentages
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        x = p.get_x() + p.get_width() / 2 - 0.05
        y = p.get_height() + 50
        ax.annotate(percentage, (x, y), ha='center')
        
    save_plot(fig, 'toxicity_risk_distribution.png')

def plot_numerical_distributions(df: pd.DataFrame, num_cols: list):
    """Plots distributions for numerical features."""
    if not num_cols: return
    cols_per_row = 3
    rows = math.ceil(len(num_cols) / cols_per_row)
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 4 * rows))
    axes = axes.flatten()
    
    for i, col in enumerate(num_cols):
        sns.histplot(df[col], kde=True, ax=axes[i], bins=30)
        axes[i].set_title(f"Distribution of {col}")
        
    # Hide empty subplots
    for i in range(len(num_cols), len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    save_plot(fig, 'numerical_distributions.png')

def plot_numerical_boxplots(df: pd.DataFrame, num_cols: list, target: str):
    """Plots boxplots for numerical features vs target."""
    if not num_cols: return
    cols_per_row = 3
    rows = math.ceil(len(num_cols) / cols_per_row)
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 4 * rows))
    axes = axes.flatten()
    
    for i, col in enumerate(num_cols):
        sns.boxplot(data=df, x=target, y=col, ax=axes[i], order=['Low', 'Moderate', 'High'], palette='viridis')
        axes[i].set_title(f"{col} by {target}")
        
    for i in range(len(num_cols), len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    save_plot(fig, 'numerical_boxplots.png')

def plot_categorical_analysis(df: pd.DataFrame, cat_cols: list, target: str):
    """Plots stacked bar charts for categorical features vs target."""
    if not cat_cols: return
    
    # Filter to interesting columns (avoid high cardinality like IDs)
    plot_cols = [c for c in cat_cols if df[c].nunique() < 20 and c != target]
    
    cols_per_row = 2
    rows = math.ceil(len(plot_cols) / cols_per_row)
    fig, axes = plt.subplots(rows, cols_per_row, figsize=(15, 5 * rows))
    if rows == 1 and cols_per_row == 1: axes = np.array([axes])
    axes = axes.flatten()
    
    for i, col in enumerate(plot_cols):
        crosstab = pd.crosstab(df[col], df[target], normalize='index') * 100
        # Reorder columns if possible
        cols = [c for c in ['Low', 'Moderate', 'High'] if c in crosstab.columns]
        crosstab = crosstab[cols]
        crosstab.plot(kind='bar', stacked=True, ax=axes[i], colormap='viridis')
        axes[i].set_title(f"{col} vs {target}")
        axes[i].set_ylabel("Percentage (%)")
        axes[i].legend(title=target, bbox_to_anchor=(1.05, 1), loc='upper left')
        
    for i in range(len(plot_cols), len(axes)):
        axes[i].axis('off')
        
    plt.tight_layout()
    save_plot(fig, 'categorical_analysis.png')

def plot_correlation_matrix(df: pd.DataFrame, num_cols: list):
    """Plots correlation matrix for numerical features."""
    if len(num_cols) < 2: return
    corr = df[num_cols].corr()
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=False, cmap='coolwarm', vmin=-1, vmax=1, ax=ax, square=True)
    ax.set_title("Correlation Matrix of Numerical Features")
    
    save_plot(fig, 'correlation_matrix.png')
