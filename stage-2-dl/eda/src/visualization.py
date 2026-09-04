"""
Stage 2 EDA - Diagnostic Visualization Module (13 Figures)
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
try:
    from . import config
except (ImportError, ValueError):
    import config

# Set overall seaborn aesthetic
sns.set_theme(style='whitegrid', font='sans-serif')
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

def plot_image_class_distribution(meta_df: pd.DataFrame, out_path: str):
    """Figure 1: image_class_distribution.png"""
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = meta_df.groupby(['split', 'class_label']).size().unstack(fill_value=0)
    # Order splits
    counts = counts.reindex(['train', 'validation', 'test'])
    # Plot grouped bar
    counts.plot(kind='bar', ax=ax, color=[config.CLASS_COLORS.get(c, '#333333') for c in counts.columns], width=0.7)
    
    ax.set_title('Pathology Tile Class Distribution across Splits (N=12,000)', weight='bold')
    ax.set_xlabel('Cohort Partition')
    ax.set_ylabel('Number of Tiles')
    ax.set_xticklabels(['Train (70%)', 'Validation (15%)', 'Test (15%)'], rotation=0)
    ax.legend(title='Pathology Class', frameon=True)
    
    # Add value annotations
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{int(height):,}',
                        (p.get_x() + p.get_width() / 2., height / 2.),
                        ha='center', va='center', fontsize=9, color='white', weight='bold')
            
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_image_examples(meta_df: pd.DataFrame, out_path: str):
    """Figure 2: image_examples.png (Representative 3x3 grid)"""
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    classes = ['benign', 'malignant', 'inflammation']
    
    for row_idx, cls in enumerate(classes):
        sub = meta_df[meta_df['class_label'] == cls]
        # Pick 3 distinct background temperatures if available
        temps = ['warm', 'neutral', 'cool']
        samples = []
        for t in temps:
            t_sub = sub[sub['background_temperature'] == t]
            if len(t_sub) > 0:
                samples.append(t_sub.sample(1, random_state=config.RANDOM_SEED).iloc[0])
            else:
                samples.append(sub.sample(1, random_state=config.RANDOM_SEED).iloc[0])
                
        for col_idx, item in enumerate(samples):
            ax = axes[row_idx, col_idx]
            p = item['image_path']
            if not os.path.exists(p):
                rel = item.get('relative_path', '')
                alt_p = config.STAGE_2_DIR / rel.replace('data-engineering/data-engineering/', 'data-engineering/')
                if alt_p.exists():
                    p = str(alt_p)
            try:
                img = Image.open(p)
                ax.imshow(img)
            except Exception:
                ax.text(0.5, 0.5, 'Image Load Error', ha='center', va='center')
            
            ax.set_title(f"{cls.upper()}\n{item['patient_id']} | {item['background_temperature']}", fontsize=10)
            ax.axis('off')
            
    plt.suptitle('Synthetic Histopathology-Like Tiles: Morphological Variations\n(Prototype Visualization; Not Clinically Validated)', 
                 weight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_image_pixel_distribution(norm_stats: Dict[str, Any], meta_df: pd.DataFrame, out_path: str):
    """Figure 3: image_pixel_distribution.png"""
    sample_df = meta_df.sample(n=min(500, len(meta_df)), random_state=config.RANDOM_SEED)
    r_list, g_list, b_list = [], [], []
    for _, row in sample_df.iterrows():
        p = row['image_path']
        if not os.path.exists(p):
            rel = row.get('relative_path', '')
            alt_p = config.STAGE_2_DIR / rel.replace('data-engineering/data-engineering/', 'data-engineering/')
            if alt_p.exists():
                p = str(alt_p)
        try:
            with Image.open(p) as img:
                arr = np.array(img, dtype=np.float32) / 255.0
                r_list.append(arr[:, :, 0].flatten()[::100])
                g_list.append(arr[:, :, 1].flatten()[::100])
                b_list.append(arr[:, :, 2].flatten()[::100])
        except Exception:
            continue
            
    r_vals = np.concatenate(r_list)
    g_vals = np.concatenate(g_list)
    b_vals = np.concatenate(b_list)
    
    fig, ax = plt.subplots(figsize=(9, 5))
    sns.kdeplot(r_vals, ax=ax, color='#d7191c', label=f"Red (Mean={norm_stats['mean_rgb_0_1'][0]:.3f}, Std={norm_stats['std_rgb_0_1'][0]:.3f})", fill=True, alpha=0.2)
    sns.kdeplot(g_vals, ax=ax, color='#2ca02c', label=f"Green (Mean={norm_stats['mean_rgb_0_1'][1]:.3f}, Std={norm_stats['std_rgb_0_1'][1]:.3f})", fill=True, alpha=0.2)
    sns.kdeplot(b_vals, ax=ax, color='#2b83ba', label=f"Blue (Mean={norm_stats['mean_rgb_0_1'][2]:.3f}, Std={norm_stats['std_rgb_0_1'][2]:.3f})", fill=True, alpha=0.2)
    
    ax.set_title('RGB Channel Pixel Intensity Distributions (Normalized [0, 1])', weight='bold')
    ax.set_xlabel('Normalized Pixel Intensity')
    ax.set_ylabel('Density')
    ax.set_xlim(0, 1)
    ax.legend(frameon=True)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_image_brightness_by_class(meta_df: pd.DataFrame, out_path: str):
    """Figure 4: image_brightness_by_class.png"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Boxplot by class
    sns.boxplot(data=meta_df, x='class_label', y='brightness_factor', hue='class_label', palette=config.CLASS_COLORS, ax=ax1, width=0.5, legend=False)
    ax1.set_title('Brightness Factor by Pathology Class', weight='bold')
    ax1.set_xlabel('Class Label')
    ax1.set_ylabel('Brightness Factor (Exposure)')
    
    # KDE by class
    for cls in config.CLASSES:
        sub = meta_df[meta_df['class_label'] == cls]
        sns.kdeplot(sub['brightness_factor'], ax=ax2, color=config.CLASS_COLORS[cls], label=cls.capitalize(), fill=True, alpha=0.15)
        
    ax2.set_title('Brightness Density Overlay (Shortcut Audit)', weight='bold')
    ax2.set_xlabel('Brightness Factor')
    ax2.set_ylabel('Density')
    ax2.legend(frameon=True)
    
    plt.suptitle('Anti-Shortcut Audit: Invariance of Brightness across Classes', weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_image_color_distribution(meta_df: pd.DataFrame, out_path: str):
    """Figure 5: image_color_distribution.png"""
    fig, ax = plt.subplots(figsize=(8, 5))
    crosstab = pd.crosstab(meta_df['class_label'], meta_df['background_temperature'], normalize='index') * 100
    temp_colors = {'cool': '#74add1', 'neutral': '#ffffbf', 'warm': '#f46d43'}
    
    crosstab.plot(kind='bar', stacked=True, ax=ax, color=[temp_colors.get(c, '#cccccc') for c in crosstab.columns], edgecolor='black', width=0.55)
    ax.set_title('Stroma Background Temperature Proportion by Class (%)', weight='bold')
    ax.set_xlabel('Pathology Class')
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(0, 105)
    ax.set_xticklabels([c.capitalize() for c in crosstab.index], rotation=0)
    ax.legend(title='Background Temp', frameon=True, bbox_to_anchor=(1.02, 1), loc='upper left')
    
    for n, c in enumerate(crosstab.index):
        cum = 0
        for col in crosstab.columns:
            val = crosstab.loc[c, col]
            if val > 5:
                ax.text(n, cum + val / 2, f"{val:.1f}%", ha='center', va='center', fontsize=9, weight='bold')
            cum += val
            
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_image_stain_variation(meta_df: pd.DataFrame, out_path: str):
    """Figure 6: image_stain_variation.png"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for cls in config.CLASSES:
        sub = meta_df[meta_df['class_label'] == cls]
        sns.kdeplot(sub['stain_variation'], ax=ax, color=config.CLASS_COLORS[cls], label=cls.capitalize(), fill=True, alpha=0.2)
        
    ax.set_title('Simulated H&E Stain Variation Distribution by Class', weight='bold')
    ax.set_xlabel('Stain Variation Factor (Hematoxylin/Eosin Gain Ratio)')
    ax.set_ylabel('Density')
    ax.legend(frameon=True)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_temporal_biomarker_distributions(bio_df: pd.DataFrame, out_path: str):
    """Figure 7: temporal_biomarker_distributions.png"""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    
    markers = config.BIOMARKERS
    for idx, m in enumerate(markers):
        ax = axes[idx]
        vals = bio_df[m].dropna()
        sns.histplot(vals, kde=True, ax=ax, color='#1f77b4', bins=30, stat='density')
        ax.set_title(config.BIOMARKER_LABELS[m], weight='bold')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        
    # Velocity distribution on the 6th subplot
    ax_vel = axes[5]
    vel = bio_df['ctDNA_velocity_30d'].dropna()
    sns.histplot(vel, kde=True, ax=ax_vel, color='#9467bd', bins=30, stat='density')
    ax_vel.set_title('ctDNA 30-Day Velocity (%/month)', weight='bold')
    ax_vel.set_xlabel('Rate of Change')
    ax_vel.set_ylabel('Density')
    
    plt.suptitle('Longitudinal Biomarker Value Distributions (N=16,012 Observations)', weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_temporal_missingness(bio_df: pd.DataFrame, out_path: str):
    """Figure 8: temporal_missingness.png"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # 1. Overall missingness per biomarker
    miss_pcts = [round(bio_df[m].isna().mean() * 100, 2) for m in config.BIOMARKERS]
    labels = [config.BIOMARKER_LABELS[m] for m in config.BIOMARKERS]
    
    bars = ax1.bar(labels, miss_pcts, color='#1f77b4', width=0.55, edgecolor='black')
    ax1.set_title('Controlled Biomarker Missingness (Target: 3% - 8%)', weight='bold')
    ax1.set_ylabel('Missing Percentage (%)')
    ax1.set_ylim(0, 10)
    ax1.axhline(3.0, color='red', linestyle='--', alpha=0.7, label='Lower Target (3%)')
    ax1.axhline(8.0, color='red', linestyle='--', alpha=0.7, label='Upper Target (8%)')
    ax1.legend(loc='upper right')
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, rotation=25, ha='right')
    
    for b in bars:
        h = b.get_height()
        ax1.annotate(f"{h:.2f}%", (b.get_x() + b.get_width() / 2., h + 0.2), ha='center', va='bottom', fontsize=9, weight='bold')
        
    # 2. Missingness by window type
    w_df = bio_df.groupby('window_type')[config.BIOMARKERS].apply(lambda df: df.isna().mean() * 100).T
    w_df.index = labels
    w_df.plot(kind='bar', ax=ax2, width=0.6, color=['#2ca02c', '#ff7f0e'])
    ax2.set_title('Missingness by Forecasting Window', weight='bold')
    ax2.set_ylabel('Missing Percentage (%)')
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=25, ha='right')
    ax2.legend(['Future Prediction (>90d)', 'Historical Input (<=90d)'])
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_temporal_observations_over_time(bio_df: pd.DataFrame, out_path: str):
    """Figure 9: temporal_observations_over_time.png"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # 1. Observation count over days from baseline
    sns.histplot(bio_df['days_from_baseline'], bins=35, kde=True, ax=ax1, color='#2b83ba')
    ax1.axvline(90, color='red', linestyle='--', linewidth=2, label='Day 90 Forecasting Boundary')
    ax1.set_title('Assessment Timelines: Days from Baseline', weight='bold')
    ax1.set_xlabel('Days from Baseline Assessment')
    ax1.set_ylabel('Number of Observations')
    ax1.legend()
    
    # 2. Delta days (interval between visits)
    delta_clean = bio_df[bio_df['delta_days'] > 0]['delta_days']
    sns.countplot(x=delta_clean, ax=ax2, color='#41b6c4')
    ax2.set_title('Visit Interval Distribution (Δ Days)', weight='bold')
    ax2.set_xlabel('Days Since Previous Assessment')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_temporal_trajectory_examples(bio_df: pd.DataFrame, out_path: str):
    """Figure 10: temporal_trajectory_examples.png"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    
    patterns = config.TRAJECTORY_PATTERNS
    for idx, pat in enumerate(patterns):
        ax = axes[idx]
        sub = bio_df[bio_df['trajectory_pattern'] == pat]
        # Pick 4 distinct sample patients
        sample_pats = list(sub['patient_id'].unique()[:4])
        for p_id in sample_pats:
            p_df = sub[sub['patient_id'] == p_id].sort_values('days_from_baseline')
            ax.plot(p_df['days_from_baseline'], p_df['ctDNA_vaf_percent'], marker='o', markersize=4, label=p_id, alpha=0.8)
            
        ax.axvline(90, color='grey', linestyle=':', label='Day 90 Boundary')
        ax.set_title(f"Archetype: {pat.replace('_', ' ').capitalize()}", weight='bold')
        ax.set_xlabel('Days from Baseline')
        ax.set_ylabel('ctDNA VAF (%)')
        ax.legend(fontsize=8, loc='upper left')
        
    # Blank 6th panel used for summary legend/context
    ax_last = axes[5]
    ax_last.axis('off')
    ax_last.text(0.1, 0.5, 
                 "Longitudinal ctDNA Archetypes\n\n"
                 "• Stable: Flat baseline fluctuations\n"
                 "• Gradual Increase: Linear indolent progression\n"
                 "• Gradual Decrease: Treatment response decay\n"
                 "• Fluctuating: Cyclical treatment responses\n"
                 "• Rapid Increase: Aggressive recurrence/progression\n\n"
                 "Note: Synthetic trajectories designed for\nDL model benchmarking; not real patient biology.",
                 fontsize=10, linespacing=1.6, bbox=dict(boxstyle='round,pad=0.8', facecolor='#f0f0f0', edgecolor='#cccccc'))
    
    plt.suptitle('Representative Longitudinal ctDNA Trajectories Across Synthetic Archetypes', weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_temporal_trajectory_distribution(bio_df: pd.DataFrame, out_path: str):
    """Figure 11: temporal_trajectory_distribution.png"""
    fig, ax = plt.subplots(figsize=(8, 5))
    pat_traj = bio_df.groupby('patient_id')['trajectory_pattern'].first()
    counts = pat_traj.value_counts().reindex(config.TRAJECTORY_PATTERNS)
    
    colors = [config.TRAJECTORY_COLORS.get(p, '#333333') for p in counts.index]
    bars = ax.bar([p.replace('_', '\n').capitalize() for p in counts.index], counts.values, color=colors, width=0.55, edgecolor='black')
    
    ax.set_title('Patient Cohort Trajectory Archetype Distribution (N=1,000 Patients)', weight='bold')
    ax.set_xlabel('Trajectory Archetype')
    ax.set_ylabel('Patient Count')
    ax.set_ylim(0, 250)
    
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:,} ({h/10:.1f}%)", (b.get_x() + b.get_width() / 2., h + 5), ha='center', va='bottom', fontsize=9, weight='bold')
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_forecasting_window_distribution(bio_df: pd.DataFrame, out_path: str):
    """Figure 12: forecasting_window_distribution.png"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # 1. Observations by window
    w_counts = bio_df['window_type'].value_counts()
    ax1.pie(w_counts.values, labels=['Future Prediction (>90d)', 'Historical Input (<=90d)'],
            autopct='%1.1f%%', colors=['#ff7f0e', '#2ca02c'], startangle=140, explode=(0.04, 0))
    ax1.set_title('Observation Allocation across Forecasting Horizon', weight='bold')
    
    # 2. Target availability
    t_avail = bio_df['future_ctDNA_30d_target'].notna().value_counts()
    labels = ['30d Target Available', 'Terminal Horizon (No Target)']
    ax2.bar(labels, t_avail.values, color=['#31a354', '#bdbdbd'], width=0.5, edgecolor='black')
    ax2.set_title('30-Day Forward ctDNA Forecasting Target Availability', weight='bold')
    ax2.set_ylabel('Observations')
    
    for idx, v in enumerate(t_avail.values):
        ax2.annotate(f"{v:,} ({v/len(bio_df)*100:.1f}%)", (idx, v / 2), ha='center', va='center', color='white', weight='bold', fontsize=10)
        
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

def plot_split_comparison(split_stats_df: pd.DataFrame, out_path: str):
    """Figure 13: split_comparison.png"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    # Cohort breakdown
    x = np.arange(len(split_stats_df))
    w = 0.25
    ax1.bar(x - w, split_stats_df['patient_count'], width=w, label='Patients', color='#1f77b4')
    ax1.bar(x, split_stats_df['tile_count'] / 10, width=w, label='Tiles (x10)', color='#ff7f0e')
    ax1.bar(x + w, split_stats_df['temporal_observation_count'] / 10, width=w, label='Temporal Obs (x10)', color='#2ca02c')
    
    ax1.set_title('Volume Comparison across Partitions', weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([s.capitalize() for s in split_stats_df['split']])
    ax1.set_ylabel('Scaled Volume')
    ax1.legend()
    
    # Class proportion parity across splits
    cls_df = split_stats_df[['split', 'benign_pct', 'malignant_pct', 'inflammation_pct']].set_index('split')
    cls_df.columns = ['Benign', 'Malignant', 'Inflammation']
    cls_df.plot(kind='bar', ax=ax2, color=[config.CLASS_COLORS['benign'], config.CLASS_COLORS['malignant'], config.CLASS_COLORS['inflammation']], width=0.6)
    
    ax2.set_title('Pathology Class Proportion Parity (%) across Splits', weight='bold')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_ylim(0, 50)
    ax2.set_xticklabels([s.capitalize() for s in cls_df.index], rotation=0)
    ax2.legend(title='Class')
    
    for p in ax2.patches:
        h = p.get_height()
        if h > 0:
            ax2.annotate(f"{h:.1f}%", (p.get_x() + p.get_width() / 2., h + 1), ha='center', va='bottom', fontsize=8)
            
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
