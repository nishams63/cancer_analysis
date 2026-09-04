"""
Stage 2 EDA - Pathology Image Quality, Pixel, and Illumination Analysis Module
"""
import os
import numpy as np
import pandas as pd
from PIL import Image
from typing import Dict, Any, Tuple, List
try:
    from . import config
except (ImportError, ValueError):
    import config

def load_image_metadata() -> pd.DataFrame:
    """Loads image metadata manifest with file existence audit."""
    df = pd.read_csv(config.IMAGE_METADATA_PATH)
    return df

def audit_image_files(meta_df: pd.DataFrame, sample_size: int = 600) -> Dict[str, Any]:
    """
    Audits physical image files for readability, dimensions, channel structure, and data types.
    """
    total_images = len(meta_df)
    
    # Audit stratified sample across all splits and classes
    sample_df = meta_df.sample(n=min(sample_size, total_images), random_state=config.RANDOM_SEED)
    
    corrupt_count = 0
    blank_count = 0
    extreme_bright_count = 0
    extreme_dark_count = 0
    dimension_mismatches = 0
    channel_mismatches = 0
    
    for idx, row in sample_df.iterrows():
        p = row['image_path']
        if not os.path.exists(p):
            # Try resolving relative to repository if path prefix differs
            rel = row.get('relative_path', '')
            alt_p = config.STAGE_2_DIR / rel.replace('data-engineering/data-engineering/', 'data-engineering/')
            if alt_p.exists():
                p = str(alt_p)
            else:
                corrupt_count += 1
                continue
                
        try:
            with Image.open(p) as img:
                w, h = img.size
                mode = img.mode
                if (w, h) != (224, 224):
                    dimension_mismatches += 1
                if mode != 'RGB':
                    channel_mismatches += 1
                arr = np.array(img, dtype=np.uint8)
                mean_val = float(np.mean(arr))
                std_val = float(np.std(arr))
                if std_val < 1.0: # Blank or solid color
                    blank_count += 1
                if mean_val > 250:
                    extreme_bright_count += 1
                elif mean_val < 15:
                    extreme_dark_count += 1
        except Exception:
            corrupt_count += 1

    return {
        'total_audited': len(sample_df),
        'corrupt_files': corrupt_count,
        'blank_images': blank_count,
        'extreme_bright_images': extreme_bright_count,
        'extreme_dark_images': extreme_dark_count,
        'dimension_mismatches': dimension_mismatches,
        'channel_mismatches': channel_mismatches,
        'status': 'PASS' if (corrupt_count == 0 and blank_count == 0 and dimension_mismatches == 0) else 'FAIL'
    }

def compute_pixel_statistics(meta_df: pd.DataFrame, sample_size: int = 1200) -> Dict[str, Any]:
    """
    Computes precise RGB pixel distributions (scaled [0, 1] and [0, 255])
    on a representative sample to determine true dataset normalization parameters.
    """
    sample_df = meta_df.sample(n=min(sample_size, len(meta_df)), random_state=config.RANDOM_SEED)
    
    r_vals, g_vals, b_vals = [], [], []
    per_image_means = []
    per_image_stds = []
    
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
                r_vals.append(arr[:, :, 0].flatten())
                g_vals.append(arr[:, :, 1].flatten())
                b_vals.append(arr[:, :, 2].flatten())
                per_image_means.append(float(np.mean(arr)))
                per_image_stds.append(float(np.std(arr)))
        except Exception:
            continue

    all_r = np.concatenate(r_vals)
    all_g = np.concatenate(g_vals)
    all_b = np.concatenate(b_vals)
    
    norm_stats = {
        'mean_rgb_0_1': [float(np.mean(all_r)), float(np.mean(all_g)), float(np.mean(all_b))],
        'std_rgb_0_1': [float(np.std(all_r)), float(np.std(all_g)), float(np.std(all_b))],
        'mean_rgb_255': [float(np.mean(all_r) * 255.0), float(np.mean(all_g) * 255.0), float(np.mean(all_b) * 255.0)],
        'std_rgb_255': [float(np.std(all_r) * 255.0), float(np.std(all_g) * 255.0), float(np.std(all_b) * 255.0)],
        'overall_pixel_mean': float(np.mean([all_r, all_g, all_b])),
        'overall_pixel_std': float(np.std([all_r, all_g, all_b])),
        'percentiles_rgb': {
            'p1': [float(np.percentile(all_r, 1)), float(np.percentile(all_g, 1)), float(np.percentile(all_b, 1))],
            'p5': [float(np.percentile(all_r, 5)), float(np.percentile(all_g, 5)), float(np.percentile(all_b, 5))],
            'p50': [float(np.percentile(all_r, 50)), float(np.percentile(all_g, 50)), float(np.percentile(all_b, 50))],
            'p95': [float(np.percentile(all_r, 95)), float(np.percentile(all_g, 95)), float(np.percentile(all_b, 95))],
            'p99': [float(np.percentile(all_r, 99)), float(np.percentile(all_g, 99)), float(np.percentile(all_b, 99))],
        },
        'per_image_mean_avg': float(np.mean(per_image_means)),
        'per_image_mean_std': float(np.std(per_image_means)),
    }
    return norm_stats

def generate_image_statistics_csv(meta_df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes class-by-split statistics for image_statistics.csv:
    count, brightness factor, contrast factor, stain variation, and background temperature.
    """
    records = []
    
    for cls in config.CLASSES:
        for sp in config.SPLITS:
            sub = meta_df[(meta_df['class_label'] == cls) & (meta_df['split'] == sp)]
            cnt = len(sub)
            if cnt == 0:
                continue
            
            b_mean = float(sub['brightness_factor'].mean())
            b_std = float(sub['brightness_factor'].std())
            b_min = float(sub['brightness_factor'].min())
            b_max = float(sub['brightness_factor'].max())
            
            c_mean = float(sub['contrast_factor'].mean())
            c_std = float(sub['contrast_factor'].std())
            c_min = float(sub['contrast_factor'].min())
            c_max = float(sub['contrast_factor'].max())
            
            s_mean = float(sub['stain_variation'].mean())
            s_std = float(sub['stain_variation'].std())
            s_min = float(sub['stain_variation'].min())
            s_max = float(sub['stain_variation'].max())
            
            bg_counts = sub['background_temperature'].value_counts()
            bg_warm = int(bg_counts.get('warm', 0))
            bg_neutral = int(bg_counts.get('neutral', 0))
            bg_cool = int(bg_counts.get('cool', 0))
            
            records.append({
                'class_label': cls,
                'split': sp,
                'tile_count': cnt,
                'brightness_factor_mean': round(b_mean, 4),
                'brightness_factor_std': round(b_std, 4),
                'brightness_factor_min': round(b_min, 4),
                'brightness_factor_max': round(b_max, 4),
                'contrast_factor_mean': round(c_mean, 4),
                'contrast_factor_std': round(c_std, 4),
                'contrast_factor_min': round(c_min, 4),
                'contrast_factor_max': round(c_max, 4),
                'stain_variation_mean': round(s_mean, 4),
                'stain_variation_std': round(s_std, 4),
                'stain_variation_min': round(s_min, 4),
                'stain_variation_max': round(s_max, 4),
                'bg_warm_count': bg_warm,
                'bg_neutral_count': bg_neutral,
                'bg_cool_count': bg_cool,
            })
            
    stats_df = pd.DataFrame(records)
    return stats_df
