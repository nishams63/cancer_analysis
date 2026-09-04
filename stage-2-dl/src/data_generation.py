"""
Deterministic Synthetic Data Generator for Stage 2 Deep Learning v2.

Features:
  - 12,000 Pathology Tiles: 1,000 patients * 12 tiles (4 benign, 4 malignant, 4 inflammation).
  - Anti-shortcut Image Generation: Background stroma, temperature, exposure, and stain gains
    are randomized independently across ALL classes to eliminate non-morphological shortcuts.
  - Longitudinal Temporal Biomarkers: 14–18 timepoints per patient (~16,000 total observations)
    spanning at least 180 days across 5 trajectory patterns.
  - Forecasting Windows: Explicit separation between Historical Input (<= 90d) and Future Prediction (> 90d),
    with forward 30-day target and progression trend labels.
  - Controlled Missingness: 3–8% with explicit binary indicators.

DISCLAIMER:
Synthetic data created for deep-learning research and pipeline prototyping.
These data are not real patient records and are not clinically validated.
Synthetic data != Real patient evidence != Clinical validation.
"""

import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from PIL import Image

try:
    from .config import (
        CHANNELS,
        CLINICAL_DISCLAIMER,
        FORECAST_SPLIT_DAY,
        INTERVAL_DAYS_BASE,
        MAX_TIMEPOINTS,
        MIN_TIMEPOINTS,
        MISSINGNESS_MAX_RATE,
        MISSINGNESS_MIN_RATE,
        NUM_PATIENTS,
        PATHOLOGY_CLASSES,
        RANDOM_SEED,
        RAW_BIOMARKERS_DIR,
        RAW_IMAGES_DIR,
        SLIDES_PER_PATIENT,
        TARGET_FORECAST_DAYS,
        TILE_SIZE,
        TILES_PER_PATIENT,
        TOTAL_TILES,
        TRAJECTORY_ARCHETYPES,
    )
    from .create_splits import generate_patient_cohort
except ImportError:
    from config import (
        CHANNELS,
        CLINICAL_DISCLAIMER,
        FORECAST_SPLIT_DAY,
        INTERVAL_DAYS_BASE,
        MAX_TIMEPOINTS,
        MIN_TIMEPOINTS,
        MISSINGNESS_MAX_RATE,
        MISSINGNESS_MIN_RATE,
        NUM_PATIENTS,
        PATHOLOGY_CLASSES,
        RANDOM_SEED,
        RAW_BIOMARKERS_DIR,
        RAW_IMAGES_DIR,
        SLIDES_PER_PATIENT,
        TARGET_FORECAST_DAYS,
        TILE_SIZE,
        TILES_PER_PATIENT,
        TOTAL_TILES,
        TRAJECTORY_ARCHETYPES,
    )
    from create_splits import generate_patient_cohort


# =====================================================================
# 1. VECTORIZED ANTI-SHORTCUT HISTOPATHOLOGY TILE GENERATOR
# =====================================================================

def _generate_histopathology_tile_v2(
    label: str,
    tile_seed: int,
    size: Tuple[int, int] = TILE_SIZE,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Vectorized procedural generator creating 224x224 RGB pathology-like tiles.
    Strictly decouples global background appearance from class labels:
      - Color temperature (warm, neutral, cool) randomized across all classes.
      - Exposure, contrast, and H&E gains randomized across all classes.
      - Only cellular morphology and spatial architecture distinguish classes.
    """
    rng = np.random.RandomState(tile_seed)
    h, w = size

    # Coordinate grids
    y_grid, x_grid = np.mgrid[0:h, 0:w]

    # --- ANTI-SHORTCUT RANDOMIZATIONS (Shared identically across all classes) ---
    temp_choice = rng.choice(["warm", "neutral", "cool"])
    if temp_choice == "warm":
        base_stroma = np.array([0.94, 0.84, 0.88], dtype=np.float32)
    elif temp_choice == "neutral":
        base_stroma = np.array([0.91, 0.85, 0.90], dtype=np.float32)
    else:  # cool
        base_stroma = np.array([0.88, 0.83, 0.93], dtype=np.float32)

    brightness_factor = float(rng.uniform(0.92, 1.08))
    contrast_factor = float(rng.uniform(0.90, 1.10))
    h_gain = float(rng.uniform(0.88, 1.12))
    e_gain = float(rng.uniform(0.88, 1.12))

    # Base stroma background with micro-texture
    stroma_noise = rng.normal(0.0, 0.02, (h, w, 1)).astype(np.float32)
    img = np.clip(base_stroma * brightness_factor + stroma_noise, 0.0, 1.0)

    # Core H&E Pigments
    hematoxylin = np.clip(np.array([0.34, 0.14, 0.54], dtype=np.float32) * h_gain, 0.0, 1.0)
    eosin = np.clip(np.array([0.87, 0.42, 0.64], dtype=np.float32) * e_gain, 0.0, 1.0)

    # --- CLASS-SPECIFIC CELLULAR MORPHOLOGY ---
    if label == "benign":
        # Regular circular/elliptical glandular rings with clear lumens
        num_glands = rng.randint(2, 5)
        for _ in range(num_glands):
            cx = rng.randint(35, w - 35)
            cy = rng.randint(35, h - 35)
            rx = rng.uniform(22, 40)
            ry = rng.uniform(18, 36)
            theta = rng.uniform(0, np.pi)
            cos_t, sin_t = np.cos(theta), np.sin(theta)

            # Transformed coordinates
            dx = (x_grid - cx) * cos_t + (y_grid - cy) * sin_t
            dy = -(x_grid - cx) * sin_t + (y_grid - cy) * cos_t
            dist_sq = (dx / rx) ** 2 + (dy / ry) ** 2

            # Cytoplasmic ring (0.6 <= dist <= 1.0)
            cyto_mask = (dist_sq >= 0.36) & (dist_sq <= 1.0)
            lumen_mask = dist_sq < 0.36

            img[cyto_mask] = 0.45 * img[cyto_mask] + 0.55 * eosin
            img[lumen_mask] = np.array([0.98, 0.98, 0.98], dtype=np.float32)

            # Organized perimeter nuclei
            n_nuclei = rng.randint(16, 26)
            angles = np.linspace(0, 2 * np.pi, n_nuclei, endpoint=False) + rng.normal(0, 0.08, n_nuclei)
            for ang in angles:
                nx = int(cx + 0.82 * rx * np.cos(ang) * cos_t - 0.82 * ry * np.sin(ang) * sin_t)
                ny = int(cy + 0.82 * rx * np.cos(ang) * sin_t + 0.82 * ry * np.sin(ang) * cos_t)
                r_n = rng.uniform(2.2, 3.2)
                r_int = int(math.ceil(r_n))
                x0, x1 = max(0, nx - r_int), min(w, nx + r_int + 1)
                y0, y1 = max(0, ny - r_int), min(h, ny + r_int + 1)
                sub_d = (x_grid[y0:y1, x0:x1] - nx) ** 2 + (y_grid[y0:y1, x0:x1] - ny) ** 2
                img[y0:y1, x0:x1][sub_d <= r_n ** 2] = hematoxylin

        # Low density stromal spindle cells (organized stroma)
        n_spindle = rng.randint(25, 45)
        for _ in range(n_spindle):
            sx, sy = rng.randint(4, w - 4), rng.randint(4, h - 4)
            img[max(0, sy-1):min(h, sy+2), max(0, sx-2):min(w, sx+3)] = hematoxylin * 1.1

    elif label == "malignant":
        # Disrupted architecture, invasive syncytial clusters, hypercellular pleomorphism
        num_clusters = rng.randint(3, 7)
        for _ in range(num_clusters):
            cx, cy = rng.randint(30, w - 30), rng.randint(30, h - 30)
            cr = rng.uniform(30, 55)
            c_mask = ((x_grid - cx) ** 2 + (y_grid - cy) ** 2) <= cr ** 2
            img[c_mask] = 0.40 * img[c_mask] + 0.60 * eosin

        # High cellularity: 120-180 crowded, pleomorphic, hyperchromatic nuclei (N:C ratio ~0.7)
        n_malignant = rng.randint(120, 180)
        xs = rng.randint(8, w - 8, size=n_malignant)
        ys = rng.randint(8, h - 8, size=n_malignant)
        rxs = rng.uniform(3.5, 6.2, size=n_malignant)
        rys = rng.uniform(2.4, 5.0, size=n_malignant)
        thetas = rng.uniform(0, np.pi, size=n_malignant)

        for nx, ny, rx, ry, th in zip(xs, ys, rxs, rys, thetas):
            r_max = int(math.ceil(max(rx, ry)))
            x0, x1 = max(0, nx - r_max), min(w, nx + r_max + 1)
            y0, y1 = max(0, ny - r_max), min(h, ny + r_max + 1)
            cos_t, sin_t = math.cos(th), math.sin(th)
            sub_x = x_grid[y0:y1, x0:x1] - nx
            sub_y = y_grid[y0:y1, x0:x1] - ny
            dx = sub_x * cos_t + sub_y * sin_t
            dy = -sub_x * sin_t + sub_y * cos_t
            mask = (dx / rx) ** 2 + (dy / ry) ** 2 <= 1.0
            # Hyperchromasia: darker, denser hematoxylin
            img[y0:y1, x0:x1][mask] = hematoxylin * rng.uniform(0.72, 0.88)

    elif label == "inflammation":
        # Diffuse high-density lymphocytic infiltration with reactive stromal patterns
        num_foci = rng.randint(3, 6)
        for _ in range(num_foci):
            fx, fy = rng.randint(30, w - 30), rng.randint(30, h - 30)
            fr = rng.uniform(25, 50)
            f_mask = ((x_grid - fx) ** 2 + (y_grid - fy) ** 2) <= fr ** 2
            img[f_mask] = 0.65 * img[f_mask] + 0.35 * np.array([0.96, 0.88, 0.92], dtype=np.float32)

        # Intense punctate lymphocytic infiltration: 280-420 small round dark nuclei
        n_lymph = rng.randint(280, 420)
        lxs = rng.randint(5, w - 5, size=n_lymph)
        lys = rng.randint(5, h - 5, size=n_lymph)
        lrs = rng.uniform(1.8, 2.6, size=n_lymph)

        for lx, ly, lr in zip(lxs, lys, lrs):
            r_int = int(math.ceil(lr))
            x0, x1 = max(0, lx - r_int), min(w, lx + r_int + 1)
            y0, y1 = max(0, ly - r_int), min(h, ly + r_int + 1)
            sub_d = (x_grid[y0:y1, x0:x1] - lx) ** 2 + (y_grid[y0:y1, x0:x1] - ly) ** 2
            img[y0:y1, x0:x1][sub_d <= lr ** 2] = hematoxylin * 0.80

    # Global contrast scaling
    img = np.clip((img - 0.5) * contrast_factor + 0.5, 0.0, 1.0)
    img_uint8 = (img * 255.0).astype(np.uint8)

    meta = {
        "background_temperature": temp_choice,
        "brightness_factor": round(brightness_factor, 3),
        "contrast_factor": round(contrast_factor, 3),
        "stain_variation": round(float(h_gain / e_gain), 3),
    }
    return img_uint8, meta


def generate_raw_pathology_dataset_v2(
    cohort_df: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """
    Generates exactly 12,000 pathology tiles across 1,000 patients.
    Each patient gets exactly 12 tiles: 4 benign, 4 malignant, 4 inflammation.
    Total: exactly 4,000 benign, 4,000 malignant, 4,000 inflammation (33.33% each).
    """
    RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    metadata_records = []
    print(f"Generating 12,000 raw synthetic pathology tiles for {len(cohort_df)} patients...")

    tile_counter = 0
    # Balanced classes per patient: 4 of each = 12 total
    classes_per_patient = ["benign"] * 4 + ["malignant"] * 4 + ["inflammation"] * 4

    for _, patient in cohort_df.iterrows():
        patient_id = patient["patient_id"]

        # Deterministically shuffle class order for each patient
        p_seed = seed + int(patient_id.split("-")[1]) * 13
        p_rng = np.random.RandomState(p_seed)
        shuffled_classes = p_rng.permutation(classes_per_patient)

        for tile_idx, label in enumerate(shuffled_classes, start=1):
            tile_counter += 1
            slide_idx = 1 if tile_idx <= 6 else 2
            slide_id = f"{patient_id}_SLIDE_{slide_idx:02d}"
            tile_id = f"{slide_id}_TILE_{tile_idx:02d}"

            tile_seed = seed + tile_counter * 997
            tile_img, meta_props = _generate_histopathology_tile_v2(label=label, tile_seed=tile_seed)

            file_name = f"{tile_id}_{label}.png"
            file_path = RAW_IMAGES_DIR / file_name

            # Fast PNG compression
            Image.fromarray(tile_img).save(file_path, format="PNG", compress_level=1)

            metadata_records.append({
                "tile_id": tile_id,
                "patient_id": patient_id,
                "slide_id": slide_id,
                "file_name": file_name,
                "raw_image_path": str(file_path),
                "label": label,
                "source_type": "synthetic",
                "width": TILE_SIZE[1],
                "height": TILE_SIZE[0],
                "channels": CHANNELS,
                "background_temperature": meta_props["background_temperature"],
                "brightness_factor": meta_props["brightness_factor"],
                "contrast_factor": meta_props["contrast_factor"],
                "stain_variation": meta_props["stain_variation"],
                "generation_seed": tile_seed,
                "disclaimer": CLINICAL_DISCLAIMER,
            })

        if tile_counter % 2400 == 0:
            print(f"  ... generated {tile_counter} / {TOTAL_TILES} tiles ({tile_counter/TOTAL_TILES*100:.1f}%)")

    df = pd.DataFrame(metadata_records)
    raw_meta_path = RAW_IMAGES_DIR / "raw_tiles_manifest.csv"
    df.to_csv(raw_meta_path, index=False)
    print(f"Completed {len(df)} tiles. Manifest saved to {raw_meta_path}")
    return df


# =====================================================================
# 2. LONGITUDINAL BIOMARKER & FORECASTING SEQUENCE GENERATOR
# =====================================================================

def _simulate_patient_temporal_series(
    patient_id: str,
    trajectory: str,
    patient_seed: int,
) -> List[Dict]:
    """
    Generates 14 to 18 chronological observations spanning >= 180 days.
    Splits into:
      - Historical Input Window (days_from_baseline <= 90)
      - Future Prediction Window (days_from_baseline > 90)
    Generates future_ctDNA_30d_target and future_progression_trend.
    Controlled missingness: 3–8% with indicator columns.
    """
    rng = np.random.RandomState(patient_seed)
    num_timepoints = rng.randint(MIN_TIMEPOINTS, MAX_TIMEPOINTS + 1)
    base_date = datetime(2026, 1, 1) + timedelta(days=int(patient_seed % 45))

    # Patient-specific missingness rate in [0.03, 0.08]
    pt_missing_rate = rng.uniform(MISSINGNESS_MIN_RATE, MISSINGNESS_MAX_RATE)

    # Initial baseline ranges
    if trajectory == "stable":
        base_ctdna = rng.uniform(0.18, 0.45)
        base_cea = rng.uniform(1.8, 3.2)
        base_ca125 = rng.uniform(14.0, 26.0)
        base_ldh = rng.uniform(150.0, 195.0)
        base_crp = rng.uniform(1.5, 4.0)
    elif trajectory == "gradual_increase":
        base_ctdna = rng.uniform(0.22, 0.52)
        base_cea = rng.uniform(2.5, 4.8)
        base_ca125 = rng.uniform(22.0, 34.0)
        base_ldh = rng.uniform(160.0, 205.0)
        base_crp = rng.uniform(2.0, 5.0)
    elif trajectory == "gradual_decrease":
        base_ctdna = rng.uniform(3.6, 6.2)
        base_cea = rng.uniform(26.0, 48.0)
        base_ca125 = rng.uniform(65.0, 98.0)
        base_ldh = rng.uniform(280.0, 380.0)
        base_crp = rng.uniform(8.0, 16.0)
    elif trajectory == "fluctuating":
        base_ctdna = rng.uniform(0.8, 1.9)
        base_cea = rng.uniform(5.5, 12.5)
        base_ca125 = rng.uniform(30.0, 52.0)
        base_ldh = rng.uniform(200.0, 265.0)
        base_crp = rng.uniform(4.0, 10.5)
    else:  # rapid_increase
        base_ctdna = rng.uniform(0.35, 0.75)
        base_cea = rng.uniform(3.2, 6.2)
        base_ca125 = rng.uniform(26.0, 42.0)
        base_ldh = rng.uniform(175.0, 225.0)
        base_crp = rng.uniform(3.0, 7.5)

    # Timepoints generation: days 0 through ~200+
    days_list = [0]
    current_day = 0
    for _ in range(1, num_timepoints):
        step = INTERVAL_DAYS_BASE + rng.randint(-2, 3)
        current_day += step
        days_list.append(current_day)

    # Ensure spans at least 180 days
    if days_list[-1] < 185:
        # Scale intervals slightly
        scale = 190.0 / days_list[-1]
        days_list = [int(round(d * scale)) for d in days_list]
        # Guarantee strict monotonicity
        for i in range(1, len(days_list)):
            if days_list[i] <= days_list[i-1]:
                days_list[i] = days_list[i-1] + 1

    # Simulate measurements
    raw_obs = []
    total_span = days_list[-1]

    for t_idx, d_val in enumerate(days_list):
        progress = d_val / float(total_span)
        v_date = base_date + timedelta(days=d_val)

        noise_ctdna = rng.normal(0, 0.04)
        noise_cea = rng.normal(0, 0.35)
        noise_ca125 = rng.normal(0, 1.8)
        noise_ldh = rng.normal(0, 8.5)
        noise_crp = rng.normal(0, 0.45)

        if trajectory == "stable":
            ctdna = max(0.05, base_ctdna + noise_ctdna)
            cea = max(0.5, base_cea + noise_cea)
            ca125 = max(5.0, base_ca125 + noise_ca125)
            ldh = max(120.0, base_ldh + noise_ldh)
            crp = max(0.5, base_crp + noise_crp)
        elif trajectory == "gradual_increase":
            ctdna = max(0.1, base_ctdna + progress * 3.4 + noise_ctdna)
            cea = max(1.0, base_cea + progress * 19.0 + noise_cea)
            ca125 = max(10.0, base_ca125 + progress * 48.0 + noise_ca125)
            ldh = max(130.0, base_ldh + progress * 160.0 + noise_ldh)
            crp = max(1.0, base_crp + progress * 8.5 + noise_crp)
        elif trajectory == "gradual_decrease":
            decay = math.exp(-2.4 * progress)
            ctdna = max(0.05, base_ctdna * decay + noise_ctdna * 0.4)
            cea = max(1.0, base_cea * decay + noise_cea * 0.4)
            ca125 = max(8.0, base_ca125 * decay + noise_ca125 * 0.4)
            ldh = max(140.0, base_ldh * decay + 140.0 * (1 - decay) + noise_ldh * 0.4)
            crp = max(0.8, base_crp * decay + 1.2 * (1 - decay) + noise_crp * 0.4)
        elif trajectory == "fluctuating":
            osc = math.sin(2.0 * math.pi * progress * 1.5)
            ctdna = max(0.1, base_ctdna + 1.0 * osc + noise_ctdna)
            cea = max(1.5, base_cea + 6.5 * osc + noise_cea)
            ca125 = max(15.0, base_ca125 + 22.0 * osc + noise_ca125)
            ldh = max(150.0, base_ldh + 55.0 * osc + noise_ldh)
            crp = max(1.0, base_crp + 4.2 * osc + noise_crp)
        else:  # rapid_increase
            exp_factor = math.exp(3.1 * progress) - 1.0
            ctdna = max(0.1, base_ctdna + exp_factor * 0.65 + noise_ctdna)
            cea = max(1.5, base_cea + exp_factor * 4.5 + noise_cea)
            ca125 = max(15.0, base_ca125 + exp_factor * 7.0 + noise_ca125)
            ldh = max(150.0, base_ldh + exp_factor * 28.0 + noise_ldh)
            crp = max(1.0, base_crp + exp_factor * 1.6 + noise_crp)

        # Missingness mask (3–8% rate)
        m_ctdna = (rng.rand() < pt_missing_rate)
        m_cea = (rng.rand() < pt_missing_rate)
        m_ca125 = (rng.rand() < pt_missing_rate)
        m_ldh = (rng.rand() < pt_missing_rate)
        m_crp = (rng.rand() < pt_missing_rate)

        # Forecasting window tag
        is_input = 1 if d_val <= FORECAST_SPLIT_DAY else 0
        w_type = "historical_input" if is_input == 1 else "future_prediction"

        raw_obs.append({
            "patient_id": patient_id,
            "timepoint_index": t_idx,
            "timestamp": v_date.strftime("%Y-%m-%d"),
            "days_from_baseline": d_val,
            "ctDNA_vaf_percent": None if m_ctdna else round(float(ctdna), 4),
            "cea_ng_ml": None if m_cea else round(float(cea), 2),
            "ca125_u_ml": None if m_ca125 else round(float(ca125), 2),
            "ldh_u_l": None if m_ldh else round(float(ldh), 1),
            "crp_mg_l": None if m_crp else round(float(crp), 2),
            "ctDNA_missing": int(m_ctdna),
            "cea_missing": int(m_cea),
            "ca125_missing": int(m_ca125),
            "ldh_missing": int(m_ldh),
            "crp_missing": int(m_crp),
            "window_type": w_type,
            "is_input_window": is_input,
            "trajectory_pattern": trajectory,
            "data_source": "synthetic",
            "disclaimer": CLINICAL_DISCLAIMER,
        })

    # Forward forecasting target creation:
    # 1. future_ctDNA_30d_target: find observation nearest to (day + 30)
    # 2. future_progression_trend: binary trend in future window
    input_ctdnas = [o["ctDNA_vaf_percent"] for o in raw_obs if o["is_input_window"] == 1 and o["ctDNA_vaf_percent"] is not None]
    future_ctdnas = [o["ctDNA_vaf_percent"] for o in raw_obs if o["is_input_window"] == 0 and o["ctDNA_vaf_percent"] is not None]
    
    mean_input = float(np.mean(input_ctdnas)) if input_ctdnas else 0.3
    mean_future = float(np.mean(future_ctdnas)) if future_ctdnas else mean_input
    # Progression: future mean exceeds input mean by > 0.15% VAF
    patient_progression_trend = 1 if (mean_future - mean_input) > 0.15 else 0

    for i, obs in enumerate(raw_obs):
        target_day = obs["days_from_baseline"] + TARGET_FORECAST_DAYS
        # Look for observation within [target_day - 10, target_day + 10]
        match_val = None
        min_diff = 999
        for future_cand in raw_obs[i+1:]:
            diff = abs(future_cand["days_from_baseline"] - target_day)
            if diff <= 12 and diff < min_diff:
                min_diff = diff
                match_val = future_cand["ctDNA_vaf_percent"]

        obs["future_ctDNA_30d_target"] = match_val
        obs["future_progression_trend"] = patient_progression_trend

    return raw_obs


def generate_raw_biomarker_dataset_v2(
    cohort_df: pd.DataFrame,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Generates 14–18 observations per patient across 1,000 patients (~16,000 observations)."""
    RAW_BIOMARKERS_DIR.mkdir(parents=True, exist_ok=True)
    all_records = []
    print(f"Generating raw temporal biomarker streams for {len(cohort_df)} patients...")

    for idx, patient in cohort_df.iterrows():
        patient_id = patient["patient_id"]
        trajectory = patient["biomarker_trajectory"]
        patient_seed = seed + idx * 79

        patient_records = _simulate_patient_temporal_series(
            patient_id=patient_id,
            trajectory=trajectory,
            patient_seed=patient_seed,
        )
        all_records.extend(patient_records)

    df = pd.DataFrame(all_records)
    # Strict sorting: patient_id, then days_from_baseline
    df = df.sort_values(by=["patient_id", "days_from_baseline"]).reset_index(drop=True)

    raw_path = RAW_BIOMARKERS_DIR / "raw_biomarkers.csv"
    df.to_csv(raw_path, index=False)
    print(f"Generated {len(df)} longitudinal biomarker observations. Saved to {raw_path}")
    return df


def run_data_generation_v2():
    """Generates both 12,000 raw visual tiles and ~16,000 biomarker records for v2."""
    cohort_df = generate_patient_cohort(seed=RANDOM_SEED)
    generate_raw_pathology_dataset_v2(cohort_df=cohort_df, seed=RANDOM_SEED)
    generate_raw_biomarker_dataset_v2(cohort_df=cohort_df, seed=RANDOM_SEED)
    print("Version 2 raw data generation completed successfully.")


if __name__ == "__main__":
    run_data_generation_v2()
