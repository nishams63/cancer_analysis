"""
Image Processing and Preparation for Stage 2 Histopathology Tiles (Version 2).

DISCLAIMER:
Synthetic data created for deep-learning research and pipeline prototyping.
These data are not real patient records and are not clinically validated.
Synthetic data != Real patient evidence != Clinical validation.
"""

import shutil
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd
from PIL import Image

try:
    from .config import (
        CHANNELS,
        CLINICAL_DISCLAIMER,
        IMAGE_METADATA_PATH,
        PATHOLOGY_CLASSES,
        PROCESSED_TILES_DIR,
        RAW_IMAGES_DIR,
        TILE_SIZE,
        TOTAL_TILES,
    )
    from .create_splits import get_patient_split_map
except ImportError:
    from config import (
        CHANNELS,
        CLINICAL_DISCLAIMER,
        IMAGE_METADATA_PATH,
        PATHOLOGY_CLASSES,
        PROCESSED_TILES_DIR,
        RAW_IMAGES_DIR,
        TILE_SIZE,
        TOTAL_TILES,
    )
    from create_splits import get_patient_split_map


def prepare_pathology_tiles_v2() -> pd.DataFrame:
    """
    Standardizes, validates, and organizes 12,000 pathology tiles into
    class-specific folders and compiles the master v2 image metadata manifest.
    """
    raw_manifest_path = RAW_IMAGES_DIR / "raw_tiles_manifest.csv"
    if not raw_manifest_path.exists():
        raise FileNotFoundError(f"Raw tiles manifest not found at {raw_manifest_path}. Run data_generation first.")

    raw_manifest = pd.read_csv(raw_manifest_path)
    patient_split_map = get_patient_split_map()

    # Ensure class subdirectories exist
    for cls_name in PATHOLOGY_CLASSES:
        (PROCESSED_TILES_DIR / cls_name).mkdir(parents=True, exist_ok=True)

    processed_records = []
    print(f"Preparing, validating, and organizing {len(raw_manifest)} pathology tiles...")

    for i, row in raw_manifest.iterrows():
        tile_id = row["tile_id"]
        patient_id = row["patient_id"]
        slide_id = row["slide_id"]
        label = row["label"]
        raw_path = Path(row["raw_image_path"])

        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw image: {raw_path}")

        # Quality Control: Verify readability, dimensions, and bands
        with Image.open(raw_path) as img:
            w, h = img.size
            bands = len(img.getbands())

        if (h, w) != TILE_SIZE or bands != CHANNELS:
            raise ValueError(f"Dimension/channel mismatch for {tile_id}: {(h, w, bands)} vs {(*TILE_SIZE, CHANNELS)}")

        dest_filename = f"{tile_id}_{label}.png"
        dest_path = PROCESSED_TILES_DIR / label / dest_filename
        relative_path = f"data/v2/processed/pathology_tiles/{label}/{dest_filename}"

        # Copy to class folder
        shutil.copy2(raw_path, dest_path)

        # Split mapping
        split = patient_split_map.get(patient_id, "unknown")
        if split == "unknown":
            raise ValueError(f"Patient {patient_id} missing from split map!")

        processed_records.append({
            "patient_id": patient_id,
            "tile_id": tile_id,
            "slide_id": slide_id,
            "class_label": label,
            "split": split,
            "image_path": str(dest_path.resolve()),
            "relative_path": relative_path,
            "file_name": dest_filename,
            "height": h,
            "width": w,
            "channels": CHANNELS,
            "background_temperature": row["background_temperature"],
            "brightness_factor": row["brightness_factor"],
            "contrast_factor": row["contrast_factor"],
            "stain_variation": row["stain_variation"],
            "augmentation_status": "original_unaugmented",
            "qc_status": "PASS",
            "data_source": "synthetic",
            "disclaimer": CLINICAL_DISCLAIMER,
        })

        if (i + 1) % 3000 == 0:
            print(f"  ... processed {i + 1} / {len(raw_manifest)} tiles")

    metadata_df = pd.DataFrame(processed_records)
    metadata_df.to_csv(IMAGE_METADATA_PATH, index=False)
    print(f"Image preparation complete. Saved metadata for {len(metadata_df)} tiles to {IMAGE_METADATA_PATH}")

    # Cross-tabulation audit
    ctab = pd.crosstab(metadata_df["split"], metadata_df["class_label"])
    print("\nTile distribution across splits and classes:")
    print(ctab)

    return metadata_df


if __name__ == "__main__":
    prepare_pathology_tiles_v2()
