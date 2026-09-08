"""
Fast parallel generator for 12,000 Stage 2 histopathology tiles.
Generates directly into data/v2/processed/pathology_tiles/{label}/{file_name}.
"""
import os
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
from PIL import Image

# Setup paths
SRC_DIR = Path(__file__).resolve().parent
DATA_ENG_DIR = SRC_DIR.parent
PROCESSED_TILES_DIR = DATA_ENG_DIR / 'data' / 'v2' / 'processed' / 'pathology_tiles'
MANIFEST_PATH = DATA_ENG_DIR / 'data' / 'v2' / 'raw' / 'images' / 'raw_tiles_manifest.csv'

from data_generation import _generate_histopathology_tile_v2


def generate_single_tile(task):
    dest_path_str, label, seed = task
    dest_path = Path(dest_path_str)
    if dest_path.exists():
        return 0
    img_uint8, _ = _generate_histopathology_tile_v2(label=label, tile_seed=seed)
    Image.fromarray(img_uint8).save(dest_path, format="PNG", compress_level=1)
    return 1


def main():
    print(f"Reading manifest from {MANIFEST_PATH}...")
    manifest = pd.read_csv(MANIFEST_PATH)
    print(f"Total tiles in manifest: {len(manifest)}")

    # Ensure class directories exist
    for cls_name in ['benign', 'malignant', 'inflammation']:
        (PROCESSED_TILES_DIR / cls_name).mkdir(parents=True, exist_ok=True)

    tasks = []
    for _, row in manifest.iterrows():
        label = row['label']
        file_name = row['file_name']
        seed = int(row['generation_seed'])
        dest_path = str(PROCESSED_TILES_DIR / label / file_name)
        tasks.append((dest_path, label, seed))

    print(f"Prepared {len(tasks)} tile tasks. Generating across CPU workers...")
    t0 = time.time()
    
    # ProcessPoolExecutor on Windows
    generated_count = 0
    with ProcessPoolExecutor(max_workers=min(12, os.cpu_count() or 4)) as executor:
        for i, count in enumerate(executor.map(generate_single_tile, tasks, chunksize=50), start=1):
            generated_count += count
            if i % 2000 == 0 or i == len(tasks):
                elapsed = time.time() - t0
                print(f"  ... checked/generated {i}/{len(tasks)} tiles ({elapsed:.1f}s, {generated_count} new)")

    total_time = time.time() - t0
    print(f"Tile generation finished in {total_time:.1f}s. Total newly generated: {generated_count}")


if __name__ == '__main__':
    main()
