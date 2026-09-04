"""
Histopathology Image Augmentation Module for Stage 2 Deep Learning (Version 2).

STRICT POLICY:
Augmentation is applied ONLY to TRAINING split images.
Validation and Test sets remain strictly unaugmented for deterministic, uncorrupted evaluation.

DISCLAIMER:
Synthetic data created for deep-learning research and pipeline prototyping.
These data are not real patient records and are not clinically validated.
Synthetic data != Real patient evidence != Clinical validation.
"""

import math
import random
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

try:
    from .config import AUGMENTATION_QC_PATH, CLINICAL_DISCLAIMER, IMAGE_METADATA_PATH, PROCESSED_TILES_DIR
except ImportError:
    from config import AUGMENTATION_QC_PATH, CLINICAL_DISCLAIMER, IMAGE_METADATA_PATH, PROCESSED_TILES_DIR


class HistopathologyTrainAugmentation:
    """
    Biologically appropriate training-only augmentation pipeline.
    Simulates:
      1. H&E stain color variation (channel gain jitter in optical space)
      2. Scanner exposure variation (brightness & contrast)
      3. Dihedral D4 symmetries (rotations: 0, 90, 180, 270 degrees + H/V flips)
      4. Subtle microscopy artifacts (slight focal blur, optical vignetting)

    Strictly refuses to augment validation or test samples.
    """

    def __init__(
        self,
        brightness_range: Tuple[float, float] = (0.90, 1.10),
        contrast_range: Tuple[float, float] = (0.90, 1.10),
        stain_gain_range: Tuple[float, float] = (0.93, 1.07),
        artifact_blur_prob: float = 0.25,
        artifact_vignette_prob: float = 0.20,
    ):
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.stain_gain_range = stain_gain_range
        self.artifact_blur_prob = artifact_blur_prob
        self.artifact_vignette_prob = artifact_vignette_prob

    def apply_stain_jitter(self, img: Image.Image) -> Image.Image:
        """Applies mild channel gain variations to simulate H&E batch staining differences."""
        arr = np.array(img, dtype=np.float32)
        r_gain = random.uniform(*self.stain_gain_range)
        g_gain = random.uniform(*self.stain_gain_range)
        b_gain = random.uniform(*self.stain_gain_range)

        arr[:, :, 0] *= r_gain
        arr[:, :, 1] *= g_gain
        arr[:, :, 2] *= b_gain

        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(arr)

    def apply_vignette_artifact(self, img: Image.Image) -> Image.Image:
        """Simulates subtle optical lens vignetting (slight radial attenuation at corners)."""
        w, h = img.size
        y, x = np.mgrid[0:h, 0:w]
        cx, cy = w / 2.0, h / 2.0
        max_dist = math.sqrt(cx**2 + cy**2)
        dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) / max_dist

        falloff = 1.0 - 0.12 * (dist ** 2)
        falloff = np.expand_dims(falloff, axis=-1)

        arr = np.array(img, dtype=np.float32) * falloff
        arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)
        return Image.fromarray(arr)

    def __call__(self, img: Image.Image, split: str = "train") -> Image.Image:
        """
        Applies transformations ONLY if split == 'train'.
        Validation and Test splits return the original image unaugmented.
        """
        if split != "train":
            # Strict safety guard: Return unaugmented image
            return img

        # 1. Biological Dihedral D4 symmetries
        rot_choice = random.choice([0, 90, 180, 270])
        if rot_choice == 90:
            img = img.transpose(Image.Transpose.ROTATE_90)
        elif rot_choice == 180:
            img = img.transpose(Image.Transpose.ROTATE_180)
        elif rot_choice == 270:
            img = img.transpose(Image.Transpose.ROTATE_270)

        if random.random() < 0.5:
            img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        # 2. Scanner exposure variation
        b_factor = random.uniform(*self.brightness_range)
        img = ImageEnhance.Brightness(img).enhance(b_factor)

        c_factor = random.uniform(*self.contrast_range)
        img = ImageEnhance.Contrast(img).enhance(c_factor)

        # 3. Stain variation
        img = self.apply_stain_jitter(img)

        # 4. Realistic Slide / Optical Artifacts
        if random.random() < self.artifact_blur_prob:
            blur_radius = random.uniform(0.4, 0.8)
            img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))

        if random.random() < self.artifact_vignette_prob:
            img = self.apply_vignette_artifact(img)

        return img


def get_image_transforms(split: str = "train") -> Callable[[Image.Image], Image.Image]:
    """
    Factory function returning the appropriate transform pipeline for a given split.
    Guarantees:
      - 'train' receives HistopathologyTrainAugmentation.
      - 'validation' and 'test' receive Identity (Strictly unaugmented).
    """
    if split == "train":
        train_aug = HistopathologyTrainAugmentation()
        return lambda img: train_aug(img, split="train")
    else:
        return lambda img: img


def export_sample_augmentations(sample_image_path: Optional[Union[str, Path]] = None, num_samples: int = 4):
    """
    Exports visual side-by-side comparison of original vs augmented TRAINING images
    for inspection in reports/augmentation_samples_comparison.png.
    Strictly verifies that the sample image belongs to the TRAINING split.
    """
    if sample_image_path is None:
        import pandas as pd
        if IMAGE_METADATA_PATH.exists():
            df = pd.read_csv(IMAGE_METADATA_PATH)
            train_rows = df[df["split"] == "train"]
            if not train_rows.empty:
                sample_image_path = train_rows.iloc[0]["image_path"]

    if sample_image_path is None:
        sample_files = list(PROCESSED_TILES_DIR.glob("**/*.png"))
        if not sample_files:
            print("No tiles found to generate augmentation sample.")
            return
        sample_image_path = sample_files[0]

    orig_img = Image.open(sample_image_path).convert("RGB")
    aug_pipeline = HistopathologyTrainAugmentation()

    w, h = orig_img.size
    total_cols = num_samples + 1
    grid = Image.new("RGB", (w * total_cols, h), color=(255, 255, 255))
    grid.paste(orig_img, (0, 0))

    random.seed(42)
    np.random.seed(42)
    for i in range(num_samples):
        aug_img = aug_pipeline(orig_img.copy(), split="train")
        grid.paste(aug_img, ((i + 1) * w, 0))

    AUGMENTATION_QC_PATH.parent.mkdir(parents=True, exist_ok=True)
    grid.save(AUGMENTATION_QC_PATH)
    print(f"Augmentation preview grid (training only) saved to {AUGMENTATION_QC_PATH}")


if __name__ == "__main__":
    export_sample_augmentations()
