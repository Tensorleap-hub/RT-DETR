"""
Generate a minimal YOLOv5-format mock dataset for integration testing.

Layout produced:
  <root>/images/{train,val}/img_XXXX.jpg
  <root>/labels/{train,val}/img_XXXX.txt

Label format (one row per object):  class_id  cx  cy  w  h  (all normalised 0-1)
"""

import argparse
import random
from pathlib import Path

import numpy as np
from PIL import Image

NUM_CLASSES = 10
TRAIN_COUNT = 10
VAL_COUNT = 5
MAX_OBJECTS = 4
IMG_W, IMG_H = 640, 480
SEED = 42


def random_box():
    cx = random.uniform(0.1, 0.9)
    cy = random.uniform(0.1, 0.9)
    w = random.uniform(0.05, 0.3)
    h = random.uniform(0.05, 0.3)
    cx = min(max(cx, w / 2), 1 - w / 2)
    cy = min(max(cy, h / 2), 1 - h / 2)
    return cx, cy, w, h


def write_split(root: Path, split: str, count: int):
    img_dir = root / "images" / split
    lbl_dir = root / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for i in range(count):
        name = f"img_{i:04d}"
        noise = np.random.randint(0, 256, (IMG_H, IMG_W, 3), dtype=np.uint8)
        Image.fromarray(noise).save(img_dir / f"{name}.jpg")

        n_objs = random.randint(1, MAX_OBJECTS)
        lines = []
        for _ in range(n_objs):
            cls = random.randint(0, NUM_CLASSES - 1)
            cx, cy, w, h = random_box()
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
        (lbl_dir / f"{name}.txt").write_text("\n".join(lines) + "\n")

    print(f"  {split}: {count} images → {img_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="~/tensorleap/data/visdrone128",
        help="Dataset root directory",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    random.seed(SEED)
    np.random.seed(SEED)

    print(f"Creating mock dataset at: {root}")
    write_split(root, "train", TRAIN_COUNT)
    write_split(root, "val", VAL_COUNT)
    print("Done.")


if __name__ == "__main__":
    main()
