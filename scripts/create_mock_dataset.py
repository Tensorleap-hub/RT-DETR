import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

NUM_CLASSES = 3
SPLIT_COUNTS = {"train": 400_000, "val": 50_000, "test": 50_000}
POOL_SIZE = 50
MAX_OBJECTS = 4
IMG_W, IMG_H = 1920, 1200
SEED = 42


def random_bbox(img_w: int, img_h: int):
    x = random.uniform(0, img_w * 0.8)
    y = random.uniform(0, img_h * 0.8)
    w = random.uniform(20, img_w * 0.2)
    h = random.uniform(20, img_h * 0.2)
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    return [round(x, 2), round(y, 2), round(w, 2), round(h, 2)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="~/tensorleap/data/rheinmetall-mock")
    parser.add_argument("--pool-size", type=int, default=POOL_SIZE)
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    random.seed(SEED)
    np.random.seed(SEED)

    categories = [{"id": i + 1, "name": f"class_{i}"} for i in range(NUM_CLASSES)]

    for split, count in SPLIT_COUNTS.items():
        img_dir = root / split / "images"
        img_dir.mkdir(parents=True, exist_ok=True)

        pool = []
        for p in range(args.pool_size):
            fname = f"pool_{p:04d}.jpg"
            noise = np.random.randint(0, 256, (IMG_H, IMG_W, 3), dtype=np.uint8)
            Image.fromarray(noise).save(img_dir / fname)
            pool.append(fname)

        images = []
        annotations = []
        ann_id = 1

        for i in range(count):
            img_id = i + 1
            fname = pool[i % len(pool)]
            images.append({"id": img_id, "file_name": fname, "width": IMG_W, "height": IMG_H})

            for _ in range(random.randint(1, MAX_OBJECTS)):
                bbox = random_bbox(IMG_W, IMG_H)
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": random.randint(1, NUM_CLASSES),
                    "bbox": bbox,
                    "area": round(bbox[2] * bbox[3], 2),
                    "iscrowd": 0,
                })
                ann_id += 1

        coco = {"info": {}, "images": images, "annotations": annotations, "categories": categories}
        ann_path = root / split / f"annotations_{split}.json"
        with open(ann_path, "w") as f:
            json.dump(coco, f)
        print(f"  {split}: {count} samples ({len(pool)} physical images), {len(annotations)} annotations -> {ann_path}")

    total = sum(SPLIT_COUNTS.values())
    print(f"\nMock dataset created at: {root} ({total} samples overall)")
    print(f"Add to leap_config.yaml dataset_path:\n  - \"{root}\"")


if __name__ == "__main__":
    main()
