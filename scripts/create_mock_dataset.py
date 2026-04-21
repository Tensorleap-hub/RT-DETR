import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

NUM_CLASSES = 3
SPLIT_COUNTS = {"train": 10, "val": 5, "test": 3}
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
    args = parser.parse_args()

    root = Path(args.root).expanduser()
    random.seed(SEED)
    np.random.seed(SEED)

    categories = [{"id": i + 1, "name": f"class_{i}"} for i in range(NUM_CLASSES)]

    for split, count in SPLIT_COUNTS.items():
        img_dir = root / "images" / split
        img_dir.mkdir(parents=True, exist_ok=True)

        images = []
        annotations = []
        ann_id = 1

        for i in range(count):
            img_id = i + 1
            fname = f"images/{split}/img_{i:04d}.jpg"
            noise = np.random.randint(0, 256, (IMG_H, IMG_W, 3), dtype=np.uint8)
            Image.fromarray(noise).save(root / fname)
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
        ann_path = root / f"annotations_{split}.json"
        ann_path.write_text(json.dumps(coco, indent=2))
        print(f"  {split}: {count} images, {len(annotations)} annotations → {ann_path}")

    print(f"\nMock dataset created at: {root}")
    print(f"Add to leap_config.yaml dataset_path:\n  - \"{root}\"")


if __name__ == "__main__":
    main()
