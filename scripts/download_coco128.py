import argparse
import tempfile
import urllib.request
import zipfile
from pathlib import Path


COCO80_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_integration_config(root: Path) -> dict:
    config = {}
    with open(root / "leap_config.yaml", "r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#") or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            elif value.startswith("'") and value.endswith("'"):
                value = value[1:-1]
            elif value.lower() in {"true", "false"}:
                value = value.lower() == "true"
            else:
                if value.isdigit():
                    value = int(value)
            config[key] = value
    return config


def write_dataset_yaml(dataset_yaml_path: Path, dataset_root: Path) -> None:
    dataset_yaml_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        f'path: "{str(dataset_root.resolve())}"',
        'train: "images/train2017"',
        'val: "images/train2017"',
        'test: "images/train2017"',
        "pred_names:",
    ]
    lines.extend([f'  - "{name}"' for name in COCO80_NAMES])
    with open(dataset_yaml_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def download_and_extract(zip_url: str, extract_dir: Path, dry_run: bool) -> Path:
    dataset_root = extract_dir / "coco128"
    if dry_run:
        return dataset_root

    extract_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_zip_path = Path(tmp.name)

    urllib.request.urlretrieve(zip_url, tmp_zip_path)
    with zipfile.ZipFile(tmp_zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)
    tmp_zip_path.unlink(missing_ok=True)
    return dataset_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Download coco128 and create dataset yaml at configured path.")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without downloading files.")
    args = parser.parse_args()

    root = project_root()
    config = load_integration_config(root)
    zip_url = config["dataset_download_url"]
    extract_dir = root / config["dataset_extract_dir"]
    dataset_yaml_path = root / config["data_yaml_path"]

    dataset_root = download_and_extract(zip_url, extract_dir, args.dry_run)
    write_dataset_yaml(dataset_yaml_path, dataset_root)

    print(f"dataset_download_url: {zip_url}")
    print(f"dataset_extract_dir: {extract_dir}")
    print(f"dataset_root: {dataset_root}")
    print(f"data_yaml_path written: {dataset_yaml_path}")
    if args.dry_run:
        print("dry-run: no dataset files were downloaded")


if __name__ == "__main__":
    main()
