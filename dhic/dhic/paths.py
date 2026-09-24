"""Dataset discovery used by the evaluation scripts."""
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
EXPECTED_IMAGES = {"clic": 41, "kodak": 24, "tecnick": 100}


def image_files(root):
    return sorted(
        path for path in Path(root).resolve().rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
