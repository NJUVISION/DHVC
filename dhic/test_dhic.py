"""Compatibility wrapper for the historical DHIC evaluation script."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dhic_codec.eval import main


if __name__ == "__main__":
    main()
