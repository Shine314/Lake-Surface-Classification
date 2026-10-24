"""Project configuration."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = Path(os.environ.get("LAKE_DATA_DIR", PROJECT_ROOT / "data")).expanduser()
OUTPUT_DIR = PROJECT_ROOT / "results"

FEATURES = [
    "Blue",
    "Green",
    "Red",
    "NIR",
    "B5(705nm)",
    "B6(740nm)",
    "B7(783nm)",
    "B8A(865nm)",
    "B11(SWIR)",
    "B12(2190)",
    "sf1_ave",
    "WAVI",
    "NDAVI",
    "NDVI",
    "ave_123",
    "MNDWI",
    "NDMI",
    "NDCI",
]

TARGET_COLUMN = "label"
CLASS_LABELS = {"hishi": 1, "water": 2}
CLASS_NAMES = ["Hishi", "Water"]
RANDOM_STATE = 1234
TEST_SIZE = 0.3
