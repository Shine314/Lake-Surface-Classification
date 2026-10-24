"""CSV loading and feature engineering for the Lake Suwa classifier."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import CLASS_LABELS, FEATURES, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE

RGB_COLUMNS = ["Blue", "Green", "Red", "NIR"]
OTHER_BAND_COLUMNS = [
    "B5(705nm)",
    "B6(740nm)",
    "B7(783nm)",
    "B8A(865nm)",
    "B11(SWIR)",
    "B12(2190)",
]


def _read_csv(path: str | Path, usecols: list[int]) -> pd.DataFrame:
    """Read a source CSV using the encoding and column layout of the original data."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(
            f"CSV file not found: {path}\n"
            "Check --data-dir and confirm that the expected input files are present."
        )

    return pd.read_csv(
        path,
        encoding="shift_jis",
        engine="python",
        usecols=usecols,
    )


def _scale_reflectance(df: pd.DataFrame, year: int) -> pd.DataFrame:
    """Scale raw reflectance values according to the source-data year."""
    df = df.apply(pd.to_numeric, errors="coerce")
    if year >= 2022:
        return (df - 1000) / 10000
    return df / 10000


def _safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide while turning zero-denominator results into NaN."""
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def _build_features(df_rgb: pd.DataFrame, df_other: pd.DataFrame) -> pd.DataFrame:
    """Combine spectral bands and calculate the vegetation/water indices."""
    bands = pd.concat([df_rgb, df_other], axis=1)
    blue = bands["Blue"]
    green = bands["Green"]
    red = bands["Red"]
    nir = bands["NIR"]
    swir = bands["B11(SWIR)"]
    red_edge = bands["B5(705nm)"]
    visible_mean = (blue + green + red) / 3

    indices = pd.DataFrame(index=bands.index)
    indices["sf1_ave"] = _safe_divide(nir - visible_mean, nir + visible_mean)
    indices["WAVI"] = _safe_divide(1.5 * (nir - blue), 0.5 + nir + blue)
    indices["NDAVI"] = _safe_divide(nir - blue, nir + blue)
    indices["NDVI"] = _safe_divide(nir - red, nir + red)
    indices["ave_123"] = visible_mean
    indices["MNDWI"] = _safe_divide(green - swir, green + swir)
    indices["NDMI"] = _safe_divide(nir - swir, nir + swir)
    indices["NDCI"] = _safe_divide(red_edge - red, red + red_edge)

    features = pd.concat([bands, indices], axis=1)
    features = features.replace([np.inf, -np.inf], np.nan)

    return features.loc[:, FEATURES]


def get_df_for_RF(
    file_10m: str | Path,
    file_20m: str | Path,
    surface_type: str,
    year: int,
) -> pd.DataFrame:
    """Load labelled samples for training (hishi=1, water=2)."""
    surface_type = surface_type.lower().strip()
    if surface_type not in {"hishi", "water"}:
        raise ValueError("surface_type must be either 'hishi' or 'water'.")

    df_rgb = _read_csv(file_10m, usecols=[1, 2, 3, 4])
    df_rgb.columns = RGB_COLUMNS
    df_rgb = _scale_reflectance(df_rgb, year)

    df_other = _read_csv(file_20m, usecols=[1, 2, 3, 4, 5, 6])
    df_other.columns = OTHER_BAND_COLUMNS
    df_other = _scale_reflectance(df_other, year)

    features = _build_features(df_rgb, df_other)
    features[TARGET_COLUMN] = CLASS_LABELS[surface_type]
    features = features.replace([np.inf, -np.inf], np.nan).dropna()
    return features


def get_df_for_prediction(
    file_10m: str | Path,
    file_20m: str | Path,
    year: int,
) -> pd.DataFrame:
    """Load whole-area imagery and return the same features used for training."""
    df_rgb = _read_csv(file_10m, usecols=[3, 4, 5, 6])
    df_rgb.columns = RGB_COLUMNS
    df_rgb = _scale_reflectance(df_rgb, year)

    df_other = _read_csv(file_20m, usecols=[3, 4, 5, 6, 7, 8])
    df_other.columns = OTHER_BAND_COLUMNS
    df_other = _scale_reflectance(df_other, year)

    return _build_features(df_rgb, df_other)


def get_geo(file_10m: str | Path) -> pd.DataFrame:
    """Read x/y coordinates for each whole-area prediction row."""
    df_geo = _read_csv(file_10m, usecols=[1, 2])
    df_geo.columns = ["xcoord", "ycoord"]
    return df_geo


def split_data(
    df: pd.DataFrame,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
):
    """Split a labelled DataFrame into train/test features and labels."""
    missing = [column for column in [*FEATURES, TARGET_COLUMN] if column not in df.columns]
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")

    x = df.loc[:, FEATURES]
    y = df[TARGET_COLUMN].astype(int)
    return train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
