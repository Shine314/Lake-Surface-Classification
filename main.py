"""Train the classifier, evaluate it, and predict the whole study area."""

import argparse
from pathlib import Path

import pandas as pd

from config import DATA_DIR, FEATURES, OUTPUT_DIR
from evaluation import evaluate_model
from model import train_rf_model
from preprocessing import (
    get_df_for_prediction,
    get_df_for_RF,
    get_geo,
    split_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Classify Lake Suwa pixels as hishi (1) or water (2)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DATA_DIR,
        help="Directory containing the six source CSV files (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for predictions and evaluation artifacts.",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2021,
        help="Year used to select the reflectance scaling rule (default: 2021).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Fraction of labelled samples reserved for testing (default: 0.3).",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="Display the confusion matrix in addition to saving it.",
    )
    return parser.parse_args()


def run(
    data_dir: str | Path = DATA_DIR,
    output_dir: str | Path = OUTPUT_DIR,
    year: int = 2021,
    test_size: float = 0.3,
    show_plot: bool = False,
) -> Path:
    data_dir = Path(data_dir).expanduser()
    output_dir = Path(output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    hishi_df = get_df_for_RF(
        data_dir / "10m_hishi.csv", data_dir / "20m_hishi.csv", "hishi", year
    )
    water_df = get_df_for_RF(
        data_dir / "10m_water.csv", data_dir / "20m_water.csv", "water", year
    )
    training_df = pd.concat([hishi_df, water_df], ignore_index=True).dropna()

    if training_df.empty:
        raise ValueError("No valid training rows remain after preprocessing.")
    print(f"Valid training samples: {len(training_df)}")
    print("Samples per class:\n", training_df["label"].value_counts().sort_index())

    X_train, X_test, y_train, y_test = split_data(
        training_df, test_size=test_size
    )

    model = train_rf_model(X_train, y_train)
    evaluate_model(
        model,
        X_test,
        y_test,
        output_path=output_dir / "confusion_matrix.png",
        show_plot=show_plot,
    )

    rgb_path = data_dir / "10m_geo.csv"
    features_all = get_df_for_prediction(
        rgb_path, data_dir / "20m_geo.csv", year
    )
    geo_all = get_geo(rgb_path)

    if len(features_all) != len(geo_all):
        raise ValueError(
            "Prediction features and coordinates have different row counts: "
            f"{len(features_all)} vs {len(geo_all)}. Check the geo CSV files."
        )

    features_all = features_all.interpolate(method="linear", limit_direction="both")
    valid_mask = features_all.loc[:, FEATURES].notna().all(axis=1)
    features_to_predict = features_all.loc[valid_mask, FEATURES]
    geo_to_predict = geo_all.loc[features_to_predict.index]

    if features_to_predict.empty:
        raise ValueError("No valid whole-area pixels remain for prediction.")

    predictions = model.predict(features_to_predict)
    result = geo_to_predict.copy()
    result.insert(0, "prediction", predictions.astype("int64"))

    output_file = output_dir / f"predictions_{year}.csv"
    result.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"Predictions saved to: {output_file}")
    print(f"Predicted pixels: {len(result)}")
    return output_file


def main() -> None:
    args = parse_args()
    run(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        year=args.year,
        test_size=args.test_size,
        show_plot=args.show_plot,
    )


if __name__ == "__main__":
    main()
