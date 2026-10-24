"""Model evaluation and confusion-matrix plotting."""

from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)

from config import CLASS_NAMES


def evaluate_model(
    model,
    X_test,
    y_test,
    output_path: str | Path | None = None,
    show_plot: bool = False,
) -> tuple[float, float]:
    """Print metrics and save (and optionally display) a confusion matrix."""
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    report = classification_report(
        y_test,
        y_pred,
        labels=[1, 2],
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    print("\n" + "=" * 48)
    print("MODEL EVALUATION")
    print("=" * 48)
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Weighted F1    : {weighted_f1:.4f}")
    print("\nClassification report:\n")
    print(report)

    fig, ax = plt.subplots(figsize=(7, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        labels=[1, 2],
        display_labels=CLASS_NAMES,
        ax=ax,
        cmap=plt.cm.Blues,
        values_format="d",
    )
    ax.set_title("Confusion Matrix: Hishi vs Water")
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Confusion matrix saved to: {output_path}")

    if show_plot:
        plt.show()
    plt.close(fig)
    return accuracy, weighted_f1
