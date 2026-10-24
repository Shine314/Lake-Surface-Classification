"""Random Forest training and hyperparameter search."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def train_rf_model(X_train, y_train) -> RandomForestClassifier:
    """Tune and fit a Random Forest using stratified cross-validation."""
    class_counts = pd.Series(y_train).value_counts()
    if len(class_counts) < 2:
        raise ValueError("Training data must contain both classes: hishi and water.")

    # CV folds cannot exceed the number of examples in the least common class.
    n_splits = min(5, int(class_counts.min()))
    if n_splits < 2:
        raise ValueError(
            "At least two training samples per class are required for cross-validation."
        )

    parameters = {
        "max_features": ["sqrt", "log2", None],
        "max_depth": [1, 2, 3, 4],
    }
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    estimator = RandomForestClassifier(n_estimators=300, random_state=0)

    search = GridSearchCV(
        estimator=estimator,
        param_grid=parameters,
        scoring="accuracy",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)

    print("Best parameters:", search.best_params_)
    print(f"Best cross-validation accuracy: {search.best_score_:.4f}")
    return search.best_estimator_
