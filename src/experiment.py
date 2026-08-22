from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from xgboost import XGBClassifier

from utils import (
    classification_metrics,
    expected_calibration_error,
    selective_prediction_analysis,
    decision_utility,
    RESULTS_DIR,
    FIGURES_DIR,
)

RANDOM_STATE = 42


def load_data():
    matches = list(Path("data").rglob("bank-additional-full.csv"))
    if not matches:
        raise FileNotFoundError(
            "Dataset not found. First run: python src/download_data.py"
        )

    path = matches[0]
    print(f"Loading: {path}")
    return pd.read_csv(path, sep=";")


def clean_data(df):
    df = df.copy()

    # Remove duration because it is only fully known after a call ends and
    # therefore causes unrealistic leakage for pre-decision prediction.
    if "duration" in df.columns:
        df = df.drop(columns=["duration"])

    df["y"] = (
        df["y"].astype(str).str.lower().map({"yes": 1, "no": 0})
    )
    df = df.dropna(subset=["y"])
    df["y"] = df["y"].astype(int)

    return df


def build_preprocessor(X):
    categorical = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numerical = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numerical),
            ("cat", categorical_pipeline, categorical),
        ]
    )


def make_pipeline(model, X_reference):
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(X_reference)),
            ("model", model),
        ]
    )


def plot_calibration(y_test, probability_dict):
    plt.figure(figsize=(8, 6))

    for name, probabilities in probability_dict.items():
        fraction_positive, mean_predicted = calibration_curve(
            y_test,
            probabilities,
            n_bins=10,
            strategy="quantile",
        )
        plt.plot(mean_predicted, fraction_positive, marker="o", label=name)

    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Observed positive frequency")
    plt.title("Probability Calibration")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "calibration_curve.png", dpi=300)
    plt.close()


def plot_selective(selective_df, metric, ylabel, title, filename):
    plt.figure(figsize=(8, 6))

    for method, group in selective_df.groupby("method"):
        plt.plot(group["coverage"], group[metric], marker="o", label=method)

    plt.xlabel("Coverage")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300)
    plt.close()


def main():
    print("=" * 72)
    print("DATA-DRIVEN DECISION INTELLIGENCE WITH UNCERTAINTY-CALIBRATED AI")
    print("=" * 72)

    df = clean_data(load_data())
    print(f"\nDataset shape after cleaning: {df.shape}")
    print("\nTarget distribution:")
    print(df["y"].value_counts())
    print("\nTarget proportions:")
    print(df["y"].value_counts(normalize=True).round(4))

    X = df.drop(columns=["y"])
    y = df["y"]

    # Final test set is never used for fitting or calibration.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print(f"\nTraining rows: {len(X_train):,}")
    print(f"Test rows:     {len(X_test):,}")

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=14,
        min_samples_leaf=3,
        class_weight="balanced",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    positive_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    xgb = XGBClassifier(
        n_estimators=220,
        max_depth=5,
        learning_rate=0.07,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
        scale_pos_weight=float(positive_weight),
    )

    base_models = {
        "Random Forest": make_pipeline(rf, X_train),
        "XGBoost": make_pipeline(xgb, X_train),
    }

    all_metrics = []
    probability_dict = {}
    fitted_models = {}

    for name, model in base_models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train, y_train)

        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)

        metrics = classification_metrics(y_test, predictions, probabilities)
        metrics["model"] = name
        metrics["calibrated"] = False
        metrics["ece"] = expected_calibration_error(y_test, probabilities)

        all_metrics.append(metrics)
        probability_dict[name] = probabilities
        fitted_models[name] = model

        print(
            f"Accuracy={metrics['accuracy']:.4f} | "
            f"F1={metrics['f1']:.4f} | "
            f"ROC-AUC={metrics['roc_auc']:.4f} | "
            f"Brier={metrics['brier_score']:.4f} | "
            f"ECE={metrics['ece']:.4f}"
        )

    print("\nTraining sigmoid-calibrated XGBoost...")

    # CalibratedClassifierCV receives an unfitted estimator and performs
    # internal CV calibration only on the training partition.
    calibrated_xgb = CalibratedClassifierCV(
        estimator=make_pipeline(
            XGBClassifier(
                n_estimators=220,
                max_depth=5,
                learning_rate=0.07,
                subsample=0.85,
                colsample_bytree=0.85,
                objective="binary:logistic",
                eval_metric="logloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=RANDOM_STATE,
                scale_pos_weight=float(positive_weight),
            ),
            X_train,
        ),
        method="sigmoid",
        cv=5,
        n_jobs=-1,
    )

    calibrated_xgb.fit(X_train, y_train)

    calibrated_probabilities = calibrated_xgb.predict_proba(X_test)[:, 1]
    calibrated_predictions = (calibrated_probabilities >= 0.5).astype(int)

    calibrated_metrics = classification_metrics(
        y_test, calibrated_predictions, calibrated_probabilities
    )
    calibrated_metrics["model"] = "XGBoost + Sigmoid Calibration"
    calibrated_metrics["calibrated"] = True
    calibrated_metrics["ece"] = expected_calibration_error(
        y_test, calibrated_probabilities
    )
    all_metrics.append(calibrated_metrics)

    print(
        f"Accuracy={calibrated_metrics['accuracy']:.4f} | "
        f"F1={calibrated_metrics['f1']:.4f} | "
        f"ROC-AUC={calibrated_metrics['roc_auc']:.4f} | "
        f"Brier={calibrated_metrics['brier_score']:.4f} | "
        f"ECE={calibrated_metrics['ece']:.4f}"
    )

    probability_dict["XGBoost + Sigmoid Calibration"] = calibrated_probabilities

    metrics_df = pd.DataFrame(all_metrics)[
        [
            "model",
            "calibrated",
            "accuracy",
            "balanced_accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
            "brier_score",
            "log_loss",
            "ece",
        ]
    ]
    metrics_df.to_csv(RESULTS_DIR / "model_metrics.csv", index=False)

    calibration_df = metrics_df[
        ["model", "brier_score", "log_loss", "ece"]
    ].copy()
    calibration_df.to_csv(
        RESULTS_DIR / "calibration_metrics.csv", index=False
    )

    plot_calibration(
        y_test,
        {
            "XGBoost": probability_dict["XGBoost"],
            "Calibrated XGBoost": calibrated_probabilities,
        },
    )

    thresholds = np.round(np.arange(0.50, 0.96, 0.05), 2)

    selective_frames = []
    methods_for_defer = {
        "Uncalibrated XGBoost": probability_dict["XGBoost"],
        "Calibrated XGBoost": calibrated_probabilities,
    }

    for method, probabilities in methods_for_defer.items():
        temp = selective_prediction_analysis(
            y_test, probabilities, thresholds
        )
        temp["method"] = method
        selective_frames.append(temp)

    selective_df = pd.concat(selective_frames, ignore_index=True)
    selective_df.to_csv(
        RESULTS_DIR / "selective_prediction.csv", index=False
    )

    utility_rows = []
    for method, probabilities in methods_for_defer.items():
        for threshold in thresholds:
            utility_rows.append(
                {
                    "method": method,
                    "threshold": threshold,
                    "utility": decision_utility(
                        y_test,
                        probabilities,
                        threshold,
                        correct_reward=1.0,
                        wrong_cost=-2.0,
                        defer_cost=-0.25,
                    ),
                }
            )

    utility_df = pd.DataFrame(utility_rows)
    utility_df.to_csv(
        RESULTS_DIR / "threshold_analysis.csv", index=False
    )

    plot_selective(
        selective_df,
        metric="selective_error",
        ylabel="Selective error",
        title="Coverage–Risk Trade-off: Predict vs. Defer",
        filename="coverage_risk_curve.png",
    )

    plot_selective(
        selective_df,
        metric="selective_accuracy",
        ylabel="Selective accuracy",
        title="Coverage–Accuracy Trade-off",
        filename="coverage_accuracy_curve.png",
    )

    plt.figure(figsize=(8, 6))
    for method, group in utility_df.groupby("method"):
        plt.plot(
            group["threshold"],
            group["utility"],
            marker="o",
            label=method,
        )

    plt.xlabel("Confidence threshold")
    plt.ylabel("Mean decision utility")
    plt.title("Decision Utility: Predict vs. Defer")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "threshold_utility.png", dpi=300)
    plt.close()

    cm = confusion_matrix(y_test, calibrated_predictions)
    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No", "Yes"],
    )
    display.plot()
    plt.title("Calibrated XGBoost Confusion Matrix")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    # Save dataset summary for direct use in the paper.
    summary = pd.DataFrame(
        [
            {
                "rows": len(df),
                "features_after_target_removal": X.shape[1],
                "positive_cases": int((y == 1).sum()),
                "negative_cases": int((y == 0).sum()),
                "positive_rate": float(y.mean()),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "random_seed": RANDOM_STATE,
            }
        ]
    )
    summary.to_csv(RESULTS_DIR / "dataset_summary.csv", index=False)

    best_row = utility_df.loc[utility_df["utility"].idxmax()]
    matching = selective_df[
        (selective_df["method"] == best_row["method"])
        & (selective_df["threshold"] == best_row["threshold"])
    ].iloc[0]

    best_policy = pd.DataFrame(
        [
            {
                "method": best_row["method"],
                "confidence_threshold": best_row["threshold"],
                "mean_utility": best_row["utility"],
                "coverage": matching["coverage"],
                "deferral_rate": matching["deferral_rate"],
                "selective_accuracy": matching["selective_accuracy"],
                "selective_error": matching["selective_error"],
            }
        ]
    )
    best_policy.to_csv(
        RESULTS_DIR / "best_decision_policy.csv", index=False
    )

    print("\n" + "=" * 72)
    print("BEST DECISION POLICY")
    print("=" * 72)
    print(best_policy.to_string(index=False))

    print("\nSaved results:")
    for p in sorted(RESULTS_DIR.glob("*.csv")):
        print(f"  {p}")

    print("\nSaved figures:")
    for p in sorted(FIGURES_DIR.glob("*.png")):
        print(f"  {p}")

    print("\nExperiment complete.")


if __name__ == "__main__":
    main()
