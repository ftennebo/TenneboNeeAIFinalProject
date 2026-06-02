import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def load_dataset(name: str):
    if name == "digits":
        data = load_digits()
    elif name == "wine":
        data = load_wine()
    elif name == "breast_cancer":
        data = load_breast_cancer()
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    return data.data, data.target, data.target_names


def build_models():
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000)),
            ]
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
        ),
    }


def save_confusion_matrix(model_name, y_test, y_pred, labels, output_dir):
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=labels,
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title(f"Confusion Matrix: {model_name}")
    fig.tight_layout()
    fig.savefig(output_dir / f"{model_name}_confusion_matrix.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train baseline models for an AI class project.")
    parser.add_argument(
        "--dataset",
        default="digits",
        choices=["digits", "wine", "breast_cancer"],
        help="Built-in dataset to use.",
    )
    args = parser.parse_args()

    X, y, labels = load_dataset(args.dataset)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    output_dir = Path(__file__).resolve().parents[1] / "results"
    output_dir.mkdir(exist_ok=True)

    summary = {
        "dataset": args.dataset,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "models": {},
    }

    for model_name, model in build_models().items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        weighted_f1 = f1_score(y_test, y_pred, average="weighted")
        report = classification_report(y_test, y_pred, output_dict=True)

        summary["models"][model_name] = {
            "accuracy": accuracy,
            "weighted_f1": weighted_f1,
            "classification_report": report,
        }

        save_confusion_matrix(model_name, y_test, y_pred, labels, output_dir)

        print(f"\nModel: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Weighted F1: {weighted_f1:.4f}")
        print(classification_report(y_test, y_pred))

    with open(output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved results to: {output_dir}")


if __name__ == "__main__":
    main()
