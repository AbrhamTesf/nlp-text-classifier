import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]


def evaluate_model(y_true, y_pred, model_name="Model", save_path=None):
    """Print full evaluation report and plot confusion matrix."""

    acc = accuracy_score(y_true, y_pred)
    f1  = f1_score(y_true, y_pred, average="macro")

    print(f"\n{model_name} Results:")
    print(f"Accuracy : {acc:.4f}")
    print(f"F1 Macro : {f1:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=LABEL_NAMES))

    # Plot confusion matrix
    cm   = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)

    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    plt.title(f"{model_name} — Confusion Matrix")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix saved to {save_path}")

    plt.show()
    return {"accuracy": acc, "f1_macro": f1}


def compare_models(results: dict):
    """Bar chart comparing multiple models."""
    models     = list(results.keys())
    accuracies = [v["accuracy"] for v in results.values()]
    f1_scores  = [v["f1_macro"] for v in results.values()]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, accuracies, width, label="Accuracy", color="steelblue")
    ax.bar(x + width/2, f1_scores,  width, label="F1 Macro", color="darkorange")

    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0.85, 1.0)
    ax.set_title("Model Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig("data/model_comparison.png")
    plt.show()
    print("Comparison chart saved to data/model_comparison.png")


if __name__ == "__main__":
    print("evaluate.py loaded successfully!")