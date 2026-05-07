import os
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from evaluate import evaluate_model


def build_pipeline():
    """Build TF-IDF + Logistic Regression pipeline."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=50000, ngram_range=(1, 2))),
        ("clf",   LogisticRegression(max_iter=1000, C=5))
    ])


def train_baseline(data_dir="data", model_dir="models"):
    """Train baseline model and save it."""
    # Load data
    df_train = pd.read_csv(f"{data_dir}/train_clean.csv")
    df_test  = pd.read_csv(f"{data_dir}/test_clean.csv")

    X_train, y_train = df_train["clean_text"], df_train["label"]
    X_test,  y_test  = df_test["clean_text"],  df_test["label"]

    # Train
    print("Training baseline model...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    print("Training complete!")

    # Evaluate
    y_pred = pipeline.predict(X_test)
    evaluate_model(y_test, y_pred, model_name="TF-IDF Baseline")

    # Save
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(pipeline, f"{model_dir}/baseline_model.pkl")
    print(f"✅ Model saved to {model_dir}/baseline_model.pkl")

    return pipeline


if __name__ == "main":
    train_baseline()