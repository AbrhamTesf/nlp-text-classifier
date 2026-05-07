import re
import pandas as pd
from datasets import load_dataset


def clean_text(text):
    """Clean raw text for NLP processing."""
    text = str(text)
    text = re.sub(r"http\S+", "", text)      
    text = re.sub(r"\s+", " ", text).strip() 
    return text


def load_and_clean_data():
    """Load AG News dataset and apply cleaning."""
    dataset = load_dataset("ag_news")

    df_train = pd.DataFrame(dataset["train"])
    df_test  = pd.DataFrame(dataset["test"])

    df_train["clean_text"] = df_train["text"].apply(clean_text)
    df_test["clean_text"]  = df_test["text"].apply(clean_text)

    df_train = df_train[df_train["clean_text"].str.strip().str.len() > 10].reset_index(drop=True)
    df_test  = df_test[df_test["clean_text"].str.strip().str.len() > 10].reset_index(drop=True)

    print(f"Data loaded — Train: {len(df_train)} | Test: {len(df_test)}")
    return df_train, df_test


def save_cleaned_data(df_train, df_test, output_dir="data"):
    """Save cleaned dataframes to CSV."""
    import os
    os.makedirs(output_dir, exist_ok=True)
    df_train.to_csv(f"{output_dir}/train_clean.csv", index=False)
    df_test.to_csv(f"{output_dir}/test_clean.csv",  index=False)
    print(f"Saved cleaned data to {output_dir}/")


if __name__ == "main":
    df_train, df_test = load_and_clean_data()
    save_cleaned_data(df_train, df_test)