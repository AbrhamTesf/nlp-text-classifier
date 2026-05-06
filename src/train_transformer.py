import os
import argparse
import numpy as np
import pandas as pd
import torch
import wandb
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score, f1_score

LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]
MODEL_NAME  = "distilbert-base-uncased"

def clean_text(text):
    import re
    text = str(text)
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions    = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1  = f1_score(labels, predictions, average="macro")
    return {"accuracy": acc, "f1_macro": f1}

def train(config=None):
    # Default hyperparameters
    default_config = {
        "epochs":           3,
        "batch_size":       32,
        "learning_rate":    2e-5,
        "max_length":       128,
        "model_name":       MODEL_NAME,
    }

    with wandb.init(config=config or default_config, project="nlp-text-classifier"):
        cfg = wandb.config

        print(f"\n Starting run with config: {dict(cfg)}\n")

        print("Loading dataset...")
        dataset  = load_dataset("ag_news")
        df_train = pd.DataFrame(dataset["train"])
        df_test  = pd.DataFrame(dataset["test"])

        df_train["clean_text"] = df_train["text"].apply(clean_text)
        df_test["clean_text"]  = df_test["text"].apply(clean_text)

        train_dataset = Dataset.from_pandas(df_train[["clean_text", "label"]].reset_index(drop=True))
        test_dataset  = Dataset.from_pandas(df_test[["clean_text", "label"]].reset_index(drop=True))

        print("Tokenizing...")
        tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

        def tokenize(batch):
            return tokenizer(
                batch["clean_text"],
                padding="max_length",
                truncation=True,
                max_length=cfg.max_length
            )

        train_dataset = train_dataset.map(tokenize, batched=True)
        test_dataset  = test_dataset.map(tokenize, batched=True)

        train_dataset = train_dataset.rename_column("label", "labels")
        test_dataset  = test_dataset.rename_column("label", "labels")

        train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
        test_dataset.set_format("torch",  columns=["input_ids", "attention_mask", "labels"])

        print("Loading model...")
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name,
            num_labels=4
        )
        training_args = TrainingArguments(
            output_dir           = "models/distilbert-wandb",
            num_train_epochs     = cfg.epochs,
            per_device_train_batch_size = cfg.batch_size,
            per_device_eval_batch_size  = 64,
            learning_rate        = cfg.learning_rate,
            eval_strategy        = "epoch",
            save_strategy        = "epoch",
            load_best_model_at_end      = True,
            metric_for_best_model       = "accuracy",
            logging_steps        = 100,
            fp16                 = torch.cuda.is_available(),
            report_to            = "wandb"   # ← sends metrics to W&B
        )
        trainer = Trainer(
            model           = model,
            args            = training_args,
            train_dataset   = train_dataset,
            eval_dataset    = test_dataset,
            compute_metrics = compute_metrics
        )

        trainer.train()

        results = trainer.evaluate()
        print(f"\n📊 Final Results:")
        print(f"Accuracy: {results['eval_accuracy']:.4f}")
        print(f"F1 Macro: {results['eval_f1_macro']:.4f}")

        # Log final metrics to W&B
        wandb.log({
            "final_accuracy": results["eval_accuracy"],
            "final_f1_macro": results["eval_f1_macro"]
        })

if __name__ == "main":
    train()