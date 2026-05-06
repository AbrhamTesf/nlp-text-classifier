from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import joblib
import os
import wandb

wandb.init(project="nlp-text-classifier",job_type="inference")

app = FastAPI(
    title="NLP Text Classifier API",
    description="Classifies news text into World, Sports, Business or Sci/Tech",
    version="1.0.0"
)


LABEL_NAMES = ["World", "Sports", "Business", "Sci/Tech"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BASELINE_PATH = os.path.join(BASE_DIR, "models", "baseline_model.pkl")
baseline_model = joblib.load(BASELINE_PATH)

DISTILBERT_PATH = os.path.join(BASE_DIR, "models", "distilbert-agnews-final")
tokenizer = AutoTokenizer.from_pretrained(DISTILBERT_PATH)
distilbert_model = AutoModelForSequenceClassification.from_pretrained(DISTILBERT_PATH)
distilbert_model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
distilbert_model.to(device)

print(f"✅ Models loaded! Running on {device}")

class TextInput(BaseModel):
    text: str
    model: str = "distilbert"  # "distilbert" or "baseline"

class PredictionOutput(BaseModel):
    label: str
    confidence: float
    all_scores: dict
    model_used: str

@app.get("/")
def root():
    return {
        "message": "NLP Text Classifier API is running!",
        "endpoints": {
            "predict": "/predict",
            "health":  "/health",
            "docs":    "/docs"
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": ["distilbert", "baseline"]}

@app.post("/predict", response_model=PredictionOutput)
def predict(input: TextInput):
    if not input.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    if input.model == "distilbert":
        # Tokenize
        inputs = tokenizer(
            input.text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Predict
        with torch.no_grad():
            outputs = distilbert_model(**inputs)
            probs   = torch.softmax(outputs.logits, dim=-1)[0]
            probs   = probs.cpu().numpy()

    elif input.model == "baseline":
        probs = baseline_model.predict_proba([input.text])[0]

    else:
        raise HTTPException(status_code=400, detail="model must be 'distilbert' or 'baseline'")

    predicted_idx = int(np.argmax(probs))

    wandb.log({
        "input_text":  input.text[:100],
        "prediction":  LABEL_NAMES[predicted_idx],
        "confidence":  float(probs[predicted_idx]),
        "model_used":  input.model
    })
    return PredictionOutput(
        label      = LABEL_NAMES[predicted_idx],
        confidence = round(float(probs[predicted_idx]), 4),
        all_scores = {
            name: round(float(prob), 4)
            for name, prob in zip(LABEL_NAMES, probs)
        },
        model_used = input.model
    )