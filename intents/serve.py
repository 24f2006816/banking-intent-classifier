"""FastAPI service: POST /predict {"text": "..."} -> top intents with probabilities.

Run: MODEL_DIR=models/distilbert uvicorn intents.serve:app
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class Query(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(3, ge=1, le=10)


class Classifier:
    """Loads a saved fine-tuned model, int8-quantized for fast CPU inference."""

    def __init__(self, model_dir: Path, quantize: bool = True):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir).eval()
        self.model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8) \
            if quantize else model
        self.labels = json.loads((Path(model_dir) / "labels.json").read_text(encoding="utf-8"))

    def predict(self, text: str, top_k: int = 3) -> list[dict]:
        enc = self.tokenizer([text], truncation=True, max_length=64, return_tensors="pt")
        with self.torch.no_grad():
            probs = self.torch.softmax(self.model(**enc).logits, dim=-1)[0]
        values, indices = probs.topk(top_k)
        return [{"intent": self.labels[i], "probability": round(float(v), 4)}
                for v, i in zip(values.tolist(), indices.tolist(), strict=True)]


def create_app(classifier=None) -> FastAPI:
    app = FastAPI(title="Banking intent classifier", version="1.0.0")
    state = {"clf": classifier}

    def get_clf():
        if state["clf"] is None:
            model_dir = os.getenv("MODEL_DIR")
            if not model_dir:
                raise HTTPException(503, "MODEL_DIR is not set")
            state["clf"] = Classifier(Path(model_dir))
        return state["clf"]

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": state["clf"] is not None}

    @app.post("/predict")
    def predict(query: Query):
        start = time.perf_counter()
        predictions = get_clf().predict(query.text, query.top_k)
        return {"predictions": predictions, "ms": round((time.perf_counter() - start) * 1000, 1)}

    return app


app = create_app()
