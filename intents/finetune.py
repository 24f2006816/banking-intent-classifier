"""Fine-tune DistilBERT for intent classification with a plain PyTorch training loop."""

from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

BASE_MODEL = "distilbert-base-uncased"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class IntentModel:
    def __init__(self, labels: list[str], base_model: str = BASE_MODEL, max_length: int = 64):
        self.labels = sorted(set(labels))
        self.label_to_id = {label: i for i, label in enumerate(self.labels)}
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            base_model, num_labels=len(self.labels),
            id2label=dict(enumerate(self.labels)), label2id=self.label_to_id,
        )

    def _batches(self, texts, labels=None, batch_size=32, shuffle=False):
        indices = list(range(len(texts)))

        def collate(idx):
            enc = self.tokenizer([texts[i] for i in idx], padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt")
            if labels is not None:
                enc["labels"] = torch.tensor([self.label_to_id[labels[i]] for i in idx])
            return enc

        return DataLoader(indices, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)

    def fit(self, texts, labels, epochs: int | None = None, lr: float = 5e-5, batch_size: int = 32,
            min_steps: int = 600, seed: int = 0, log=print) -> dict:
        """Train; small datasets get more epochs so every run takes at least `min_steps` optimizer steps."""
        set_seed(seed)
        epochs = epochs or max(3, math.ceil(min_steps * batch_size / len(texts)))
        loader = self._batches(texts, labels, batch_size, shuffle=True)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=0.01)
        total = epochs * len(loader)
        scheduler = get_linear_schedule_with_warmup(optimizer, int(0.1 * total), total)
        self.model.train()
        start = time.time()
        for epoch in range(epochs):
            running = 0.0
            for batch in loader:
                loss = self.model(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                running += loss.item()
            log(f"epoch {epoch + 1}/{epochs} loss {running / len(loader):.4f} ({time.time() - start:.0f}s)")
        self.model.eval()
        return {"epochs": epochs, "steps": total, "train_seconds": round(time.time() - start)}

    @torch.no_grad()
    def predict_proba(self, texts, batch_size: int = 64, model=None) -> np.ndarray:
        model = model or self.model
        out = []
        for batch in self._batches(list(texts), batch_size=batch_size):
            out.append(torch.softmax(model(**batch).logits, dim=-1).numpy())
        return np.concatenate(out)

    def predict(self, texts, model=None) -> list[str]:
        return [self.labels[i] for i in self.predict_proba(texts, model=model).argmax(axis=1)]

    def quantized(self):
        """int8 dynamic quantization of the linear layers: ~2x faster on CPU, a quarter of the weight size."""
        return torch.quantization.quantize_dynamic(self.model, {torch.nn.Linear}, dtype=torch.qint8)

    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        (path / "labels.json").write_text(json.dumps(self.labels), encoding="utf-8")
