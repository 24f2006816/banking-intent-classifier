"""Accuracy, macro-F1 and the intents a model confuses most often."""

from __future__ import annotations

from collections import Counter

from sklearn.metrics import accuracy_score, f1_score


def scores(y_true: list[str], y_pred: list[str]) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro"), 4),
    }


def top_confusions(y_true: list[str], y_pred: list[str], n: int = 5) -> list[dict]:
    pairs = Counter((t, p) for t, p in zip(y_true, y_pred, strict=True) if t != p)
    return [{"true": t, "predicted": p, "count": c} for (t, p), c in pairs.most_common(n)]
