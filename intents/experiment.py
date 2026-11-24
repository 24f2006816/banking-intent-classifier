"""Run every model at every training-set size and write results/metrics.json and results/metrics.md.

Usage: python -m intents.experiment [--sizes 10 30 full] [--skip-finetune] [--export-model models/distilbert]
"""

from __future__ import annotations

import argparse
import json
import os
import tarfile
import time
from pathlib import Path

from .baselines import EmbeddingClassifier, tfidf_svm
from .data import load_banking77, per_class_subset
from .metrics import scores, top_confusions

ROOT = Path(__file__).resolve().parent.parent


def timed_predict(predict, texts) -> tuple[list[str], float]:
    """Predictions plus average milliseconds per query, measured one query at a time (like a live API)."""
    sample = texts[:200]
    start = time.perf_counter()
    for text in sample:
        predict([text])
    ms = (time.perf_counter() - start) * 1000 / len(sample)
    return predict(texts), round(ms, 2)


def model_size_mb(model) -> float:
    import io

    import torch

    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return round(buffer.tell() / 1e6, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sizes", nargs="+", default=["10", "30", "full"])
    parser.add_argument("--skip-finetune", action="store_true")
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--export-model", type=Path, default=None, help="save the full-data DistilBERT here")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    if not args.skip_finetune:
        import torch

        torch.set_num_threads(os.cpu_count() or 2)
    train, test = load_banking77()
    print(f"train {len(train)} | test {len(test)} | intents {len(set(train.labels))}")
    results = []

    def record(model_name, size, n_train, preds, ms, extra=None):
        row = {"model": model_name, "per_class": size, "train_examples": n_train, **scores(test.labels, preds),
               "ms_per_query": ms, **(extra or {})}
        results.append(row)
        print(json.dumps(row))
        return row

    embedder = None
    for size in args.sizes:
        per_class = None if size == "full" else int(size)
        subset = per_class_subset(train, per_class, seed=0)

        clf = tfidf_svm().fit(subset.texts, subset.labels)
        preds, ms = timed_predict(lambda t, c=clf: list(c.predict(t)), test.texts)
        record("TF-IDF + linear SVM", size, len(subset), preds, ms)

        if not args.skip_embeddings:
            emb = EmbeddingClassifier(encoder=embedder)
            embedder = emb.encoder
            emb.fit(subset.texts, subset.labels)
            preds, ms = timed_predict(emb.predict, test.texts)
            record("MiniLM embeddings + logistic regression (frozen)", size, len(subset), preds, ms)

        if not args.skip_finetune:
            from .finetune import IntentModel

            model = IntentModel(train.labels)
            info = model.fit(subset.texts, subset.labels)
            preds, ms = timed_predict(model.predict, test.texts)
            row = record("DistilBERT fine-tuned", size, len(subset), preds, ms,
                         {**info, "size_mb": model_size_mb(model.model)})
            if size == "full":
                row["top_confusions"] = top_confusions(test.labels, preds)
                q = model.quantized()
                qpreds, qms = timed_predict(lambda t, m=q, mdl=model: mdl.predict(t, model=m), test.texts)
                record("DistilBERT fine-tuned, int8 quantized", size, len(subset), qpreds, qms,
                       {"size_mb": model_size_mb(q)})
                if args.export_model:
                    model.save(args.export_model)
                    with tarfile.open(args.export_model.with_suffix(".tar.gz"), "w:gz") as tar:
                        tar.add(args.export_model, arcname="model")

    args.out.mkdir(parents=True, exist_ok=True)
    report = {"dataset": "Banking77", "train": len(train), "test": len(test), "results": results}
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (args.out / "metrics.md").write_text(to_markdown(results), encoding="utf-8")
    print(to_markdown(results))


def to_markdown(results: list[dict]) -> str:
    lines = ["| Model | Examples per intent | Accuracy | Macro-F1 | ms / query (CPU) |",
             "| --- | --- | --- | --- | --- |"]
    for r in results:
        lines.append(f"| {r['model']} | {r['per_class']} | {r['accuracy']:.1%} | {r['macro_f1']:.3f} | "
                     f"{r['ms_per_query']} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
