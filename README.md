# Banking Intent Classifier

[![CI](https://github.com/24f2006816/banking-intent-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/24f2006816/banking-intent-classifier/actions/workflows/ci.yml)
[![Train](https://github.com/24f2006816/banking-intent-classifier/actions/workflows/train.yml/badge.svg)](https://github.com/24f2006816/banking-intent-classifier/actions/workflows/train.yml)

**Route customer-support messages to the right team automatically.** This project classifies banking customer queries into 77 intents (card arrival, failed top-up, exchange rate, stolen card and so on). It compares three approaches at three amounts of training data, so the trade-off between accuracy, labelling effort and speed is measured rather than assumed.

## Results

Official Banking77 test set (3,080 queries, 77 intents), measured by the [training workflow](https://github.com/24f2006816/banking-intent-classifier/actions/workflows/train.yml) on a CPU runner:

| Model | 10 per intent | 30 per intent | Full (10,003) | ms / query (CPU) |
| --- | --- | --- | --- | --- |
| TF-IDF + linear SVM | 73.0% | 83.8% | 91.2% | 0.8 |
| **MiniLM embeddings + logistic regression (frozen)** | **85.4%** | **89.9%** | **92.9%** | 6.4 |
| DistilBERT fine-tuned | 78.6% | 86.0% | 90.8% | 16.3 |
| DistilBERT fine-tuned, int8 quantized | | | 90.2% | 4.7 |

Accuracy shown; macro-F1 is within 0.004 of accuracy in every row (`results/metrics.json` in the release).

What the numbers say:

- **Frozen sentence embeddings win at every data size.** With only 10 labelled examples per intent, MiniLM + logistic regression reaches 85.4%, 12 points above TF-IDF. For a new support team with little labelled data, this is the model to ship.
- **Full fine-tuning did not pay off on this budget.** DistilBERT trained for 3 epochs on CPU lands at 90.8%, below the frozen MiniLM model, and is 2.5× slower per query. Published fine-tuned BERT results on Banking77 are around 93%, so longer training would likely close the gap; this repository reports what its own runs measured.
- **int8 quantization** halves the DistilBERT model size (268 MB to 139 MB) and makes it 3.5× faster on CPU, for 0.6 points of accuracy.
- **Most errors are between intents that really are close.** The most common confusion is "why verify identity" vs "verify my identity" (17 test queries), followed by "balance not updated after bank transfer" vs "transfer timing" (5).

**[Try the live demo](https://huggingface.co/spaces/24f2006816/banking-intent-classifier)**: the quantized DistilBERT running in your browser with Transformers.js.


## Dataset

[Banking77](https://github.com/PolyAI-LDN/task-specific-datasets) (Casanueva et al., 2020, CC BY 4.0): 10,003 training and 3,080 test queries, 77 fine-grained intents, many of them close neighbours ("card arrival" vs "card delivery estimate"). All models are scored on the full official test set.

## Approaches

| Model | What it is | Why include it |
| --- | --- | --- |
| TF-IDF + linear SVM | Word 1–2 grams plus character 2–5 grams, linear SVM | Strong, cheap baseline; character n-grams handle typos |
| MiniLM embeddings + logistic regression | Frozen `all-MiniLM-L6-v2` sentence vectors, linear classifier on top | Transfer learning without training the network |
| DistilBERT fine-tuned | `distilbert-base-uncased`, full fine-tuning with a plain PyTorch loop (AdamW, linear warm-up and decay, gradient clipping) | Best accuracy; tests whether fine-tuning is worth the cost |
| DistilBERT, int8 quantized | Dynamic int8 quantization of the fine-tuned model's linear layers | Faster, smaller CPU serving; measures the accuracy cost |

Each model is trained with **10 examples per intent, 30 per intent, and the full set**, because labelling is the real cost for a new support team. Small training sets get more epochs, so every fine-tuning run takes at least 600 optimizer steps.

## Run it

```bash
pip install -e ".[train]"                     # torch, transformers, sentence-transformers
python -m intents.experiment --export-model models/distilbert
```

Serve the fine-tuned model as an API:

```bash
pip install -e ".[serve]"
MODEL_DIR=models/distilbert uvicorn intents.serve:app
curl -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{"text": "Where is my new card?"}'
```

or with Docker (downloads the model from the `model-v1` release):

```bash
docker build -t intents . && docker run -p 8000:8000 intents
```

## Reproducibility

The **Train and evaluate** workflow runs the whole experiment on a GitHub-hosted CPU runner, writes the table above to the job summary, and publishes the fine-tuned model and `metrics.json` to the [`model-v1` release](https://github.com/24f2006816/banking-intent-classifier/releases/tag/model-v1). Seeds are fixed. CI runs 9 tests (data loading, subsets, both baselines, the API) on Python 3.10 and 3.12 without downloading any model.

## Project layout

```
intents/data.py         download, parse and subsample Banking77
intents/baselines.py    TF-IDF + SVM, frozen embeddings + logistic regression
intents/finetune.py     DistilBERT fine-tuning loop, prediction, quantization, export
intents/experiment.py   runs every model at every data size; writes results/
intents/serve.py        FastAPI service (/predict, /health)
scripts/export_onnx.py  int8 ONNX export for Transformers.js, with an agreement check
web/                    in-browser demo, deployed as a static Hugging Face Space
```

## License

MIT. Banking77 is by PolyAI, CC BY 4.0, downloaded from its repository.
