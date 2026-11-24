"""Hugging Face Space for banking-intent-classifier (free CPU tier).

Downloads the fine-tuned DistilBERT published to the GitHub release by the training workflow.
"""

import io
import json
import tarfile
import urllib.request
from pathlib import Path

try:
    import spaces  # ZeroGPU (free tier) needs at least one @spaces.GPU function; inference itself runs on CPU
except ImportError:
    spaces = None

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

URL = "https://github.com/24f2006816/banking-intent-classifier/releases/download/model-v1/distilbert.tar.gz"
MODEL_DIR = Path("model")

if not MODEL_DIR.exists():
    with urllib.request.urlopen(URL, timeout=300) as response:  # noqa: S310 - fixed https URL
        tarfile.open(fileobj=io.BytesIO(response.read()), mode="r:gz").extractall(".", filter="data")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).eval()
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
labels = json.loads((MODEL_DIR / "labels.json").read_text())


def classify(text: str):
    if not text or not text.strip():
        return {}
    enc = tokenizer([text], truncation=True, max_length=64, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**enc).logits, dim=-1)[0]
    top = probs.topk(5)
    return {labels[i].replace("_", " "): float(v) for v, i in zip(top.values, top.indices, strict=True)}


if spaces:
    classify = spaces.GPU(duration=15)(classify)

demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(label="Customer message", placeholder="I was charged twice for the same coffee"),
    outputs=gr.Label(label="Top intents", num_top_classes=5),
    title="Banking Intent Classifier",
    description=("DistilBERT fine-tuned on Banking77 (77 intents), int8-quantized for CPU. "
                 "[Code, benchmark and training workflow](https://github.com/24f2006816/banking-intent-classifier)"),
    examples=["My card still hasn't arrived after two weeks", "Why did my top-up fail?",
              "What exchange rate do you use?", "Someone stole my card, block it now",
              "I was charged twice for the same coffee"],
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
