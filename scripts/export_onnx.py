"""Export the fine-tuned DistilBERT to an int8 ONNX model laid out for Transformers.js.

Usage: python scripts/export_onnx.py models/distilbert web/model
Writes config.json, tokenizer files and onnx/model_quantized.onnx, then checks that the
ONNX model's predictions match PyTorch on sample queries.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoModelForSequenceClassification, AutoTokenizer

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
(dst / "onnx").mkdir(parents=True, exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(src)
model = AutoModelForSequenceClassification.from_pretrained(src).eval()
model.config.save_pretrained(dst)
tokenizer.save_pretrained(dst)

sample = tokenizer(["where is my card?"], return_tensors="pt")
fp32 = dst / "onnx" / "model.onnx"
torch.onnx.export(
    model, (sample["input_ids"], sample["attention_mask"]), fp32,
    input_names=["input_ids", "attention_mask"], output_names=["logits"],
    dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                  "logits": {0: "batch"}},
    opset_version=17, dynamo=False,
)
q8 = dst / "onnx" / "model_quantized.onnx"
quantize_dynamic(fp32, q8, weight_type=QuantType.QInt8)
fp32.unlink()
for extra in ("labels.json",):
    if (src / extra).exists():
        shutil.copy(src / extra, dst / extra)

queries = ["My card still hasn't arrived", "why did my top up fail", "what exchange rate do you use",
           "someone stole my card", "I was charged twice", "how do I change my PIN"]
enc = tokenizer(queries, padding=True, return_tensors="pt")
with torch.no_grad():
    ref = model(**enc).logits.argmax(-1).numpy()
session = ort.InferenceSession(str(q8))
got = session.run(None, {"input_ids": enc["input_ids"].numpy(), "attention_mask": enc["attention_mask"].numpy()})[0]
agree = float((got.argmax(-1) == ref).mean())
print(f"{q8} {q8.stat().st_size / 1e6:.1f} MB; top-1 agreement with PyTorch on samples: {agree:.0%}")
assert agree >= 5 / 6
print(np.array(model.config.id2label and [model.config.id2label[i] for i in got.argmax(-1)]))
