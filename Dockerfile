FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml ./
COPY intents ./intents
RUN pip install --no-cache-dir ".[serve]"
# Download the fine-tuned model published by the training workflow.
ADD https://github.com/24f2006816/banking-intent-classifier/releases/download/model-v1/distilbert.tar.gz /tmp/model.tar.gz
RUN mkdir -p /models && tar -xzf /tmp/model.tar.gz -C /models && rm /tmp/model.tar.gz
ENV MODEL_DIR=/models/model
EXPOSE 8000
CMD ["uvicorn", "intents.serve:app", "--host", "0.0.0.0", "--port", "8000"]
