"""Two models that need no fine-tuning: TF-IDF + linear SVM, and frozen sentence embeddings + logistic regression."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC


def tfidf_svm():
    """Word 1-2 grams plus character 2-5 grams (robust to typos such as 'recieved'), linear SVM."""
    features = make_union(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
    )
    return make_pipeline(features, LinearSVC(C=1.0))


class EmbeddingClassifier:
    """Encode each query once with a frozen sentence-transformer, then fit logistic regression on top."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", encoder=None):
        if encoder is None:
            from sentence_transformers import SentenceTransformer

            encoder = SentenceTransformer(model_name, device="cpu")
        self.encoder = encoder
        self.clf = LogisticRegression(C=10.0, max_iter=3000)

    def _embed(self, texts):
        return self.encoder.encode(list(texts), batch_size=64, normalize_embeddings=True, show_progress_bar=False)

    def fit(self, texts, labels):
        self.clf.fit(self._embed(texts), labels)
        return self

    def predict(self, texts):
        return list(self.clf.predict(self._embed(texts)))
