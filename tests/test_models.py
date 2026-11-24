import numpy as np

from intents.baselines import EmbeddingClassifier, tfidf_svm
from intents.data import per_class_subset
from intents.metrics import scores, top_confusions


def test_tfidf_baseline_learns_the_fixture(mini):
    train = per_class_subset(mini, 9, seed=0)
    clf = tfidf_svm().fit(train.texts, train.labels)
    assert scores(mini.labels, list(clf.predict(mini.texts)))["accuracy"] > 0.8


class FakeEncoder:
    """Bag-of-words hashing stand-in for a sentence-transformer, so the test needs no download."""

    def encode(self, texts, **_):
        out = np.zeros((len(texts), 256))
        for i, text in enumerate(texts):
            for word in text.lower().split():
                out[i, hash(word) % 256] += 1
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


def test_embedding_classifier_with_injected_encoder(mini):
    clf = EmbeddingClassifier(encoder=FakeEncoder()).fit(mini.texts, mini.labels)
    assert len(clf.predict(mini.texts[:5])) == 5


def test_metrics_and_confusions():
    y_true = ["a", "a", "b", "b"]
    y_pred = ["a", "b", "b", "a"]
    assert scores(y_true, y_pred)["accuracy"] == 0.5
    assert {c["count"] for c in top_confusions(y_true, y_pred)} == {1}
