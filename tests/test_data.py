from intents.data import per_class_subset


def test_reads_texts_and_labels(mini):
    assert len(mini) == 48 and len(set(mini.labels)) == 4


def test_per_class_subset_is_balanced_and_deterministic(mini):
    a = per_class_subset(mini, 3, seed=0)
    b = per_class_subset(mini, 3, seed=0)
    assert a.texts == b.texts
    assert sorted({lab: a.labels.count(lab) for lab in a.labels}.values()) == [3, 3, 3, 3]


def test_full_subset_is_unchanged(mini):
    assert per_class_subset(mini, None) is mini
