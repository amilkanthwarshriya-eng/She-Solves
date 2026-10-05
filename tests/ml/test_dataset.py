import csv
import os
import re
import unittest

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "ml", "nlp", "data")
LABELS = {"FALSE_URGENCY", "CONFIRM_SHAMING", "NONE"}


def load(name):
    with open(os.path.join(DATA, f"{name}.csv"), encoding="utf-8") as f:
        return list(csv.DictReader(f))


def key(t):
    return re.sub(r"[^a-z0-9]+", " ", t.lower()).strip()


class TestDataset(unittest.TestCase):
    def test_labels_valid_and_text_present(self):
        for split in ("train", "val", "test"):
            for r in load(split):
                self.assertIn(r["label"], LABELS)
                self.assertTrue(r["text"].strip())

    def test_no_leakage_between_splits(self):
        sets = {s: {key(r["text"]) for r in load(s)} for s in ("train", "val", "test")}
        self.assertFalse(sets["train"] & sets["val"])
        self.assertFalse(sets["train"] & sets["test"])
        self.assertFalse(sets["val"] & sets["test"])

    def test_enough_samples_and_all_classes_in_every_split(self):
        self.assertGreaterEqual(len(load("all")), 150)
        for split in ("train", "val", "test"):
            self.assertEqual({r["label"] for r in load(split)}, LABELS, split)


if __name__ == "__main__":
    unittest.main()
