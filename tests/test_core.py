import unittest
import numpy as np
from core import load_data, partitions
from evaluation import choose_threshold, metrics


class FraudTests(unittest.TestCase):
    def test_disjoint_ordered_splits(self):
        X, y, _ = load_data()
        parts = partitions(X, y)
        self.assertEqual(sum(len(b) for a, b in parts), len(y))
        self.assertTrue(set(parts[0][0].index).isdisjoint(parts[2][0].index))

    def test_threshold_cost(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.2, 0.8, 0.9])
        t, c = choose_threshold(y, p)
        self.assertEqual(c, 0)
        self.assertEqual(metrics(y, p, t)["recall"], 1)

    def test_high_miss_cost(self):
        y = np.array([0, 1])
        p = np.array([0.8, 0.2])
        t, c = choose_threshold(y, p, 100, 1)
        self.assertLessEqual(t, 0.2)
