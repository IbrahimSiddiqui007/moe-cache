import unittest
from stats import mean, median, variance


class T(unittest.TestCase):
    def test_mean(self):
        self.assertEqual(mean([1, 2, 3, 4]), 2.5)

    def test_median_odd(self):
        self.assertEqual(median([3, 1, 2]), 2)

    def test_median_even(self):
        self.assertEqual(median([4, 1, 3, 2]), 2.5)

    def test_variance(self):
        self.assertEqual(variance([1, 2, 3, 4]), 1.25)


if __name__ == "__main__":
    unittest.main()
