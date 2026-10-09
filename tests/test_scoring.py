import math
import unittest

import numpy as np

from nara_bid_stat import scoring
from nara_bid_stat.mechanism import RateDistribution


def brute_crps(samples, y):
    s = np.asarray(samples, float)
    return np.mean(np.abs(s - y)) - 0.5 * np.mean(np.abs(s[:, None] - s[None, :]))


class CrpsTests(unittest.TestCase):
    def test_point_mass_equals_abs_error(self):
        d = RateDistribution.from_samples(np.array([0.3]))
        self.assertAlmostEqual(scoring.crps(d, -0.2), 0.5)

    def test_matches_bruteforce(self):
        rng = np.random.default_rng(0)
        s = rng.normal(0, 1, 300)
        d = RateDistribution.from_samples(s)
        for y in (-1.3, 0.0, 0.7):
            self.assertAlmostEqual(scoring.crps(d, y), brute_crps(s, y), places=10)

    def test_weighted_matches_repeated(self):
        d1 = RateDistribution(np.array([0.0, 1.0]), np.array([1.0, 3.0]))
        d2 = RateDistribution.from_samples(np.array([0.0, 1.0, 1.0, 1.0]))
        self.assertAlmostEqual(scoring.crps(d1, 0.4), scoring.crps(d2, 0.4), places=12)

    def test_normal_analytic(self):
        # N(0,1) 에서 y=0 의 CRPS = 2*phi(0) - 1/sqrt(pi) ≈ 0.23369
        rng = np.random.default_rng(1)
        d = RateDistribution.from_samples(rng.normal(0, 1, 200_000))
        self.assertAlmostEqual(scoring.crps(d, 0.0), 2 / math.sqrt(2 * math.pi) - 1 / math.sqrt(math.pi), delta=0.003)


class PitTests(unittest.TestCase):
    def test_pit_mid(self):
        d = RateDistribution.from_samples(np.array([0.0, 1.0, 2.0, 3.0]))
        self.assertAlmostEqual(scoring.pit(d, 1.0), 0.375)
        self.assertAlmostEqual(scoring.pit(d, 1.5), 0.5)

    def test_ks(self):
        rng = np.random.default_rng(2)
        self.assertGreater(scoring.ks_uniform(rng.random(5000))["p"], 0.01)
        self.assertLess(scoring.ks_uniform(rng.random(5000) ** 2)["p"], 1e-6)


class TestsAndIntervals(unittest.TestCase):
    def test_wilson(self):
        lo, hi = scoring.wilson_ci(30, 60)
        self.assertAlmostEqual(lo, 0.3773, places=3)
        self.assertAlmostEqual(hi, 0.6227, places=3)

    def test_binom_greater(self):
        self.assertAlmostEqual(scoring.binom_test_greater(0, 10, 0.5), 1.0)
        self.assertAlmostEqual(scoring.binom_test_greater(10, 10, 0.5), 0.5 ** 10)
        self.assertAlmostEqual(scoring.binom_test_greater(7, 10, 0.5), 176 / 1024)

    def test_paired_loss_detects_difference(self):
        rng = np.random.default_rng(3)
        base = rng.random(2000)
        better = base - 0.05 + rng.normal(0, 0.01, 2000)
        r = scoring.paired_loss_test(better, base)
        self.assertLess(r["mean_diff"], 0)
        self.assertLess(r["p_two_sided"], 1e-6)

    def test_candidate_bucket_hit(self):
        self.assertTrue(scoring.candidates_bucket_hit([0.12, -0.33], -0.31))
        self.assertFalse(scoring.candidates_bucket_hit([0.05], -0.05))  # 부호 다르면 다른 구간


if __name__ == "__main__":
    unittest.main()
