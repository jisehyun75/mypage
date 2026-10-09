import unittest

import numpy as np

from nara_bid_stat.mechanism import (
    COMBOS,
    KNOWN_SCHEMES,
    RateDistribution,
    combo_means,
    equal_bins,
    identify_scheme,
    mechanism_signature,
    split_bins,
)


class ComboTests(unittest.TestCase):
    def test_1365_combinations(self):
        self.assertEqual(len(COMBOS), 1365)
        self.assertEqual(len({tuple(c) for c in COMBOS}), 1365)

    def test_combo_means_order_invariant(self):
        rng = np.random.default_rng(1)
        x = rng.uniform(-3, 3, 15)
        a = combo_means(x)
        b = combo_means(rng.permutation(x))
        np.testing.assert_allclose(a, b)
        # 모든 조합 평균의 평균 = 15개 평균
        self.assertAlmostEqual(a.mean(), x.mean(), places=12)

    def test_combo_means_batch(self):
        x = np.vstack([np.linspace(-2, 2, 15), np.zeros(15)])
        m = combo_means(x)
        self.assertEqual(m.shape, (2, 1365))
        np.testing.assert_allclose(m[1], 0.0)

    def test_wrong_size_rejected(self):
        with self.assertRaises(ValueError):
            combo_means(np.zeros(14))


class SchemeTests(unittest.TestCase):
    def test_split_edges(self):
        s = split_bins(3.0, 8)
        self.assertEqual(len(s.edges), 16)
        self.assertAlmostEqual(s.edges[8], 0.0)
        self.assertAlmostEqual(s.edges[1] - s.edges[0], 3.0 / 8)
        self.assertAlmostEqual(s.edges[-1] - s.edges[-2], 3.0 / 7)

    def test_identify_sampled(self):
        rng = np.random.default_rng(0)
        for s in (equal_bins(2.0), split_bins(3.0, 8), equal_bins(2.5)):
            x = s.sample(300, rng)
            names = identify_scheme(x)
            # 대부분 정확히 판별되고, 다른 규칙으로 잘못 판별되지 않아야 한다
            self.assertGreater(np.mean(names == s.name), 0.5)
            self.assertTrue(set(names) <= {s.name, "AMBIGUOUS"})

    def test_unknown(self):
        x = np.r_[np.full(14, -0.1), 0.1]
        self.assertEqual(identify_scheme(x)[0], "UNKNOWN")

    def test_theory_split_8_7_is_negatively_biased(self):
        # 음수 8칸/양수 7칸이면 P(양수) 가 0.5 보다 낮다(실데이터: 이론 0.449, 실제 0.439)
        d = split_bins(3.0, 8).theoretical(n_sim=100_000, seed=3)
        self.assertAlmostEqual(d.prob_positive, 0.449, delta=0.01)
        self.assertAlmostEqual(d.mean, -0.1, delta=0.01)

    def test_theory_equal_bins_symmetric(self):
        d = equal_bins(2.0).theoretical(n_sim=100_000, seed=4)
        self.assertAlmostEqual(d.prob_positive, 0.5, delta=0.01)
        self.assertAlmostEqual(d.std, 0.511, delta=0.01)

    def test_signature(self):
        s = split_bins(3.0, 8).sample(5, np.random.default_rng(2))
        self.assertTrue(all(v == "3.0/8" for v in mechanism_signature(s)))


class DistributionTests(unittest.TestCase):
    def setUp(self):
        self.d = RateDistribution.from_samples(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))

    def test_cdf_quantile(self):
        self.assertAlmostEqual(float(self.d.cdf(0.0)), 0.6)
        self.assertAlmostEqual(float(self.d.cdf(-2)), 0.0)
        self.assertAlmostEqual(float(self.d.quantile(0.5)), 0.0)
        self.assertAlmostEqual(self.d.prob_positive, 0.4)

    def test_prob_between(self):
        self.assertAlmostEqual(self.d.prob_between(-0.5, 0.5), 0.4)

    def test_buckets_signed_zero(self):
        d = RateDistribution.from_samples(np.array([-0.05, -0.03, 0.02, 0.15]))
        table = {r["bucket"]: r["prob"] for r in d.bucket_table(0.1)}
        self.assertAlmostEqual(table["-0.0x"], 0.5)
        self.assertAlmostEqual(table["+0.0x"], 0.25)
        self.assertAlmostEqual(table["+0.1x"], 0.25)
        self.assertEqual(d.bucket_of(-0.0001), "-0.0x")
        self.assertEqual(d.bucket_of(0.0), "+0.0x")

    def test_from_prebid_history_recent(self):
        h = np.vstack([np.linspace(-2, 2, 15)] * 100)
        d = RateDistribution.from_prebid_history(h, recent=10)
        self.assertEqual(d.n_history, 10)
        self.assertEqual(len(d.values), 10 * 1365)

    def test_invalid_weights(self):
        with self.assertRaises(ValueError):
            RateDistribution(np.array([1.0]), np.array([-1.0]))


if __name__ == "__main__":
    unittest.main()
