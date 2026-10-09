import unittest

import numpy as np

from nara_bid_stat.bid import (
    AwardRule,
    best_rates,
    bid_for_assumed_rate,
    implied_rate,
    is_valid_bid,
    lower_limit,
    net_cost_floor,
    win_probability_curve,
)
from nara_bid_stat.mechanism import RateDistribution


class FormulaTests(unittest.TestCase):
    def test_without_a_value(self):
        rule = AwardRule(87.745)
        # 기초 100,000,000 / 사정율 -0.5 -> 예정가격 99,500,000 -> x 0.87745 = 87,306,275
        self.assertEqual(lower_limit(100_000_000, -0.5, rule), 87_306_275)

    def test_with_a_value(self):
        rule = AwardRule(87.745, a_value=5_000_000)
        # (99,500,000 - 5,000,000) * 0.87745 + 5,000,000 = 87,919,025
        self.assertEqual(lower_limit(100_000_000, -0.5, rule), 87_919_025)

    def test_ceil_never_below(self):
        rule = AwardRule(87.745, rounding="CEIL")
        rule_half = AwardRule(87.745, rounding="HALF_UP")
        exact = 123_456_789 * (1 - 0.0123 / 100) * 0.87745
        self.assertGreaterEqual(lower_limit(123_456_789, -0.0123, rule), exact)
        self.assertLessEqual(abs(lower_limit(123_456_789, -0.0123, rule_half) - exact), 0.5)

    def test_implied_rate_roundtrip(self):
        for a in (0.0, 3_210_000.0):
            rule = AwardRule(86.745, a_value=a)
            bid = lower_limit(250_000_000, 0.4321, rule)
            self.assertAlmostEqual(implied_rate(250_000_000, bid, rule), 0.4321, delta=1e-5)

    def test_validity_monotone(self):
        rule = AwardRule(87.745)
        bid = bid_for_assumed_rate(100_000_000, 0.2, rule)["bid"]
        self.assertTrue(is_valid_bid(bid, 100_000_000, 0.19, rule))
        self.assertTrue(is_valid_bid(bid, 100_000_000, 0.2, rule))
        self.assertFalse(is_valid_bid(bid, 100_000_000, 0.21, rule))

    def test_net_cost_floor_binding(self):
        rule = AwardRule(87.745, net_cost=95_000_000)
        r = bid_for_assumed_rate(100_000_000, 0.0, rule)
        self.assertEqual(r["net_cost_floor"], 93_100_000)
        self.assertEqual(r["binding"], "net_cost")
        self.assertEqual(r["bid"], 93_100_000)

    def test_net_cost_scaled_option(self):
        rule = AwardRule(87.745, net_cost=100_000_000, net_cost_scales_with_rate=True)
        self.assertEqual(net_cost_floor(rule, -1.0), 97_020_000)

    def test_bad_rule(self):
        with self.assertRaises(ValueError):
            AwardRule(187.0)


class WinProbabilityTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.dist = RateDistribution.from_samples(rng.uniform(-1, 1, 50_000))

    def test_no_competitors_equals_cdf(self):
        curve = win_probability_curve(self.dist, [], 0, grid=[-0.5, 0.0, 0.5])
        np.testing.assert_allclose(curve["win_prob"], [0.25, 0.5, 0.75], atol=0.01)

    def test_point_competitor(self):
        # 경쟁사 전원이 0.2 에 몰리면: x<0.2 -> F(x), x>0.2 -> F(x)-F(0.2)
        curve = win_probability_curve(self.dist, [0.2], 5, grid=[0.19, 0.25, 0.99])
        np.testing.assert_allclose(curve["win_prob"], [0.595, 0.025, 0.395], atol=0.01)

    def test_more_competitors_lower_win(self):
        comp = np.random.default_rng(1).uniform(-1, 1, 5000)
        c5 = win_probability_curve(self.dist, comp, 5, grid=[0.0])["win_prob"].iloc[0]
        c50 = win_probability_curve(self.dist, comp, 50, grid=[0.0])["win_prob"].iloc[0]
        self.assertLess(c50, c5)

    def test_gap_in_competitors_is_exploited(self):
        # 경쟁사가 -0.3~-0.1 에 거의 없으면 최적값이 그 빈 구간 위쪽 끝 근처에 생긴다
        rng = np.random.default_rng(2)
        comp = rng.uniform(-1, 1, 20000)
        comp = comp[(comp < -0.3) | (comp > -0.1)]
        curve = win_probability_curve(self.dist, comp, 30)
        top = best_rates(curve, k=1)["assumed_rate"].iloc[0]
        self.assertTrue(-0.3 < top <= -0.09, top)


if __name__ == "__main__":
    unittest.main()
