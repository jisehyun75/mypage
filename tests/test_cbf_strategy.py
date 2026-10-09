import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from nara_bid_stat.bid import AwardRule, lower_limit, win_probability_curve
from nara_bid_stat.cbf import (
    gap_sample_mask,
    industry_group,
    org_key,
    parse_band,
    quality_report,
    standardize_cbf,
    verify_floor_formula,
    verify_net_cost_rule,
)
from nara_bid_stat.competition import _make_objective, _softmax, fit_competitor_model, fit_model_set, gap_diagnostics
from nara_bid_stat.consult import consult, write_consult
from nara_bid_stat.data import P_COLS
from nara_bid_stat.mechanism import RateDistribution, split_bins
from nara_bid_stat.strategy import (
    BidderCountSampler,
    RateSourceResolver,
    _band_strategy,
    _would_win,
    learn_org_aliases,
    recommend,
    strategy_backtest,
    summarize_backtest,
)

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_cbf import make_raw_cbf  # noqa: E402


class CbfLoaderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = make_raw_cbf(600, seed=1, net_cost_share=0.3)
        cls.df = standardize_cbf(cls.raw)

    def test_helpers(self):
        self.assertEqual(org_key("조달청 대전지방지방조달청"), "조달청대전지방조달청")
        self.assertEqual(parse_band("-2.5/+2.5"), 2.5)
        self.assertEqual(parse_band(None, -3.0, 3.0), 3.0)
        self.assertEqual(industry_group("일반소방(전기),전문소방"), "소방")
        self.assertEqual(industry_group("전기,통신"), "전기+통신")

    def test_standardized_columns(self):
        df = self.df
        self.assertTrue({"notice", "org_key", "band", "rate", "winner_rate", "gap", "status", "n_bidders"} <= set(df.columns))
        self.assertEqual(int(df["status"].eq("PENDING").sum()), 3)
        done = df[df["status"].eq("COMPLETED")]
        self.assertLess(done["winner_rate_diff"].max(), 1e-3)
        self.assertTrue((done["gap"] > -1e-3).all())
        self.assertTrue(df["date"].is_monotonic_increasing)

    def test_floor_formula_ceil(self):
        v = verify_floor_formula(self.df).set_index("rounding")
        self.assertEqual(v.loc["CEIL", "exact_match"], 1.0)

    def test_net_cost_rule_report(self):
        r = verify_net_cost_rule(self.df)
        self.assertEqual(list(r["rule"]), ["공고금액 그대로(미반영)", "사정율 반영"])

    def test_quality_report_counts_typo(self):
        q = quality_report(self.df).set_index("항목")["건수"]
        self.assertGreater(q["발주기관명 '지방지방' 오타"], 0)
        self.assertEqual(q["개찰 전(PENDING)"], 3)

    def test_excel_roundtrip(self):
        from nara_bid_stat.cbf import load_cbf

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "CBF.xlsx"
            self.raw.head(50).to_excel(p, sheet_name="통합데이터", index=False)
            df = load_cbf(p)
            self.assertEqual(len(df), 50)
            self.assertAlmostEqual(float(df["band"].dropna().iloc[0]) in (2.0, 3.0), True)


class CompetitorModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        df = standardize_cbf(make_raw_cbf(2500, seed=2, sparse_below=-1.0, sparse_keep=0.4))
        cls.gaps = df[gap_sample_mask(df)]

    def test_gradient(self):
        g = self.gaps[self.gaps["band"] == 3.0].head(200)
        edges = np.linspace(-3, 3, 41)
        a = np.log(np.full(40, 1 / 40))
        f = _make_objective(edges, a, np.zeros(40), g["rate"].to_numpy(float), g["winner_rate"].to_numpy(float),
                            g["n_bidders"].to_numpy(float), 5.0, 0.3)
        h = np.random.default_rng(0).normal(0, 0.2, 40)
        _, grad = f(h)
        num = np.array([(f(h + 1e-6 * e)[0] - f(h - 1e-6 * e)[0]) / 2e-6 for e in np.eye(40)])
        np.testing.assert_allclose(grad, num, rtol=1e-4, atol=1e-4)

    def test_recovers_sparse_region(self):
        g = self.gaps[self.gaps["band"] == 3.0]
        m = fit_competitor_model(g["rate"], g["winner_rate"], g["n_bidders"], 3.0)
        t = m.table()
        mid = (t["lo"] + t["hi"]) / 2
        low = t.loc[(mid > -2.0) & (mid < -1.2), "ratio"].mean()
        center = t.loc[(mid > -0.6) & (mid < 0.6), "ratio"].mean()
        self.assertLess(low, 0.8, (low, center))
        self.assertGreater(center, 0.9)
        self.assertTrue(m.converged)
        pit = m.gap_pit(g["rate"], g["winner_rate"], g["n_bidders"])
        self.assertAlmostEqual(float(np.mean(pit)), 0.5, delta=0.04)

    def test_cdf_pdf_consistent(self):
        g = self.gaps[self.gaps["band"] == 2.0]
        m = fit_competitor_model(g["rate"], g["winner_rate"], g["n_bidders"], 2.0)
        # cdf 의 구간 증가량 = pdf 의 구간 적분 (칸 안 모양이 기준 분포를 따르므로 미세 격자로 적분)
        x = np.linspace(-1.9, 1.9, 39)
        fine = np.linspace(-1.9, 1.9, 380_001)
        mids = (fine[:-1] + fine[1:]) / 2
        integ = np.add.reduceat(m.pdf(mids) * np.diff(fine), np.arange(0, 380_000, 10_000))
        np.testing.assert_allclose(np.diff(m.cdf_left(x)), integ, rtol=1e-3)
        self.assertTrue(np.all(np.diff(m.cdf_left(np.linspace(-2.5, 4.5, 2001))) >= 0))
        self.assertEqual(float(m.cdf_left(-5)), 0.0)
        self.assertEqual(float(m.cdf_left(5)), 1.0)

    def test_model_set_and_diagnostics(self):
        ms = fit_model_set(self.gaps, min_segment_obs=200)
        self.assertIn(3.0, ms.by_band)
        self.assertIsNotNone(ms.get(3.0, "전기"))
        d = gap_diagnostics(self.gaps, ms)
        low = d[(d["band"] == 3.0) & d["region(y/band)"].str.startswith("(-0.5")]
        self.assertLess(float(low["local_ratio"].iloc[0]), 0.9)
        self.assertGreater(float(low["mean_pit_null"].iloc[0]), 0.5)  # 드문 구간 -> 간격이 길다
        center = d[(d["band"] == 3.0) & d["region(y/band)"].str.startswith("(-0.1")]
        self.assertAlmostEqual(float(center["mean_pit_null"].iloc[0]), 0.5, delta=0.06)


class StrategyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = standardize_cbf(make_raw_cbf(1500, seed=3, sparse_below=-1.0, sparse_keep=0.3, n_range=(5, 40)))

    def test_resolver_uses_only_past(self):
        rng = np.random.default_rng(0)
        x = split_bins(3.0, 8).sample(80, rng)
        pb = pd.DataFrame(x, columns=P_COLS)
        pb["org"] = "충청북도 가상시"
        pb["date"] = pd.date_range("2025-01-01", periods=80, freq="D")
        future = pb.copy()
        future["date"] = future["date"] + pd.Timedelta(days=200)
        future[P_COLS] = np.sort(rng.uniform(2.9, 3.0, (80, 15)), axis=1)  # 미래에는 극단값
        r = RateSourceResolver(pd.concat([pb, future]), None)
        d, src = r.distribution("충청북도 가상시", 3.0, "2025-06-01")
        self.assertIn("복수예가", src)
        self.assertLess(d.mean, 0.5)  # 미래 극단값이 섞였다면 2.9 이상

    def test_alias_and_suffix_matching(self):
        rng = np.random.default_rng(1)
        x = split_bins(3.0, 8).sample(40, rng)
        pb = pd.DataFrame(x, columns=P_COLS)
        pb["org"] = "청주교육지원청"
        pb["date"] = pd.date_range("2025-01-01", periods=40, freq="D")
        pb["base"], pb["expected"] = 1e8, 1e8
        r = RateSourceResolver(pb, None)
        _, src = r.distribution("충청북도교육청 충청북도청주교육지원청", 3.0, "2025-03-01")
        self.assertIn("뒷부분", src)
        cbf = pd.DataFrame({"org_key": ["가상기관A"] * 4, "base": [1e8, 2e8, 3e8, 4e8], "expected": [1e8, 2e8, 3e8, 4e8],
                            "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-04"])})
        pb2 = pd.DataFrame({"org": ["파일기관명"] * 4, "base": cbf["base"], "expected": cbf["expected"], "date": cbf["date"]})
        self.assertEqual(learn_org_aliases(cbf, pb2), {"가상기관A": "파일기관명"})

    def test_resolver_fallbacks(self):
        r = RateSourceResolver(None, self.df)
        _, s3 = r.distribution("없는기관", 3.0, "2026-01-01")
        self.assertIn("이론분포(±3", s3)
        _, s2 = r.distribution("조달청 가상지방조달청", 2.0, "2026-06-01")
        self.assertIn("CBF", s2)

    def test_bidder_sampler_past_only(self):
        s = BidderCountSampler(self.df)
        v, src = s.sample("충청북도 가상시", "전기", 3.0, 1e8, "2025-03-01")
        past = self.df[(self.df["date"] < "2025-03-01") & self.df["status"].eq("COMPLETED")]
        self.assertTrue(set(v.tolist()) <= set(past["n_bidders"].tolist()))

    def test_would_win_amount_logic(self):
        rule = AwardRule(87.745, 3_000_000)
        base, y = 100_000_000.0, -0.3
        wb = lower_limit(base, 0.1, rule)
        row = pd.Series({"base": base, "rate": y, "winner_rate": 0.1, "winner_bid": wb, "floor": lower_limit(base, y, rule),
                         "lower_rate": 87.745, "a_value": 3_000_000.0, "net_cost": np.nan})
        self.assertTrue(_would_win(row, 0.0))
        self.assertTrue(_would_win(row, -0.3))
        self.assertFalse(_would_win(row, 0.1))      # 같은 금액은 이기지 못함
        self.assertFalse(_would_win(row, -0.31))    # 하한 미달

    def test_band_strategy(self):
        curve = pd.DataFrame({"assumed_rate": [0.0, 0.01, 0.02], "contested_prob": [1.0, 0.95, 0.5],
                              "cond_prob": [0.2, 0.19, 0.1]})
        real, pred, xs = _band_strategy(curve, 0.005, 0.015)
        np.testing.assert_allclose(xs, [0.0, 0.01])
        self.assertAlmostEqual(real, (0.5 + 0.5) / 2)
        self.assertAlmostEqual(pred, 0.195)

    def test_recommend_prefers_sparse_region(self):
        gaps = self.df[gap_sample_mask(self.df)]
        m = fit_model_set(gaps).get(3.0)
        dist = split_bins(3.0, 8).theoretical(100_000, seed=1)
        row = {"base": 1e8, "lower_rate": 87.745, "a_value": 3e6, "net_cost": np.nan}
        rec = recommend(row, dist, m, np.array([20.0, 30.0]))
        self.assertLess(rec["best_rate"], -0.8)
        self.assertGreater(rec["lift"], 1.1)
        c = rec["candidates"]
        self.assertEqual(int(c.loc[0, "bid_amount"]), lower_limit(1e8, c.loc[0, "assumed_rate"], AwardRule(87.745, 3e6)))
        self.assertTrue(c["contested_prob"].is_monotonic_decreasing)
        self.assertTrue((c["sole_prob"] <= c["win_prob"] + 1e-12).all())

    def test_recommend_net_cost_binding_gives_valid_bid(self):
        from nara_bid_stat.bid import bid_for_assumed_rate, net_cost_floor

        gaps = self.df[gap_sample_mask(self.df)]
        m = fit_model_set(gaps).get(3.0)
        dist = split_bins(3.0, 8).theoretical(100_000, seed=1)
        row = {"base": 1e8, "lower_rate": 87.745, "a_value": 3e6, "net_cost": 9.2e7, "est_price": 9e7}
        rec = recommend(row, dist, m, np.array([20.0]))
        c = rec["candidates"]
        self.assertGreater(rec["best_cond_prob"], 0)
        rule = AwardRule(87.745, 3e6, 9.2e7)
        for x, amt in zip(c["assumed_rate"], c["bid_amount"]):
            self.assertEqual(int(amt), bid_for_assumed_rate(1e8, x, rule)["bid"])
            self.assertGreaterEqual(int(amt), net_cost_floor(rule, x))

    def test_backtest_no_leakage_and_summary(self):
        seen = []
        cases = strategy_backtest(self.df, None, start="2026-01-01", max_cases=60, seed=0,
                                  spy=lambda cutoff, train: seen.append((cutoff, train["date"].max())))
        self.assertGreater(len(cases), 0)
        for cutoff, latest in seen:
            self.assertLess(latest, cutoff)
        for d in cases["date"]:
            self.assertTrue(any(c <= d for c, _ in seen))
        s = summarize_backtest(cases)
        for col in ("band90_edge_vs_null", "model_calib_p", "median_p_vs_null", "detectable_lift"):
            self.assertIn(col, s.columns)
        self.assertEqual(int(s["n"].iloc[0]), len(cases))

    def test_win_curve_n_samples_between(self):
        dist = RateDistribution.from_samples(np.random.default_rng(0).uniform(-1, 1, 20000))
        comp = np.random.default_rng(1).uniform(-1, 1, 5000)
        w5 = win_probability_curve(dist, comp, 5, grid=[0.0])["win_prob"].iloc[0]
        w50 = win_probability_curve(dist, comp, 50, grid=[0.0])["win_prob"].iloc[0]
        wmix = win_probability_curve(dist, comp, [5, 50], grid=[0.0])["win_prob"].iloc[0]
        self.assertAlmostEqual(wmix, (w5 + w50) / 2, delta=0.002 * wmix)  # 구적 해상도 차이 이내


class SmoothingTests(unittest.TestCase):
    def test_rate_history_bandwidth(self):
        d = RateDistribution.from_rate_history(np.random.default_rng(0).normal(0, 0.7, 150))
        self.assertGreater(d.bandwidth, 0.1)
        x = np.linspace(-2, 2, 41)
        c = d.smooth_cdf(x)
        self.assertTrue(np.all(np.diff(c) >= 0))
        self.assertAlmostEqual(float(d.smooth_cdf(10.0)), 1.0, places=6)
        pts = d.sample_points(1000)
        self.assertTrue(np.all(np.diff(pts) >= 0))

    def test_zero_bandwidth_matches_cdf(self):
        d = RateDistribution.from_samples(np.array([-1.0, 0.0, 1.0]))
        np.testing.assert_allclose(d.smooth_cdf([-0.5, 0.5]), d.cdf([-0.5, 0.5]))


class ConsultTests(unittest.TestCase):
    def test_consult_outputs(self):
        df = standardize_cbf(make_raw_cbf(700, seed=4, n_pending=2))
        tables = consult(df, None, top_k=3)
        self.assertEqual(len(tables["요약"]), 2)
        self.assertTrue({"기본추천_사정율", "기본추천_투찰금액", "기본추천_낙찰확률"} <= set(tables["요약"].columns))
        row = tables["요약"].iloc[0]
        from nara_bid_stat.bid import AwardRule, bid_for_assumed_rate

        rule = AwardRule(float(row["낙찰하한율"]), float(row["A값"]), None)
        self.assertEqual(int(row["기본추천_투찰금액"]), bid_for_assumed_rate(float(row["기초금액"]), float(row["기본추천_사정율"]), rule)["bid"])
        with tempfile.TemporaryDirectory() as tmp:
            p = write_consult(tables, tmp)
            self.assertTrue(p["xlsx"].exists() and p["md"].exists())
            import openpyxl

            wb = openpyxl.load_workbook(p["xlsx"], read_only=True)
            self.assertIn("요약", wb.sheetnames)
            wb.close()


if __name__ == "__main__":
    unittest.main()


class EfficientMarketTests(unittest.TestCase):
    def test_null_win_formula_matches_simulation(self):
        from nara_bid_stat.bid import null_win_prob

        rng = np.random.default_rng(5)
        n, trials = 6, 200_000
        for x in (-0.5, 0.0, 0.7):
            a = (x + 1) / 2
            y = rng.uniform(-1, 1, trials)
            comp = rng.uniform(-1, 1, (trials, n))
            blocked = ((comp >= y[:, None]) & (comp < x)).any(axis=1)
            contest = (comp >= y[:, None]).any(axis=1)
            win = (y <= x) & ~blocked
            self.assertAlmostEqual(float(null_win_prob(a, n, conditional=False)), float(np.mean(win)), delta=0.003)
            self.assertAlmostEqual(float(null_win_prob(a, n)), float(np.mean(win[contest])), delta=0.003)
        # 무작위 선택의 기대값은 두 경우 모두 1/(N+2), 조건부 최댓값은 중앙값에서 (1-2^-N)/N
        a = rng.uniform(0, 1, 100_000)
        self.assertAlmostEqual(float(np.mean(null_win_prob(a, n))), 1 / (n + 2), delta=0.002)
        self.assertAlmostEqual(float(null_win_prob(0.5, n)), (1 - 2.0 ** -n) / n, places=12)

    def test_lambda_selection_prefers_null_when_market_efficient(self):
        from nara_bid_stat.competition import select_lambda

        df = standardize_cbf(make_raw_cbf(1500, seed=7, sparse_keep=1.0))  # 경쟁사 = 자연 분포
        g = df[gap_sample_mask(df) & df["band"].eq(3.0)]
        lam, t = select_lambda(g, 3.0)
        best = t.loc[t["valid_mean_loglik"].idxmax()]
        null = t.loc[t["lam"] == "null", "valid_mean_loglik"].iloc[0]
        # 귀무모형이 최선이거나, 최선과의 차이가 무시할 수준이어야 한다
        self.assertLess(best["valid_mean_loglik"] - null, 0.01)

    def test_lambda_selection_finds_structure(self):
        from nara_bid_stat.competition import select_lambda

        df = standardize_cbf(make_raw_cbf(2500, seed=8, sparse_keep=0.2))
        g = df[gap_sample_mask(df) & df["band"].eq(3.0)]
        lam, t = select_lambda(g, 3.0)
        self.assertIsNotNone(lam)

    def test_model_set_auto_and_null(self):
        from nara_bid_stat.competition import null_competitor_model

        df = standardize_cbf(make_raw_cbf(800, seed=9))
        gaps = df[gap_sample_mask(df)]
        ms = fit_model_set(gaps, lam=None)
        self.assertTrue(all(m.lam is None for m in ms.by_band.values()))
        np.testing.assert_allclose(ms.get(3.0).density_ratio(), 1.0)
        m0 = null_competitor_model(2.0)
        self.assertAlmostEqual(float(m0.cdf_left(2.0)), 0.999)  # 상단 초과 칸 질량 0.001
        self.assertEqual(float(m0.cdf_left(4.0)), 1.0)
        auto = fit_model_set(gaps)
        self.assertIn("lam", auto.summary().columns)

    def test_summary_has_null_columns(self):
        df = standardize_cbf(make_raw_cbf(900, seed=10, n_range=(3, 15)))
        cases = strategy_backtest(df, None, start="2026-01-01", max_cases=40, seed=1)
        s = summarize_backtest(cases)
        for col in ("model_null_expected", "median_edge_vs_null", "band90_edge_vs_null", "efficient_market_best_wins"):
            self.assertIn(col, s.columns)
        self.assertLessEqual(float(s["model_null_expected"].iloc[0]), float(s["efficient_market_best_wins"].iloc[0]) + 1e-9)


class Round2RegressionTests(unittest.TestCase):
    def test_poisson_binomial_exact(self):
        from nara_bid_stat.scoring import poisson_binomial_cdf, poisson_binomial_test

        p = np.full(300, 0.003)
        self.assertAlmostEqual(poisson_binomial_cdf(p, 0), 0.997 ** 300, places=12)
        r = poisson_binomial_test(0, p)
        self.assertGreater(r["p_two_sided"], 0.5)  # 0건은 흔한 결과 -> 유의하지 않아야 함

    def test_empty_backtest_summary(self):
        s = summarize_backtest(pd.DataFrame())
        self.assertEqual(int(s["n"].iloc[0]), 0)

    def test_null_model_curve_is_flat(self):
        from nara_bid_stat.bid import null_win_prob, rate_density_grid
        from nara_bid_stat.competition import _base_for, null_competitor_model

        F, G = _base_for(3.0), null_competitor_model(3.0)
        yc, wc, st = rate_density_grid(F, 0.0005, min_bw=0.01)
        fs = lambda x: np.interp(x, yc + st / 2, np.cumsum(wc))  # noqa: E731
        grid = np.arange(-1.2, 1.0, 0.01)
        for n in (30, 300, 1000):
            c = win_probability_curve(F, G, n, grid=grid)
            ratio = c["cond_prob"].to_numpy() / null_win_prob(fs(grid), n)
            self.assertLess(np.max(np.abs(ratio - 1)), 0.03, n)
            rec = recommend({"base": 1e8, "lower_rate": 87.745}, F, G, np.array([n]))
            self.assertLess(rec["best_cond_prob"] / ((1 - 2.0 ** -n) / n), 1.03)

    def test_lambda_margin_prefers_null_without_structure(self):
        from nara_bid_stat.competition import select_lambda

        picked = []
        for seed in (11, 12, 13):
            df = standardize_cbf(make_raw_cbf(1500, seed=seed, sparse_keep=1.0))
            g = df[gap_sample_mask(df) & df["band"].eq(3.0)]
            lam, t = select_lambda(g, 3.0)
            picked.append(lam)
        self.assertGreaterEqual(sum(v is None for v in picked), 2)

    def test_known_scheme_uses_theory(self):
        rng = np.random.default_rng(3)
        x = split_bins(3.0, 8).sample(70, rng)
        pb = pd.DataFrame(x, columns=P_COLS)
        pb["org"] = "충청북도 가상시"
        pb["date"] = pd.date_range("2025-01-01", periods=70, freq="D")
        d, src = RateSourceResolver(pb, None).distribution("충청북도 가상시", 3.0, "2025-06-01")
        self.assertIn("SPLIT8_7_3", src)
        d2, src2 = RateSourceResolver(pb, None, use_known_scheme=False).distribution("충청북도 가상시", 3.0, "2025-06-01")
        self.assertIn("60건", src2)

    def test_csv_thousands_separators(self):
        from nara_bid_stat.cbf import load_cbf

        raw = make_raw_cbf(30, seed=5)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "cbf.csv"
            r2 = raw.copy()
            r2["기초금액"] = r2["기초금액"].map(lambda v: f"{v:,.0f}")
            r2.to_csv(p, index=False, encoding="cp949")
            df = load_cbf(p)
            self.assertEqual(int(df["base"].notna().sum()), len(raw))


class Round3RegressionTests(unittest.TestCase):
    def test_quadrature_independent_of_cell_offset(self):
        from nara_bid_stat.bid import null_win_prob
        from nara_bid_stat.competition import DistributionCompetitors, _base_for, null_competitor_model

        for band, n in ((2.0, 30), (3.0, 3000), (2.0, 3000)):
            F = _base_for(band)
            C = DistributionCompetitors(F)
            grid = float(F.quantile(0.5)) + np.arange(-0.3, 0.3, 0.0037)
            c = win_probability_curve(F, C, n, grid=grid)
            rel = c["cond_prob"].to_numpy() / null_win_prob(C.cdf_left(grid), n) - 1
            self.assertLess(np.max(np.abs(rel)), 5e-4, (band, n))
            rec = recommend({"base": 1e8, "lower_rate": 87.745}, F, null_competitor_model(band), np.array([n]))
            self.assertEqual(rec["best_rate"], rec["median_rate"], (band, n))  # 평평한 곡선 -> 중앙값

    def test_competitors_for_gates_nonstandard_distribution(self):
        from nara_bid_stat.competition import DistributionCompetitors, _base_for, null_competitor_model
        from nara_bid_stat.strategy import competitors_for

        df = standardize_cbf(make_raw_cbf(800, seed=21, sparse_keep=0.3))
        g = df[gap_sample_mask(df) & df["band"].eq(3.0)]
        m = fit_competitor_model(g["rate"], g["winner_rate"], g["n_bidders"], 3.0, lam=20.0)
        self.assertIs(competitors_for(m, _base_for(3.0), 3.0), m)
        other = RateDistribution.from_samples(np.linspace(-1, 1, 50), source="mechanism_bootstrap(recent=60)")
        self.assertIsInstance(competitors_for(m, other, 3.0), DistributionCompetitors)
        self.assertIsInstance(competitors_for(null_competitor_model(3.0), _base_for(3.0), 3.0), DistributionCompetitors)

    def test_selection_bias_correction(self):
        from nara_bid_stat.competition import _base_for, bootstrap_refits
        from nara_bid_stat.strategy import selection_bias

        df = standardize_cbf(make_raw_cbf(900, seed=22, sparse_keep=0.3))
        g = df[gap_sample_mask(df)]
        g3 = g[g["band"].eq(3.0)]
        m = fit_competitor_model(g3["rate"], g3["winner_rate"], g3["n_bidders"], 3.0, lam=20.0)
        refits = bootstrap_refits(g, 3.0, None, lam=20.0, n_boot=4, seed=1)
        self.assertEqual(len(refits), 4)
        sb = selection_bias({"base": 1e8, "lower_rate": 87.745}, _base_for(3.0), m, refits, np.array([20, 40]))
        self.assertTrue(np.isfinite(sb["factor"]) and 0.8 < sb["factor"] < 1.5, sb)

    def test_consult_missing_base_and_efficient_market_default(self):
        from nara_bid_stat.strategy import efficient_market_probs

        df = standardize_cbf(make_raw_cbf(700, seed=4, n_pending=2))
        pend = df.index[df["status"].eq("PENDING")]
        df.loc[pend[0], "base"] = np.nan
        tables = consult(df, None, top_k=3, n_boot=2)
        s = tables["요약"].set_index("공고번호")
        r = s.loc[df.loc[pend[0], "notice"]]
        self.assertTrue(np.isfinite(r["기본추천_사정율"]))
        self.assertTrue(pd.isna(r["기본추천_투찰금액"]))
        self.assertIn("투찰금액 미제공", r["비고"])
        r2 = s.loc[df.loc[pend[1], "notice"]]
        ns, _ = BidderCountSampler(df).sample(df.loc[pend[1], "org"], df.loc[pend[1], "industry_group"],
                                              df.loc[pend[1], "band"], df.loc[pend[1], "base"], df.loc[pend[1], "date"])
        self.assertAlmostEqual(r2["기본추천_낙찰확률"], efficient_market_probs(ns)["median"], places=12)
        self.assertGreaterEqual(r2["기본추천_낙찰확률"], r2["무작위_낙찰확률"])

    def test_backtest_keeps_band_without_model(self):
        df = standardize_cbf(make_raw_cbf(900, seed=23))
        win = (df["date"] >= "2026-03-01") & (df["date"] < "2026-04-01") & df["band"].eq(2.0)
        df.loc[win, "band"] = 2.5  # 학습 자료가 없는 예가변동폭
        cases = strategy_backtest(df, None, start="2026-03-01", end="2026-03-31")
        sub = cases[cases["band"].eq(2.5)]
        self.assertGreater(len(sub), 0)
        self.assertTrue((sub["lam"] == "null(학습자료 부족)").all())
        s = summarize_backtest(cases)
        self.assertIn("n_sampler_vs_actual_ratio", s.columns)

    def test_bidder_sampler_mixes_wider_level(self):
        df = standardize_cbf(make_raw_cbf(900, seed=24))
        r = df[df["status"].eq("COMPLETED")].iloc[-1]
        v, label = BidderCountSampler(df).sample(r["org"], r["industry_group"], r["band"], r["base"], r["date"])
        self.assertIn("비중 30%", label)
        v0, _ = BidderCountSampler(df, pool_weight=0.0).sample(r["org"], r["industry_group"], r["band"], r["base"], r["date"])
        self.assertLessEqual(len(v0), 30)
        self.assertGreater(len(v), len(v0))

    def test_rate_csv_reader(self):
        from nara_bid_stat.__main__ import read_rate_csv

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "c.csv"
            p.write_bytes('사정율\n0.1\n-0.2\n\n abc\n0.5\n"0,5"\n"1,234,567"\n'.encode("cp949"))
            rates, bad = read_rate_csv(p)
            np.testing.assert_allclose(rates, [0.1, -0.2, 0.5])
            self.assertEqual(bad, 3)  # 비숫자, 소수점 쉼표, 금액

    def test_lambda_default_row_marked_chosen(self):
        from nara_bid_stat.competition import select_lambda

        df = standardize_cbf(make_raw_cbf(200, seed=25))
        g = df[gap_sample_mask(df) & df["band"].eq(2.0)]
        lam, t = select_lambda(g, 2.0, min_valid=10_000)
        self.assertIsNone(lam)
        self.assertTrue(bool(t["chosen"].iloc[0]))
