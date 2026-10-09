import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from nara_bid_stat.audit import null_tuning_simulation, regime_changes, selection_pit, strict_gate, utility_gate
from nara_bid_stat.data import P_COLS, load_prebid_folder
from nara_bid_stat.mechanism import N_PICK, equal_bins, split_bins
from nara_bid_stat.walkforward import (
    evaluate_candidate_ledger,
    history_rate_forecaster,
    mcnemar_exact,
    mechanism_forecaster,
    recent_normal_forecaster,
    summarize,
    walk_forward,
)


def synthetic(n_per_org=160, seed=0):
    """실제 메커니즘(칸별 균등난수 + 무작위 4개 평균)으로 만든 가짜 이력."""
    rng = np.random.default_rng(seed)
    rows = []
    for org, scheme in (("A시", split_bins(3.0, 8)), ("B청", equal_bins(2.0))):
        x = scheme.sample(n_per_org, rng)
        pick = np.argsort(rng.random(x.shape), axis=1)[:, :N_PICK]
        y = np.take_along_axis(x, pick, axis=1).mean(axis=1)
        dates = pd.Timestamp("2023-01-02") + pd.to_timedelta(np.sort(rng.integers(0, 700, n_per_org)), unit="D")
        for i in range(n_per_org):
            r = {"org": org, "date": dates[i], "title": f"{org}-{i}", "base": 1e8, "expected": 1e8 * (1 + y[i] / 100), "rate": y[i]}
            r.update(dict(zip(P_COLS, np.sort(x[i]))))
            rows.append(r)
    return pd.DataFrame(rows)


class WalkForwardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.df = synthetic()
        cls.res = walk_forward(
            cls.df,
            {"mech": mechanism_forecaster(60), "hist": history_rate_forecaster(), "recent": recent_normal_forecaster(10)},
            min_history=40,
        )

    def test_no_same_day_or_future_history(self):
        seen = []

        def spy(hist):
            seen.append(hist["date"].max())
            return mechanism_forecaster(60)(hist)

        res = walk_forward(self.df, {"spy": spy}, min_history=40)
        self.assertEqual(len(seen), len(res))
        for latest, target in zip(seen, res["date"]):
            self.assertLess(latest, target)

    def test_mechanism_is_calibrated(self):
        self.assertAlmostEqual(self.res["mech.cov80"].mean(), 0.8, delta=0.06)
        self.assertAlmostEqual(self.res["mech.cov50"].mean(), 0.5, delta=0.07)

    def test_summary_has_baseline_comparison(self):
        s = summarize(self.res, baseline="mech")
        self.assertEqual(set(s["forecaster"]), {"mech", "hist", "recent"})
        self.assertTrue(s.loc[s["forecaster"] == "recent", "crps_diff_vs_baseline"].notna().all())
        # 최근 10건 정규분포는 메커니즘보다 나쁘다(과신)
        recent = s.loc[s["forecaster"] == "recent"].iloc[0]
        mech = s.loc[s["forecaster"] == "mech"].iloc[0]
        self.assertGreater(recent["crps"], mech["crps"])

    def test_candidate_ledger(self):
        h = self.df
        last = h.groupby("org").tail(20)
        ledger = last[["org", "date"]].copy()
        ledger["actual"] = last["rate"].to_numpy()
        ledger["c1"], ledger["c2"] = -0.15, 0.05
        r = evaluate_candidate_ledger(ledger, h, candidate_cols=["c1", "c2"], min_history=40)
        self.assertGreater(r["n"], 0)
        self.assertTrue(0 <= r["mcnemar_p"] <= 1)

    def test_mcnemar(self):
        self.assertEqual(mcnemar_exact(0, 0), 1.0)
        self.assertAlmostEqual(mcnemar_exact(0, 10), 2 * 0.5 ** 10)


class AuditTests(unittest.TestCase):
    def test_selection_pit_uniform_under_random_selection(self):
        df = synthetic(n_per_org=400, seed=5)
        pit, eq = selection_pit(df[P_COLS].to_numpy(), df["rate"].to_numpy(), tol=1e-9)
        self.assertTrue((eq >= 1).all())
        self.assertAlmostEqual(float(np.mean(pit)), 0.5, delta=0.03)

    def test_utility_gate_fires_by_chance(self):
        r = null_tuning_simulation(utility_gate(), n_cases=60, trials=800, seed=1)
        self.assertGreater(r["fire_rate"], 0.3)  # 우연 통과가 흔하다

    def test_strict_gate_rarely_fires(self):
        r = null_tuning_simulation(strict_gate(0.44), n_cases=600, trials=300, seed=1)
        self.assertLess(r["fire_rate"], 0.05)

    def test_regime_change_detected(self):
        rng = np.random.default_rng(0)
        old = np.sort(rng.uniform(-2, 0, (40, 15)), axis=1)
        new = equal_bins(2.0).sample(40, rng)
        dates = pd.date_range("2021-01-01", periods=80, freq="7D")
        df = pd.DataFrame(np.vstack([old, new]), columns=P_COLS)
        df["org"], df["date"] = "X부대", dates
        ch = regime_changes(df)
        self.assertGreaterEqual(len(ch), 1)


class LoaderTests(unittest.TestCase):
    def test_load_folder_dedupes(self):
        import openpyxl

        with tempfile.TemporaryDirectory() as tmp:
            header = ["개찰일", "공고명", "기초금액", "예정가격", "예가/기초"] + [f"{i}번" for i in range(1, 16)] + ["평균"]
            x = list(np.linspace(-2.9, 2.9, 15))
            row = ["24.03.05", "도로 보수공사", 100000000, 99880000, "-0.1200", *x, "0.0"]
            for name in ("기관A", "기관A_사본"):
                wb = openpyxl.Workbook()
                ws = wb.active
                ws.append(header)
                ws.append(row)
                ws.append(["24.03.06", "다른 공사", 50000000, None, None, *x, "0.0"])  # 예가 없음 -> 제외
                wb.save(Path(tmp) / f"{name}.xlsx")
            df = load_prebid_folder(tmp)
            self.assertEqual(len(df), 1)
            self.assertEqual(df.loc[0, "date"], pd.Timestamp("2024-03-05"))
            self.assertAlmostEqual(df.loc[0, "rate"], -0.12)
            self.assertAlmostEqual(df.loc[0, "half_range"], 2.9)


if __name__ == "__main__":
    unittest.main()
