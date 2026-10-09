import contextlib
import io
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from nara_bid_stat.__main__ import main
from nara_bid_stat.cache import cached_frame
from nara_bid_stat.cbf import manual_notice_row, quality_report, standardize_cbf
from nara_bid_stat.consult import consult, notice_cards

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_cbf import make_raw_cbf  # noqa: E402


def _run(argv) -> str:
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        main(argv)
    return buf.getvalue()


class CacheTests(unittest.TestCase):
    def test_cached_frame_reuses_and_refreshes(self):
        with tempfile.TemporaryDirectory() as tmp:
            src = Path(tmp) / "src.txt"
            src.write_text("a")
            calls = []

            def build():
                calls.append(1)
                return pd.DataFrame({"x": [len(calls)]})

            d = Path(tmp) / "cache"
            a = cached_frame("t", "origin", [src], build, cache_dir=d)
            b = cached_frame("t", "origin", [src], build, cache_dir=d)
            self.assertEqual(len(calls), 1)
            pd.testing.assert_frame_equal(a, b)
            time.sleep(0.01)
            src.write_text("bb")  # 원본이 바뀌면 다시 만든다
            os.utime(src, ns=(time.time_ns(), time.time_ns() + 10**9))
            c = cached_frame("t", "origin", [src], build, cache_dir=d)
            self.assertEqual(len(calls), 2)
            self.assertEqual(int(c["x"].iloc[0]), 2)
            self.assertEqual(len(list(d.glob("t_*.pkl"))), 1)  # 예전 캐시 정리
            next(d.glob("t_*.pkl")).write_bytes(b"broken")  # 손상된 캐시 -> 다시 만든다
            cached_frame("t", "origin", [src], build, cache_dir=d)
            self.assertEqual(len(calls), 3)


class AnalyzeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = make_raw_cbf(700, seed=31, n_pending=2)
        cls.df = standardize_cbf(cls.raw)

    def test_manual_notice_row_is_consistent(self):
        row = manual_notice_row(self.df.columns, notice="NEW-1", org="충청북도 가상시", base=1e8, lower_rate=87.745,
                                band=3, open_dt="2026-07-01 11:00", a_value=3e6, industry="전기")
        self.assertEqual(list(row.columns[: len(self.df.columns)]), list(self.df.columns))
        both = pd.concat([self.df, row], ignore_index=True)
        quality_report(both)  # 개찰 전 행이 섞여도 동작
        t = consult(both, None, notices=["NEW-1"], n_boot=0)
        r = t["요약"].iloc[0]
        self.assertEqual(r["공고번호"], "NEW-1")
        self.assertTrue(np.isfinite(r["기본추천_사정율"]) and r["기본추천_투찰금액"] > 0)
        with self.assertRaises(ValueError):
            manual_notice_row(self.df.columns, notice="X", org="a", base=-1, lower_rate=87.7, band=3, open_dt="2026-07-01")
        with self.assertRaises(ValueError):
            manual_notice_row(self.df.columns, notice="X", org="a", base=1e8, lower_rate=87.7, band=3, open_dt="2026-13-45")

    def test_best10_and_actual_result(self):
        done = self.df[self.df["status"].eq("COMPLETED") & (self.df["date"] >= "2026-03-01")].iloc[0]
        t = consult(self.df, None, notices=[done["notice"]], n_boot=0)
        b = t["BEST10"]
        self.assertEqual(len(b), 10)
        self.assertTrue((b["누적확률"].diff().dropna() > 0).all())
        r = t["요약"].iloc[0]
        self.assertAlmostEqual(r["BEST10_적중확률"], b["누적확률"].iloc[-1])
        self.assertAlmostEqual(r["실제_사정율"], done["rate"])
        self.assertIn(bool(r["BEST10_적중"]), (True, False))
        self.assertIn("실제결과_기본추천낙찰", r.index)
        card = notice_cards(t)[0]
        self.assertIn("BEST10", card)
        self.assertIn("최종 추천 사정율", card)
        self.assertIn("실제 사정율", card)

    def test_cli_notice_manual_and_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv = Path(tmp) / "cbf.csv"
            self.raw.to_csv(csv, index=False, encoding="utf-8-sig")
            pend = self.df.loc[self.df["status"].eq("PENDING"), "notice"].tolist()
            out = Path(tmp) / "out"
            text = _run(["analyze", "--cbf", str(csv), "--notice", pend[0], "--n-boot", "0", "--no-cache", "--out", str(out)])
            self.assertIn("최종 추천 사정율(투찰용)", text)
            self.assertTrue((out / f"분석_{pend[0]}.xlsx").exists())
            text = _run(["analyze", "--cbf", str(csv), "--list", "--no-cache"])
            self.assertIn(pend[0], text)
            text = _run(["analyze", "--cbf", str(csv), "--notice", "NEW-2", "--org", "충청북도 가상시", "--industry", "전기",
                         "--base", "123456000", "--lower-rate", "87.745", "--a-value", "3700000", "--band=-3/+3",
                         "--date", "2026-07-02 11:00", "--n-boot", "0", "--no-cache", "--out", str(out)])
            self.assertIn("NEW-2", text)
            self.assertTrue((out / "분석_NEW-2.xlsx").exists())
            with self.assertRaises(SystemExit):
                _run(["analyze", "--cbf", str(csv), "--notice", "NO-SUCH", "--no-cache", "--out", str(out)])
            with self.assertRaises(SystemExit):  # 직접 입력 필수값 누락
                _run(["analyze", "--cbf", str(csv), "--notice", "NEW-3", "--org", "x", "--no-cache", "--out", str(out)])



class AnalyzeRobustnessTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raw = make_raw_cbf(700, seed=32, n_pending=2)
        pend = raw[raw["record_status"].eq("PENDING")].iloc[0]
        old = raw[raw["record_status"].eq("COMPLETED")].iloc[5].copy()
        old["공고번호"] = pend["공고번호"]  # 국방처럼 번호 재사용
        cls.raw = pd.concat([raw, old.to_frame().T], ignore_index=True)
        cls.pending = pend["공고번호"]

    def _csv(self, tmp):
        csv = Path(tmp) / "cbf.csv"
        self.raw.to_csv(csv, index=False, encoding="utf-8-sig")
        return csv

    def test_duplicate_number_analyses_one_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv, out = self._csv(tmp), Path(tmp) / "out"
            text = _run(["analyze", "--cbf", str(csv), "--notice", self.pending, "--n-boot", "0", "--no-cache", "--out", str(out)])
            self.assertIn("2건 있습니다", text)
            self.assertEqual(text.count("최종 추천 사정율(투찰용)"), 1)
            b = pd.read_excel(out / f"분석_{self.pending}.xlsx", sheet_name="BEST10")
            self.assertEqual(len(b), 10)
            text = _run(["analyze", "--cbf", str(csv), "--notice", self.pending, "--band=-2/+2", "--base", "1e8",
                         "--n-boot", "0", "--no-cache", "--out", str(out)])
            self.assertIn("±2", text)
            self.assertIn("100,000,000원", text)

    def test_input_validation_messages(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv = self._csv(tmp)
            base = ["analyze", "--cbf", str(csv), "--notice", "N1", "--org", "x", "--base", "1e8", "--band", "3",
                    "--date", "2026-07-01", "--no-cache", "--out", str(Path(tmp) / "o")]
            for extra, msg in ((["--lower-rate", "0.89745"], "% 숫자"), (["--lower-rate", "120"], "범위"),
                               (["--lower-rate", "89.745", "--a-value", "2e8"], "기초금액보다"),
                               (["--lower-rate", "89.745", "--band", "abc"], "예가변동폭")):
                err = io.StringIO()
                with contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
                    _run(base + extra)
                self.assertIn(msg, str(cm.exception.code))

    def test_prebid_folder_detection_and_env_paths(self):
        from nara_bid_stat.__main__ import _load_prebid_opt, _resolve_cbf, _resolve_prebid

        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty"
            empty.mkdir()
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                self.assertIsNone(_load_prebid_opt(str(empty), cache=False))
            self.assertIn("xlsx 파일이 없습니다", buf.getvalue())
            cbf = Path(tmp) / "CBF.xlsx.xlsx"
            cbf.write_bytes(b"x")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(_resolve_cbf(str(Path(tmp) / "CBF.xlsx")), str(cbf))
            os.environ["NARA_CBF"] = f'"{cbf}"'
            try:
                self.assertEqual(_resolve_cbf(None), str(cbf))
            finally:
                del os.environ["NARA_CBF"]
            (Path(tmp) / "복수예가").mkdir()
            (Path(tmp) / "복수예가" / "기관.xlsx").write_bytes(b"x")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(Path(_resolve_prebid(None, str(cbf))).name, "복수예가")


    def test_base_override_checked_against_cbf_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            csv = self._csv(tmp)
            with self.assertRaises(SystemExit) as cm:  # A값(기초금액의 3%)보다 작은 기초금액
                _run(["analyze", "--cbf", str(csv), "--notice", self.pending, "--base", "1000", "--no-cache",
                      "--out", str(Path(tmp) / "o")])
            self.assertIn("CBF 의 A값", str(cm.exception.code))

    def test_path_fallbacks_shared_by_all_commands(self):
        from nara_bid_stat.__main__ import _resolve_cbf, _resolve_prebid

        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "data"
            (d / "prebid").mkdir(parents=True)  # 비어 있음
            (d / "복수예가").mkdir()
            (d / "복수예가" / "기관.xlsx").write_bytes(b"x")
            cbf = d / "CBF.xlsx"
            cbf.write_bytes(b"x")
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(_resolve_cbf(str(d)), str(cbf))  # 폴더를 주면 그 안의 CBF.xlsx
                self.assertEqual(Path(_resolve_prebid(str(d / "prebid"), str(cbf))).name, "복수예가")  # 빈 폴더 -> 옆 폴더
            with self.assertRaises(SystemExit):
                with contextlib.redirect_stdout(io.StringIO()):
                    _resolve_cbf(str(Path(tmp)))  # CBF.xlsx 없는 폴더
            csv = self._csv(tmp)
            out = Path(tmp) / "ra"
            text = _run(["run-all", "--cbf", str(csv), "--skip-backtest", "--no-cache", "--n-boot", "0", "--out", str(out)])
            self.assertTrue((out / "3_consult" / "컨설팅_보고서.xlsx").exists())
            self.assertIn("복수예가 폴더 없이", text)


if __name__ == "__main__":
    unittest.main()
