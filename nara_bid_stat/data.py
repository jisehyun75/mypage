"""복수예가 Excel 폴더 로더.

입력 형식(발주기관별 xlsx, 첫 행 헤더):
    개찰일 | 공고명 | 기초금액 | 예정가격 | 예가/기초 | 1번 ... 15번 | 평균

* 1번~15번, 예가/기초는 기초금액 대비 편차(%) 이다.
* 파일명(확장자 제외)을 발주기관명으로 쓴다.
* 여러 파일에 같은 공고가 들어 있으면(개찰일·공고명·기초금액·예정가격 동일) 1건만 남긴다.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .mechanism import N_PRICES

PRICE_COLS = [f"{i}번" for i in range(1, N_PRICES + 1)]
P_COLS = [f"p{i}" for i in range(1, N_PRICES + 1)]


def _num(v) -> float:
    if v is None:
        return np.nan
    if isinstance(v, (int, float, np.integer, np.floating)):
        return float(v)
    s = str(v).replace(",", "").replace("%", "").strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


_YYMMDD = re.compile(r"^(\d{2})[.\-/](\d{1,2})[.\-/](\d{1,2})$")


def parse_date(v) -> pd.Timestamp:
    if v is None:
        return pd.NaT
    if isinstance(v, pd.Timestamp):
        return v.normalize()
    if hasattr(v, "year") and hasattr(v, "month"):
        return pd.Timestamp(v).normalize()
    s = str(v).strip()
    m = _YYMMDD.match(s)
    if m:
        return pd.Timestamp(2000 + int(m.group(1)), int(m.group(2)), int(m.group(3)))
    try:
        return pd.Timestamp(s).normalize()
    except (ValueError, TypeError):
        return pd.NaT


def read_prebid_workbook(path: str | Path, org: str | None = None) -> pd.DataFrame:
    import openpyxl  # 지연 import: 분석만 할 때는 필요 없음

    path = Path(path)
    org = org or path.stem
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    rows = []
    try:
        for ws in wb.worksheets:
            it = ws.iter_rows(values_only=True)
            header = next(it, None)
            if not header:
                continue
            header = [str(h).strip() if h is not None else "" for h in header]
            if not all(c in header for c in PRICE_COLS):
                continue
            pos = {h: i for i, h in enumerate(header)}
            for r in it:
                if r is None or all(v is None for v in r):
                    continue
                get = lambda c: r[pos[c]] if c in pos and pos[c] < len(r) else None  # noqa: E731
                rec = {
                    "org": org,
                    "date": parse_date(get("개찰일")),
                    "title": get("공고명"),
                    "base": _num(get("기초금액")),
                    "expected": _num(get("예정가격")),
                    "rate": _num(get("예가/기초")),
                }
                for c, p in zip(PRICE_COLS, P_COLS):
                    rec[p] = _num(get(c))
                rows.append(rec)
    finally:
        wb.close()
    return pd.DataFrame(rows)


def load_prebid_folder(folder: str | Path, patterns: Iterable[str] = ("*.xlsx",)) -> pd.DataFrame:
    """폴더 안 모든 복수예가 Excel 을 읽어 정제된 하나의 표로 돌려준다.

    반환 컬럼: org, date, title, base, expected, rate, p1..p15(오름차순), half_range
    """
    folder = Path(folder)
    files = sorted({p for pat in patterns for p in folder.glob(pat) if not p.name.startswith("~$")})
    frames = [read_prebid_workbook(p) for p in files]
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise FileNotFoundError(f"복수예가 Excel 을 찾지 못했습니다: {folder}")
    return clean_prebid_frame(pd.concat(frames, ignore_index=True))


def clean_prebid_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 예가/기초가 비어 있으면 금액으로 계산
    calc = (df["expected"] / df["base"] - 1.0) * 100.0
    df["rate"] = df["rate"].where(df["rate"].notna(), calc)
    ok = df[P_COLS].notna().all(axis=1) & df["rate"].notna() & df["date"].notna()
    df = df[ok].copy()
    x = np.sort(df[P_COLS].to_numpy(dtype=float), axis=1)
    df[P_COLS] = x
    df["half_range"] = np.abs(x).max(axis=1)
    # 비정상 범위(±5% 초과) 제거
    df = df[df["half_range"] <= 5.0]
    df = df.drop_duplicates(["date", "title", "base", "expected"], keep="first")
    return df.sort_values(["org", "date"], kind="mergesort").reset_index(drop=True)


def prices(df: pd.DataFrame) -> np.ndarray:
    return df[P_COLS].to_numpy(dtype=float)
