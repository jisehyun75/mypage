"""엄격한 시간순 워크포워드 검증.

규칙
----
* 대상 공고의 예측에는 같은 발주기관의 '개찰일 < 대상 개찰일' 이력만 쓴다
  (같은 날·미래 공고 제외). 예측 함수에는 그 이력만 전달되므로 구조적으로 누수가 불가능하다.
* 모든 예측기는 같은 사례 집합에서 비교한다.
* 기준선(baseline)은 '메커니즘 부트스트랩'이다. 새 엔진은 이 기준선을
  같은 사례에서 통계적으로 이겨야 의미가 있다.
"""
from __future__ import annotations

import math
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import scoring
from .data import P_COLS
from .mechanism import RateDistribution

Forecaster = Callable[[pd.DataFrame], RateDistribution]


# ---------------------------------------------------------------------------
# 기본 예측기
# ---------------------------------------------------------------------------
def mechanism_forecaster(recent: int = 60) -> Forecaster:
    """최근 N건의 15개 예가 x 1365조합 평균(권장 기준선)."""

    def f(hist: pd.DataFrame) -> RateDistribution:
        return RateDistribution.from_prebid_history(hist[P_COLS].to_numpy(dtype=float), recent=recent)

    f.__name__ = f"mechanism_{recent}"
    return f


def history_rate_forecaster(recent: int | None = None) -> Forecaster:
    """과거 실제 사정율의 경험분포."""

    def f(hist: pd.DataFrame) -> RateDistribution:
        y = hist["rate"].to_numpy(dtype=float)
        if recent:
            y = y[-recent:]
        return RateDistribution.from_samples(y, source=f"history_rates(recent={recent})", n_history=len(y))

    return f


def recent_normal_forecaster(recent: int = 20, n_points: int = 2001) -> Forecaster:
    """'최근 흐름' 방식 비교용: 최근 N건 평균·표준편차의 정규분포."""
    from statistics import NormalDist

    z = np.array([NormalDist().inv_cdf((i + 0.5) / n_points) for i in range(n_points)])

    def f(hist: pd.DataFrame) -> RateDistribution:
        y = hist["rate"].to_numpy(dtype=float)[-recent:]
        mu = float(np.mean(y))
        sd = float(np.std(y, ddof=1)) if len(y) > 1 else 0.5
        return RateDistribution.from_samples(mu + max(sd, 1e-6) * z, source=f"recent_normal({recent})", n_history=len(y))

    return f


def point_forecaster(fn: Callable[[pd.DataFrame], float]) -> Forecaster:
    """점예측 엔진(예: 기존 엔진 최종값)을 분포 평가에 넣기 위한 래퍼."""

    def f(hist: pd.DataFrame) -> RateDistribution:
        return RateDistribution.from_samples(np.array([float(fn(hist))]), source="point")

    return f


# ---------------------------------------------------------------------------
# 워크포워드
# ---------------------------------------------------------------------------
def walk_forward(
    df: pd.DataFrame,
    forecasters: Mapping[str, Forecaster],
    *,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    min_history: int = 30,
    group_col: str = "org",
    max_cases: int | None = None,
    seed: int = 0,
    levels: Sequence[float] = (0.5, 0.8),
    top_k: Sequence[int] = (1, 3, 5, 10),
    width: float = 0.1,
) -> pd.DataFrame:
    """사례별 점수표를 돌려준다(행 = 대상 공고)."""
    df = df.sort_values([group_col, "date"], kind="mergesort").reset_index(drop=True)
    start = pd.Timestamp(start) if start is not None else None
    end = pd.Timestamp(end) if end is not None else None

    targets: list[tuple[str, np.ndarray, int, int]] = []
    groups = {k: idx for k, idx in df.groupby(group_col, sort=False).indices.items()}
    for g, idx in groups.items():
        dates = df["date"].to_numpy()[idx]
        for j in range(len(idx)):
            t = pd.Timestamp(dates[j])
            if (start is not None and t < start) or (end is not None and t > end):
                continue
            n_past = int(np.searchsorted(dates, dates[j], side="left"))  # 같은 날 제외
            if n_past >= min_history:
                targets.append((g, idx, j, n_past))
    if max_cases is not None and len(targets) > max_cases:
        rng = np.random.default_rng(seed)
        pick = np.sort(rng.choice(len(targets), size=max_cases, replace=False))
        targets = [targets[i] for i in pick]

    out = []
    for g, idx, j, n_past in targets:
        row_i = idx[j]
        hist = df.iloc[idx[:n_past]]
        y = float(df.at[row_i, "rate"])
        rec = {group_col: g, "date": df.at[row_i, "date"], "title": df.at[row_i, "title"] if "title" in df else "", "y": y, "n_history": n_past}
        for name, fc in forecasters.items():
            dist = fc(hist)
            rec[f"{name}.crps"] = scoring.crps(dist, y)
            rec[f"{name}.pit"] = scoring.pit(dist, y)
            for lev in levels:
                rec[f"{name}.cov{int(round(lev * 100))}"] = scoring.interval_hit(dist, y, lev)
            p = dist.prob_positive
            rec[f"{name}.p_pos"] = p
            rec[f"{name}.brier"] = scoring.brier(p, y)
            table = dist.bucket_table(width)
            yb = dist.bucket_of(y, width)
            ranked = [r["bucket"] for r in table]
            for k in top_k:
                rec[f"{name}.top{k}"] = yb in ranked[:k]
        out.append(rec)
    return pd.DataFrame(out)


def summarize(results: pd.DataFrame, baseline: str, *, levels: Iterable[int] = (50, 80), top_k: Iterable[int] = (1, 3, 5, 10)) -> pd.DataFrame:
    """예측기별 요약 + 기준선 대비 CRPS 대응검정."""
    names = sorted({c.split(".")[0] for c in results.columns if "." in c})
    n = len(results)
    rows = []
    for name in names:
        r = {"forecaster": name, "n": n, "crps": results[f"{name}.crps"].mean()}
        for lev in levels:
            col = f"{name}.cov{lev}"
            if col in results:
                r[f"cov{lev}"] = results[col].mean()
        r["brier_sign"] = results[f"{name}.brier"].mean()
        for k in top_k:
            col = f"{name}.top{k}"
            if col in results:
                hits = int(results[col].sum())
                lo, hi = scoring.wilson_ci(hits, n)
                r[f"top{k}"] = hits / n if n else float("nan")
                r[f"top{k}_ci"] = f"[{lo:.3f},{hi:.3f}]"
        ks = scoring.ks_uniform(results[f"{name}.pit"])
        r["pit_ks_p"] = ks["p"]
        if name != baseline:
            t = scoring.paired_loss_test(results[f"{name}.crps"], results[f"{baseline}.crps"])
            r["crps_diff_vs_baseline"] = t["mean_diff"]
            r["crps_diff_p"] = t["p_two_sided"]
        rows.append(r)
    return pd.DataFrame(rows).sort_values("crps").reset_index(drop=True)


# ---------------------------------------------------------------------------
# 기존 엔진 후보(BEST-k) 평가
# ---------------------------------------------------------------------------
def mcnemar_exact(b: int, c: int) -> float:
    """대응 이진결과 McNemar 정확검정(양측). b=모델만 적중, c=기준선만 적중."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return float(min(1.0, 2 * tail))


def evaluate_candidate_ledger(
    ledger: pd.DataFrame,
    history: pd.DataFrame,
    *,
    candidate_cols: Sequence[str],
    group_col: str = "org",
    baseline: Forecaster | None = None,
    min_history: int = 30,
    width: float = 0.1,
) -> dict:
    """기존 시스템이 개찰 전에 낸 후보값(BEST-k)을 같은 사례의 무기술 기준선과 비교한다.

    ledger 필수 컬럼: group_col, date, actual, candidate_cols...
    기준선은 같은 개수(k)의 '메커니즘 분포 상위 k개 구간'이다.
    """
    baseline = baseline or mechanism_forecaster(60)
    history = history.sort_values([group_col, "date"], kind="mergesort")
    model_hits, base_hits = [], []
    for _, row in ledger.iterrows():
        h = history[(history[group_col] == row[group_col]) & (history["date"] < pd.Timestamp(row["date"]))]
        if len(h) < min_history:
            continue
        cands = [row[c] for c in candidate_cols if pd.notna(row.get(c))]
        if not cands:
            continue
        y = float(row["actual"])
        dist = baseline(h)
        k = len({(float(c) < 0, int(math.floor(abs(float(c)) / width + 1e-12))) for c in cands})
        model_hits.append(scoring.candidates_bucket_hit(cands, y, width))
        base_hits.append(scoring.top_bucket_hit(dist, y, k, width))
    m = np.array(model_hits, dtype=bool)
    b = np.array(base_hits, dtype=bool)
    n = len(m)
    only_m = int((m & ~b).sum())
    only_b = int((~m & b).sum())
    return {
        "n": n,
        "model_hit_rate": float(m.mean()) if n else float("nan"),
        "model_ci": scoring.wilson_ci(int(m.sum()), n),
        "baseline_hit_rate": float(b.mean()) if n else float("nan"),
        "baseline_ci": scoring.wilson_ci(int(b.sum()), n),
        "model_only_hits": only_m,
        "baseline_only_hits": only_b,
        "mcnemar_p": mcnemar_exact(only_m, only_b),
    }
