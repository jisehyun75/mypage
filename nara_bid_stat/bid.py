"""투찰금액 단일 산식과 낙찰확률 최적화.

같은 산식을 여러 곳에 중복 구현하면 원단위 처리(round/HALF_UP)와 순공사원가 반영 여부가
어긋나기 쉽다. 투찰금액은 이 모듈 하나로만 계산하는 것을 권장한다.

적격심사(최저가 하한 근접) 낙찰 구조
------------------------------------
내가 사정율 x 를 가정해 그 하한가로 투찰하면
    실제 사정율 Y <= x  ->  내 금액 >= 실제 하한가  ->  유효
    실제 사정율 Y >  x  ->  하한 미달  ->  무효
유효한 투찰 중 가장 낮은 금액(= 가장 작은 x)이 1순위가 된다.
따라서 낙찰확률은 '사정율 분포 F' 와 '경쟁사 가정 사정율 분포 G' 가 함께 결정한다.
사정율 분포는 메커니즘으로 거의 확정되므로, 실질적인 우위는 G(경쟁사 분포) 추정에서 나온다.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, ROUND_HALF_UP, Decimal
from typing import Sequence

import numpy as np
import pandas as pd

from .mechanism import RateDistribution

_ROUNDING = {"CEIL": ROUND_CEILING, "HALF_UP": ROUND_HALF_UP, "FLOOR": ROUND_FLOOR}


@dataclass(frozen=True)
class AwardRule:
    """공고별 낙찰 하한 규칙.

    net_cost_scales_with_rate:
        순공사원가 98% 기준을 공고된 금액 그대로 쓸지(False, 기본),
        사정율을 곱해 쓸지(True) 선택한다.
        발주기관 공고문·계약예규로 확인한 뒤 고정해서 쓰십시오.
    rounding:
        원 미만 처리. 기본 CEIL(절상)은 반올림 때문에 하한에 1원 미달하는 일을 막는다.
    """

    lower_rate_pct: float
    a_value: float = 0.0
    net_cost: float | None = None
    net_cost_ratio: float = 0.98
    net_cost_scales_with_rate: bool = False
    rounding: str = "CEIL"

    def __post_init__(self) -> None:
        if not 0 < self.lower_rate_pct <= 100:
            raise ValueError("낙찰하한율(%)은 0~100 사이여야 합니다")
        if self.rounding not in _ROUNDING:
            raise ValueError(f"rounding 은 {sorted(_ROUNDING)} 중 하나")


def _d(x) -> Decimal:
    return Decimal(str(x))


def _won(v: Decimal, rounding: str) -> int:
    return int(v.quantize(Decimal("1"), rounding=_ROUNDING[rounding]))


def expected_price(base: float, rate_pct: float) -> Decimal:
    """사정율(0기준 %) -> 예정가격(원, 소수 유지)."""
    return _d(base) * (Decimal(1) + _d(rate_pct) / Decimal(100))


def lower_limit(base: float, rate_pct: float, rule: AwardRule) -> int:
    """사정율 rate_pct 일 때 낙찰하한가(원). A값이 있으면 (예정가격 - A) x 하한율 + A."""
    p = expected_price(base, rate_pct)
    r = _d(rule.lower_rate_pct) / Decimal(100)
    a = _d(rule.a_value or 0)
    v = (p - a) * r + a if a > 0 else p * r
    return _won(v, rule.rounding)


def net_cost_floor(rule: AwardRule, rate_pct: float | None = None) -> int | None:
    if not rule.net_cost:
        return None
    v = _d(rule.net_cost) * _d(rule.net_cost_ratio)
    if rule.net_cost_scales_with_rate:
        if rate_pct is None:
            raise ValueError("net_cost_scales_with_rate=True 이면 rate_pct 가 필요합니다")
        v = v * (Decimal(1) + _d(rate_pct) / Decimal(100))
    return _won(v, "CEIL")


def bid_for_assumed_rate(base: float, x_pct: float, rule: AwardRule) -> dict:
    """사정율 x 를 가정했을 때 투찰금액과 산식 근거."""
    lim = lower_limit(base, x_pct, rule)
    ncf = net_cost_floor(rule, x_pct)
    bid = max(lim, ncf) if ncf is not None else lim
    return {
        "assumed_rate": float(x_pct),
        "expected_price": float(expected_price(base, x_pct)),
        "lower_limit": lim,
        "net_cost_floor": ncf,
        "bid": bid,
        "binding": "net_cost" if ncf is not None and ncf > lim else "lower_limit",
        "formula": ("(예정가격-A값)x하한율+A값" if rule.a_value else "예정가격x하한율") + f"; 원단위={rule.rounding}",
    }


def implied_rate(base: float, bid: float, rule: AwardRule) -> float:
    """투찰금액 -> 그 업체가 가정한 사정율(0기준 %). 개찰결과로 경쟁사 분포를 만들 때 쓴다."""
    r = rule.lower_rate_pct / 100.0
    a = float(rule.a_value or 0.0)
    p = (bid - a) / r + a if a > 0 else bid / r
    return (p / base - 1.0) * 100.0


def is_valid_bid(bid: int, base: float, actual_rate: float, rule: AwardRule) -> bool:
    ncf = net_cost_floor(rule, actual_rate)
    return bid >= lower_limit(base, actual_rate, rule) and (ncf is None or bid >= ncf)


def competitor_rates_from_results(results: pd.DataFrame, *, base_col: str = "base", bid_col: str = "bid",
                                  lower_rate_col: str = "lower_rate", a_col: str | None = "a_value") -> np.ndarray:
    """개찰결과(전체 투찰업체 금액) 표 -> 업체별 가정 사정율 배열."""
    out = []
    for _, r in results.iterrows():
        rule = AwardRule(float(r[lower_rate_col]), float(r[a_col]) if a_col and pd.notna(r.get(a_col)) else 0.0)
        out.append(implied_rate(float(r[base_col]), float(r[bid_col]), rule))
    return np.asarray(out, dtype=float)


# ---------------------------------------------------------------------------
# 낙찰확률
# ---------------------------------------------------------------------------
def win_probability_curve(
    dist: RateDistribution,
    competitor_rates: Sequence[float],
    n_competitors: int,
    grid: Sequence[float] | None = None,
    *,
    n_y: int = 20_000,
) -> pd.DataFrame:
    """P(낙찰 | 가정 사정율 x).

    가정: 경쟁사 n_competitors 곳이 competitor_rates 경험분포 G 에서 독립적으로 x 를 고른다.
        P(win | x) = E_Y[ 1(Y <= x) * (1 - P(Y <= C < x))^N ]
    동률과 적격심사 비가격 점수 탈락은 무시한다(보수적으로 해석).
    """
    c = np.sort(np.asarray(competitor_rates, dtype=float))
    c = c[np.isfinite(c)]
    if n_competitors > 0 and len(c) == 0:
        raise ValueError("경쟁사 사정율 표본이 필요합니다")
    y = dist.quantile((np.arange(n_y) + 0.5) / n_y)  # 층화 분위 표본(정렬됨)
    if grid is None:
        lo, hi = dist.quantile([0.0005, 0.9995])
        grid = np.round(np.arange(lo, hi + 1e-9, 0.001), 6)
    grid = np.asarray(grid, dtype=float)

    def g_left(t: np.ndarray) -> np.ndarray:  # P(C < t)
        return np.searchsorted(c, t, side="left") / len(c) if len(c) else np.zeros_like(t)

    gy = g_left(y)
    gx = g_left(grid)
    cnt = np.searchsorted(y, grid, side="right")
    win = np.empty(len(grid))
    for i, (k, g) in enumerate(zip(cnt, gx)):
        if k == 0:
            win[i] = 0.0
        elif n_competitors == 0:
            win[i] = k / n_y
        else:
            win[i] = float(np.sum((1.0 - (g - gy[:k])) ** n_competitors)) / n_y
    return pd.DataFrame({"assumed_rate": grid, "win_prob": win, "valid_prob": cnt / n_y})


def best_rates(curve: pd.DataFrame, k: int = 5, min_separation: float = 0.02) -> pd.DataFrame:
    """낙찰확률 상위 k개 가정 사정율(서로 min_separation 이상 떨어진 값)."""
    order = curve.sort_values("win_prob", ascending=False)
    picked: list[int] = []
    for i, row in order.iterrows():
        if all(abs(row["assumed_rate"] - curve.at[j, "assumed_rate"]) >= min_separation for j in picked):
            picked.append(i)
        if len(picked) >= k:
            break
    return curve.loc[picked].reset_index(drop=True)
