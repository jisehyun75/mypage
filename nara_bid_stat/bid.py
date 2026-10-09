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

import math
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
        순공사원가 98% 기준에 사정율을 곱해 쓸지(True, 기본) 공고 금액 그대로 쓸지(False).
        실데이터 검증(전기·통신·소방 2025~2026): 미반영 기준이 하한가보다 높은 34건 중 1순위가 그 기준을
        지킨 것은 12건(35%) 뿐이지만, 사정율 반영 기준은 27건 중 26건(96%)이 지켰다 -> 사정율 반영이 실제 운용이다.
        (cbf.verify_net_cost_rule 로 본인 데이터에서 다시 확인할 수 있다.)
    rounding:
        원 미만 처리. 기본 절상(CEIL). CBF 의 낙찰하한가 컬럼과 일치한다(내부 일관성). 1순위 금액이
        '절상 하한가 - 1원'에 놓인 사례는 없지만 검정력이 낮은 보조 근거다(cbf.verify_floor_formula).
        절상은 반올림으로 하한에 1원 미달하는 일을 막는 보수적 선택이기도 하다.
    """

    lower_rate_pct: float
    a_value: float = 0.0
    net_cost: float | None = None
    net_cost_ratio: float = 0.98
    net_cost_scales_with_rate: bool = True
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


def lower_limit_from_expected(expected: float | Decimal, rule: AwardRule) -> int:
    """예정가격(원) -> 낙찰하한가(원). A값이 있으면 (예정가격 - A) x 하한율 + A."""
    p = expected if isinstance(expected, Decimal) else _d(expected)
    r = _d(rule.lower_rate_pct) / Decimal(100)
    a = _d(rule.a_value or 0)
    v = (p - a) * r + a if a > 0 else p * r
    return _won(v, rule.rounding)


def lower_limit(base: float, rate_pct: float, rule: AwardRule) -> int:
    """사정율 rate_pct 일 때 낙찰하한가(원)."""
    return lower_limit_from_expected(expected_price(base, rate_pct), rule)


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
    """투찰금액 -> 그 금액이 유효한 실제 사정율의 상한(0기준 %). 개찰결과로 경쟁사 분포를 만들 때 쓴다.

    rule 에 순공사원가가 있으면(사정율 반영 기준) min(하한가 역산, 순공사원가 98% 역산) 이다.
    """
    r = rule.lower_rate_pct / 100.0
    a = float(rule.a_value or 0.0)
    p = (bid - a) / r + a if a > 0 else bid / r
    x = (p / base - 1.0) * 100.0
    if rule.net_cost and rule.net_cost_scales_with_rate:
        x = min(x, (bid / (float(rule.net_cost) * float(rule.net_cost_ratio)) - 1.0) * 100.0)
    return x


def implied_rate_array(base, bid, lower_rate_pct, a_value=None, net_cost=None, net_cost_ratio: float = 0.98) -> np.ndarray:
    """투찰금액 -> 그 금액이 유효한 실제 사정율의 상한(%), 벡터판(결측은 NaN).

    하한 산식만 있으면 하한가 역산값이고, 순공사원가가 있으면(사정율 반영 98% 기준)
        min(하한가 역산값, (투찰금액 / (0.98 x 순공사원가) - 1) x 100)
    이다. 이 값은 투찰금액에 대해 증가하므로 '금액 순위 = 이 값의 순위'가 성립한다.
    CBF 의 1순위사정율(0%) 도 같은 정의다(순공사원가가 걸린 27건에서 확인).
    """
    b = np.asarray(base, dtype=float)
    m = np.asarray(bid, dtype=float)
    r = np.asarray(lower_rate_pct, dtype=float) / 100.0
    a = np.zeros_like(b) if a_value is None else np.nan_to_num(np.asarray(a_value, dtype=float), nan=0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = np.where(a > 0, (m - a) / r + a, m / r)
        x = (p / b - 1.0) * 100.0
        if net_cost is not None:
            nc = np.nan_to_num(np.asarray(net_cost, dtype=float), nan=0.0)
            x_nc = np.where(nc > 0, (m / (net_cost_ratio * np.where(nc > 0, nc, 1.0)) - 1.0) * 100.0, np.inf)
            x = np.minimum(x, x_nc)
        return x


def net_cost_valid_upto(bid: int, rule: AwardRule) -> float:
    """순공사원가 기준 때문에 이 투찰금액이 유효한 실제 사정율의 상한(%). 기준이 없으면 +inf."""
    if not rule.net_cost:
        return float("inf")
    nc98 = float(rule.net_cost) * float(rule.net_cost_ratio)
    if rule.net_cost_scales_with_rate:
        return (bid / nc98 - 1.0) * 100.0
    return float("inf") if bid >= nc98 else float("-inf")


def is_valid_bid(bid: int, base: float, actual_rate: float, rule: AwardRule) -> bool:
    ncf = net_cost_floor(rule, actual_rate)
    return bid >= lower_limit(base, actual_rate, rule) and (ncf is None or bid >= ncf)


def competitor_rates_from_results(results: pd.DataFrame, *, base_col: str = "base", bid_col: str = "bid",
                                  lower_rate_col: str = "lower_rate", a_col: str | None = "a_value",
                                  net_cost_col: str | None = "net_cost") -> np.ndarray:
    """개찰결과(전체 투찰업체 금액) 표 -> 업체별 가정 사정율 배열."""
    out = []
    for _, r in results.iterrows():
        nc = r.get(net_cost_col) if net_cost_col else None
        rule = AwardRule(float(r[lower_rate_col]), float(r[a_col]) if a_col and pd.notna(r.get(a_col)) else 0.0,
                         float(nc) if nc is not None and pd.notna(nc) and float(nc) > 0 else None)
        out.append(implied_rate(float(r[base_col]), float(r[bid_col]), rule))
    return np.asarray(out, dtype=float)


# ---------------------------------------------------------------------------
# 낙찰확률
# ---------------------------------------------------------------------------
def _competitor_cdf(competitors):
    """경쟁사 표본 배열 또는 cdf_left(x) 를 가진 모델 -> P(C < x) 함수."""
    if hasattr(competitors, "cdf_left"):
        return competitors.cdf_left
    c = np.sort(np.asarray(competitors if competitors is not None else [], dtype=float))
    c = c[np.isfinite(c)]
    if len(c) == 0:
        return None
    return lambda t: np.searchsorted(c, np.asarray(t, dtype=float), side="left") / len(c)


def _n_mixture(n_competitors, *, exact_upto: float = 100.0, group_ratio: float = 1.25) -> tuple[np.ndarray, np.ndarray]:
    """업체수 표본 -> (대표값, 표본 비중).

    작은 N(exact_upto 이하)은 서로 다른 값을 그대로 쓴다(낙찰확률을 좌우하므로 압축하지 않음).
    큰 N 은 비율 group_ratio 이내끼리 묶고 조화평균으로 대표한다(큰 N 에서 낙찰확률 ~ 1/N 이므로 기대값 보존).
    """
    ns = np.atleast_1d(np.asarray(n_competitors, dtype=float))
    ns = np.round(ns[np.isfinite(ns) & (ns >= 0)])
    if len(ns) == 0:
        raise ValueError("업체수가 필요합니다")
    vals, wts = [], []
    small = ns[ns <= exact_upto]
    if len(small):
        u, c = np.unique(small, return_counts=True)
        vals += list(u)
        wts += list(c)
    big = np.sort(ns[ns > exact_upto])
    start = 0
    for k in range(1, len(big) + 1):
        if k == len(big) or big[k] > big[start] * group_ratio:
            grp = big[start:k]
            vals.append(len(grp) / float(np.sum(1.0 / grp)))
            wts.append(len(grp))
            start = k
    w = np.asarray(wts, dtype=float)
    return np.asarray(vals, dtype=float), w / w.sum()


def rate_density_grid(dist: RateDistribution, step: float, *, min_bw: float = 0.01,
                      max_cells: int = 400_000) -> tuple[np.ndarray, np.ndarray, float]:
    """사정율 분포 -> 등간격 칸 (중심, 질량, 실제 칸폭). 가우스 평활(대역폭 max(dist.bandwidth, min_bw))."""
    bw = max(float(dist.bandwidth), float(min_bw))
    lo = float(dist.values[0]) - 6 * bw - step
    hi = float(dist.values[-1]) + 6 * bw + step
    n = int(np.ceil((hi - lo) / step))
    if n > max_cells:
        n = max_cells
        step = (hi - lo) / n
    edges = lo + step * np.arange(n + 1)
    mass, _ = np.histogram(dist.values, bins=edges, weights=dist.weights)
    if bw > 0:
        half = int(np.ceil(6 * bw / step))
        k = np.exp(-0.5 * (np.arange(-half, half + 1) * step / bw) ** 2)
        k /= k.sum()
        size = 1 << int(np.ceil(np.log2(n + len(k))))
        conv = np.fft.irfft(np.fft.rfft(mass, size) * np.fft.rfft(k, size), size)[half:half + n]
        mass = np.clip(conv, 0.0, None)
    mass = mass / mass.sum()
    return (edges[:-1] + edges[1:]) / 2, mass, step


def win_probability_curve(
    dist: RateDistribution,
    competitors,
    n_competitors,
    grid: Sequence[float] | None = None,
    *,
    valid_upto: Sequence[float] | None = None,
    min_bw: float = 0.01,
    resolution: float = 0.02,
    n_y: int | None = None,
    density: tuple | None = None,
) -> pd.DataFrame:
    """P(낙찰 | 가정 사정율 x) — 결정적 구적법.

    competitors : 경쟁사 가정 사정율(=유효 상한) 표본 배열, 또는 cdf_left(x)=P(C<x) 를 가진 모델
    n_competitors : 경쟁 업체 수(정수) 또는 업체수 표본(혼합으로 평균)
    valid_upto : x 별 유효 상한(기본 x)

    반환 컬럼
        win_prob       = E[1(Y<=x) (1 - P(Y<=C<x))^N]           : 전체 낙찰확률
        sole_prob      = E[1(Y<=x) P(C<Y)^N]                     : 경쟁사가 모두 하한 미달이라 '나만 유효'해서 낙찰
        contested_prob = win_prob - sole_prob                    : 다른 유효 투찰이 있는데도 이기는 확률
        cond_prob      = E_N[contested_N / P_N(경쟁사 1곳 이상 유효)] : 1순위가 존재하는 공고(CBF 개찰결과)에서의 낙찰확률
                         (업체수 표본이 개찰결과가 있는 공고에서 나오므로 N 별 조건부 확률을 평균한다)
        valid_prob     = P(Y <= x)
    단독 유효 사례는 개찰결과 데이터에 나타나지 않으므로(1순위가 있는 공고만 기록) 검증되지 않는다.
    사정율 격자 해상도는 업체수에 맞춘다(칸폭 ~ resolution/(N+1)). n_y 는 하위호환용으로 무시한다.
    density 에 rate_density_grid 결과를 주면 다시 계산하지 않는다(칸폭이 충분히 작을 때).
    동률과 적격심사 비가격 점수 탈락은 무시한다.
    """
    nvals, nw = _n_mixture(n_competitors)
    cdf = _competitor_cdf(competitors)
    if cdf is None and nvals.max() > 0:
        raise ValueError("경쟁사 사정율 표본 또는 모델이 필요합니다")
    if grid is None:
        lo, hi = dist.quantile([0.0005, 0.9995])
        grid = np.round(np.arange(lo, hi + 1e-9, 0.001), 6)
    grid = np.asarray(grid, dtype=float)
    vu = grid if valid_upto is None else np.minimum(grid, np.asarray(valid_upto, dtype=float))
    gx = cdf(grid) if cdf is not None else np.zeros_like(grid)
    step_fine = float(np.clip(resolution / (nvals.max() + 1.0), 2e-5, 2e-3))
    if density is not None and density[2] <= step_fine * 1.0001:
        yc, wc, step_fine = density  # 같은 분포로 여러 번 계산할 때 재사용
    else:
        yc, wc, step_fine = rate_density_grid(dist, step_fine, min_bw=min_bw)
    total = np.zeros(len(grid))
    sole = np.zeros(len(grid))
    cond = np.zeros(len(grid))
    p_contest_mix = 0.0
    levels: dict[int, tuple] = {}
    for n, pw in zip(nvals, nw):
        k = max(1, int(round(min(resolution / (n + 1.0), 2e-3) / step_fine)))
        if k not in levels:  # 같은 해상도를 쓰는 업체수끼리 격자·경쟁사 CDF 를 공유
            starts = np.arange(0, len(yc), k)
            w = np.add.reduceat(wc, starts) if k > 1 else wc
            y = (np.add.reduceat(wc * yc, starts) / np.where(w > 0, w, 1.0)) if k > 1 else yc
            if k > 1:
                y = np.where(w > 0, y, yc[starts])
            gy = cdf(y) if cdf is not None else np.zeros_like(y)
            lg = np.log(np.clip(gy, 1e-300, 1.0))  # 단조 증가
            # x 가 들어 있는 칸은 x 아래 부분만 센다(칸 전체를 넣거나 빼면 x 의 칸 안 위치에 따라
            # 상대오차 ~ N f 칸폭/2 가 생겨 평평한 곡선에서도 가짜 최댓값이 나온다).
            le = yc[starts] - step_fine / 2  # 칸 아래 경계
            wd = np.diff(np.r_[le, yc[-1] + step_fine / 2])
            jc = np.searchsorted(le, vu, side="right") - 1  # x 가 든 칸(-1: 분포 아래)
            jcc = np.clip(jc, 0, len(le) - 1)
            frac = np.where(jc >= 0, np.clip((vu - le[jcc]) / wd[jcc], 0.0, 1.0), 0.0)
            ym = le[jcc] + frac * wd[jcc] / 2  # 칸 안에서 x 아래 부분의 가운데
            gm = cdf(ym) if cdf is not None else np.zeros_like(ym)
            levels[k] = (y, w, gy, lg, jc, frac, gm)
        y, w, gy, lg, jc, frac, gm = levels[k]
        if n > 0:
            jl = int(np.searchsorted(lg, -700.0 / n, side="left"))  # gy^n 이 0 이 아닌 구간만
            p_none = float((w[jl:] * np.exp(n * lg[jl:])).sum())
        else:
            p_none = float(w.sum())
        p_c = max(1.0 - p_none, 1e-12)  # 이 N 에서 경쟁사 1곳 이상 유효할 확률
        p_contest_mix += pw * p_c
        tot_n = np.zeros(len(grid))
        sole_n = np.zeros(len(grid))
        for i, c in enumerate(jc):
            if c < 0:
                continue
            pm = float(w[c]) * float(frac[i])  # x 가 든 칸 중 x 아래 부분의 질량
            if n == 0:
                tot_n[i] = sole_n[i] = float(w[:c].sum()) + pm
                continue
            if pm > 0:
                dm = min(max(float(gx[i] - gm[i]), 0.0), 1.0 - 1e-15)
                tot_n[i] = pm * math.exp(n * math.log1p(-dm))
                sole_n[i] = pm * float(gm[i]) ** n
            # gy 는 단조이므로 기여가 e^-60 보다 큰 구간(d < 60/n)만 계산한다. 그 밖의 gy^n 도 그보다 작다.
            # (np.dot 대신 곱셈합: 멀티스레드 BLAS 의 동기화 비용을 피한다)
            j0 = int(np.searchsorted(gy, gx[i] - 60.0 / n, side="left"))
            if j0 >= c:
                continue
            d = np.clip(gx[i] - gy[j0:c], 0.0, 1.0 - 1e-15)
            ww = w[j0:c]
            tot_n[i] += float((ww * np.exp(n * np.log1p(-d))).sum())
            sole_n[i] += float((ww * np.exp(n * lg[j0:c])).sum())
        total += pw * tot_n
        sole += pw * sole_n
        cond += pw * np.clip(tot_n - sole_n, 0.0, None) / p_c  # 업체수 표본이 '1순위 있는 공고'에서 나왔으므로 조건부의 평균
    valid = np.array([rate_cdf_on_grid(yc, wc, v, step_fine) for v in vu])
    contested = np.clip(total - sole, 0.0, None)
    out = pd.DataFrame({"assumed_rate": grid, "win_prob": total, "sole_prob": sole, "contested_prob": contested,
                        "cond_prob": cond, "valid_prob": valid})
    out.attrs["p_contest"] = p_contest_mix
    return out


def rate_cdf_on_grid(yc: np.ndarray, wc: np.ndarray, x: float, step: float | None = None) -> float:
    """등간격 칸(중심 yc, 질량 wc)의 P(Y <= x). step(칸폭)을 주면 x 가 든 칸은 x 아래 비율만 센다."""
    if step is None:
        return float(wc[: np.searchsorted(yc, x, side="right")].sum())
    j = int(np.searchsorted(yc - step / 2, x, side="right")) - 1
    if j < 0:
        return 0.0
    if j >= len(yc):
        return float(wc.sum())
    return float(wc[:j].sum() + wc[j] * min(max((x - (yc[j] - step / 2)) / step, 0.0), 1.0))


def null_win_prob(f_x, n, *, conditional: bool = True):
    """효율적 시장(경쟁사 N곳이 사정율 분포와 같은 분포)에서 F(x)=f_x 인 x 의 낙찰확률.

    conditional=True  : 1순위가 존재하는(경쟁사 1곳 이상 유효) 공고 기준
                        (1 - (1-a)^(N+1) - a^(N+1)) / N   -> 중앙값(a=0.5)에서 최대 (1-2^-N)/N
    conditional=False : 전체 (1 - (1-a)^(N+1)) / (N+1)     -> 높을수록 유리(나만 유효 포함)
    x 를 분포대로 무작위로 고르면 두 경우 모두 1/(N+2).
    """
    a = np.clip(np.asarray(f_x, dtype=float), 0.0, 1.0)
    n = np.asarray(n, dtype=float)
    if conditional:
        return (1.0 - (1.0 - a) ** (n + 1.0) - a ** (n + 1.0)) / np.maximum(n, 1.0)
    return (1.0 - (1.0 - a) ** (n + 1.0)) / (n + 1.0)


def best_rates(curve: pd.DataFrame, k: int = 5, min_separation: float = 0.02) -> pd.DataFrame:
    """낙찰확률 상위 k개 가정 사정율(서로 min_separation 이상 떨어진 값).

    contested_prob(다른 유효 투찰이 있을 때의 낙찰확률) 컬럼이 있으면 그것으로, 없으면 win_prob 로 정렬한다.
    """
    key = "contested_prob" if "contested_prob" in curve.columns else "win_prob"
    order = curve.sort_values(key, ascending=False)
    picked: list[int] = []
    for i, row in order.iterrows():
        if all(abs(row["assumed_rate"] - curve.at[j, "assumed_rate"]) >= min_separation for j in picked):
            picked.append(i)
        if len(picked) >= k:
            break
    return curve.loc[picked].reset_index(drop=True)
