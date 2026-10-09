"""예측 성능 평가: 적정 점수규칙(proper scoring rule)과 통계 검정.

scipy 없이 numpy/math 만 사용한다(Windows 오프라인 환경 고려).

핵심 원칙
--------
* '적중률' 하나만 보지 말고 분포 전체를 평가한다(CRPS, PIT, 구간 적중률).
* 모든 성능 수치는 '무기술 기준선(no-skill baseline)'과 같은 사례에서 비교한다.
* 표본이 작으면 신뢰구간이 넓다는 사실을 함께 보고한다.
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from .mechanism import RateDistribution


# ---------------------------------------------------------------------------
# 분포 예측 점수
# ---------------------------------------------------------------------------
def crps(dist: RateDistribution, y: float) -> float:
    """연속순위확률점수(CRPS). 낮을수록 좋다. 단위는 사정율(%p).

    가중 경험분포에 대한 정확식:
        CRPS = E|X - y| - 0.5 * E|X - X'|
    점분포(값 하나)이면 |x - y| 와 같다.
    """
    x, w = dist.values, dist.weights
    term1 = float(np.dot(w, np.abs(x - y)))
    cum = np.cumsum(w)
    # 정렬된 x 에 대해 E|X-X'| = 2 * sum_i w_i x_i (2F_i - w_i - 1)
    term2 = float(2.0 * np.dot(w * x, 2.0 * cum - w - 1.0))
    return term1 - 0.5 * term2


def pit(dist: RateDistribution, y: float) -> float:
    """확률적분변환(중간점 PIT). 잘 보정된 예측이면 PIT 는 U(0,1)."""
    below = float(dist.cdf(np.nextafter(y, -np.inf)))
    at_or_below = float(dist.cdf(y))
    return 0.5 * (below + at_or_below)


def interval_hit(dist: RateDistribution, y: float, level: float) -> bool:
    lo, hi = dist.interval(level)
    return bool(lo <= y <= hi)


def top_bucket_hit(dist: RateDistribution, y: float, k: int, width: float = 0.1) -> bool:
    """확률 상위 k개 부호구분 구간(BEST-k) 안에 실제값이 들어갔는지."""
    top = {r["bucket"] for r in dist.bucket_table(width)[:k]}
    return dist.bucket_of(y, width) in top


def candidates_bucket_hit(candidates: Sequence[float], y: float, width: float = 0.1) -> bool:
    """후보값 목록(예: 기존 엔진의 BEST10)이 실제값과 같은 부호구분 구간에 있는지."""
    def key(v: float) -> tuple[bool, int]:
        return (v < 0, int(math.floor(abs(v) / width + 1e-12)))

    ky = key(y)
    return any(key(float(c)) == ky for c in candidates if c is not None and np.isfinite(c))


def brier(p_positive: float, y: float) -> float:
    o = 1.0 if y > 0 else 0.0
    return (p_positive - o) ** 2


def log_loss(p_positive: float, y: float, eps: float = 1e-6) -> float:
    p = min(max(p_positive, eps), 1 - eps)
    return -math.log(p if y > 0 else 1 - p)


# ---------------------------------------------------------------------------
# 통계 검정 / 신뢰구간
# ---------------------------------------------------------------------------
def wilson_ci(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """이항비율의 Wilson 95% 신뢰구간."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, centre - half), min(1.0, centre + half))


def _log_binom_pmf(k: int, n: int, p: float) -> float:
    if p <= 0:
        return 0.0 if k == 0 else -math.inf
    if p >= 1:
        return 0.0 if k == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
            + k * math.log(p) + (n - k) * math.log1p(-p))


def binom_test_greater(k: int, n: int, p0: float) -> float:
    """단측 정확 이항검정 P(X >= k | n, p0). '기준선보다 높다'는 주장 검증용."""
    if n <= 0:
        return float("nan")
    logs = [_log_binom_pmf(i, n, p0) for i in range(k, n + 1)]
    m = max(logs)
    if m == -math.inf:
        return 0.0
    return float(min(1.0, math.exp(m) * sum(math.exp(v - m) for v in logs)))


def normal_sf(z: float) -> float:
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def paired_loss_test(loss_model: Sequence[float], loss_baseline: Sequence[float], max_lag: int | None = None) -> dict:
    """Diebold-Mariano 형 대응 손실 비교(HAC 표준오차).

    음수 mean_diff = 모델 손실이 더 작음(더 좋음).
    """
    a = np.asarray(loss_model, dtype=float)
    b = np.asarray(loss_baseline, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    d = a[ok] - b[ok]
    n = len(d)
    if n < 3:
        return {"n": n, "mean_diff": float("nan"), "se": float("nan"), "z": float("nan"), "p_two_sided": float("nan")}
    lag = int(max_lag if max_lag is not None else math.floor(n ** (1 / 3)))
    dc = d - d.mean()
    gamma0 = float(np.dot(dc, dc) / n)
    var = gamma0
    for k in range(1, min(lag, n - 1) + 1):
        gk = float(np.dot(dc[k:], dc[:-k]) / n)
        var += 2 * (1 - k / (lag + 1)) * gk
    se = math.sqrt(max(var, 1e-18) / n)
    z = float(d.mean() / se)
    return {"n": n, "mean_diff": float(d.mean()), "se": se, "z": z, "p_two_sided": 2 * normal_sf(abs(z))}


def ks_uniform(u: Sequence[float]) -> dict:
    """PIT 균등성 Kolmogorov-Smirnov 검정(점근 p값)."""
    x = np.sort(np.asarray(u, dtype=float))
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return {"n": 0, "D": float("nan"), "p": float("nan")}
    i = np.arange(1, n + 1)
    d = float(max(np.max(i / n - x), np.max(x - (i - 1) / n)))
    lam = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d
    p = 2 * sum((-1) ** (k - 1) * math.exp(-2 * k * k * lam * lam) for k in range(1, 101))
    return {"n": n, "D": d, "p": float(min(max(p, 0.0), 1.0))}


def pit_histogram(u: Sequence[float], bins: int = 10) -> list[float]:
    h, _ = np.histogram(np.asarray(u, dtype=float), bins=bins, range=(0, 1))
    return (h / max(1, h.sum())).tolist()
