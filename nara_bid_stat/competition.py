"""경쟁사 가정 사정율 분포(G) 추정 — 1순위 간격(gap) 모델.

아이디어
--------
개찰 결과에는 전체 업체 금액이 없어도 다음 세 값이 있다.
    y  : 실제 사정율
    x1 : 1순위(유효 최저가) 업체의 가정 사정율 = 1순위 투찰금액을 하한 산식으로 역산
    N  : 참여 업체 수
업체들의 가정 사정율이 분포 G 에서 독립적으로 나온다면 1순위는 'y 이상인 값 중 최솟값'이므로

    p(x1 | y, N) = N g(x1) (1 - (G(x1) - G(y)))^(N-1) / (1 - G(y)^N)

간격 x1 - y 가 평균보다 길면 그 근처에 경쟁사가 드물고, 짧으면 몰려 있다는 뜻이다.
이 우도를 공고 수천 건에 대해 최대화하면 G 를 추정할 수 있다.

모형
----
G 는 [-R, +R] 을 K개 칸으로 나눈 조각별 균등 밀도에 '상단 초과' 칸 [+R, +2R] 하나를 더한 것이다
(1순위가 예가변동폭 상단을 넘는 공고 = 그 위로 경쟁사가 없었다는 정보도 우도에 들어간다). 칸 질량은
    pi = softmax(log(기준질량) + h)
기준질량은 그 예가변동폭의 이론 사정율 분포(경쟁사가 '자연과 똑같이' 고른다는 귀무가설)이고,
h 는 그 대비 로그 배율이다. h 에는 인접 칸 차이 벌점(lam)과 사전값 쪽 수축 벌점(mu)을 준다.
데이터가 없는 구간은 h -> 0 (= 귀무가설)으로 수렴한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .mechanism import RateDistribution, equal_bins, split_bins


def default_base_distribution(band: float, n_sim: int = 300_000) -> RateDistribution:
    """예가변동폭별 기본 이론 분포: ±3 -> 음수8/양수7, 그 외 -> 15등분."""
    if abs(band - 3.0) < 1e-9:
        return split_bins(3.0, 8).theoretical(n_sim=n_sim, seed=11)
    return equal_bins(float(band)).theoretical(n_sim=n_sim, seed=11)


_BASE_CACHE: dict[float, RateDistribution] = {}


def _base_for(band: float) -> RateDistribution:
    key = round(float(band), 4)
    if key not in _BASE_CACHE:
        _BASE_CACHE[key] = default_base_distribution(key)
    return _BASE_CACHE[key]


#: '상단 초과' 칸의 기준 질량(경쟁사가 예가변동폭 상단보다 높은 사정율로 투찰할 사전 확률)
OVERFLOW_MASS = 1e-3


def _edges_and_base(band: float, bins: int, base: RateDistribution | None = None) -> tuple[np.ndarray, np.ndarray]:
    """칸 경계(정규 bins 개 + 상단 초과 1개)와 로그 기준질량."""
    base = base or _base_for(band)
    edges = np.r_[np.linspace(-band, band, bins + 1), 2.0 * band]
    bm = np.maximum(np.diff(base.cdf(edges[:-1])), 1e-6)
    bm = bm / bm.sum() * (1.0 - OVERFLOW_MASS)
    return edges, np.log(np.r_[bm, OVERFLOW_MASS])


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


_SHAPE_CACHE: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _shape_grid(band: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """기준(이론) 분포의 평활 누적분포를 0.001 간격 격자로: (칸 경계, 누적질량, 칸 밀도)."""
    key = round(float(band), 4)
    if key not in _SHAPE_CACHE:
        from .bid import rate_density_grid

        yc, wc, st = rate_density_grid(_base_for(key), 0.001, min_bw=0.01)
        xs = np.r_[yc - st / 2, yc[-1] + st / 2]
        _SHAPE_CACHE[key] = (xs, np.r_[0.0, np.cumsum(wc)], wc / st)
    return _SHAPE_CACHE[key]


class _BinShape:
    """칸 내부 모양. 정규 칸은 기준 분포의 모양을 따르고, 기준 질량이 없는 칸과 상단 초과 칸은 균등하다.

    칸 안을 균등 밀도로 두면 사정율 분포(매끈함)와의 비율이 칸마다 톱니처럼 변해, 낙찰확률 최댓값이
    칸 경계에 생기는 인공물이 생긴다. 칸 안 모양을 기준 분포와 같게 두면 귀무모형에서 그 비율이 상수가 된다.
    """

    def __init__(self, edges: np.ndarray, band: float | None):
        self.edges = np.asarray(edges, dtype=float)
        self.w = np.diff(self.edges)
        k = len(self.w)
        if band is None:
            self.use = np.zeros(k, dtype=bool)
            self.xs = self.cs = self.dens = None
            return
        self.xs, self.cs, self.dens = _shape_grid(band)
        fe = np.interp(self.edges, self.xs, self.cs)
        self.fe0 = fe[:-1]
        self.mass = np.diff(fe)
        self.use = self.mass > 1e-7
        self.use &= self.edges[1:] <= float(band) + 1e-9  # 상단 초과 칸은 균등

    def frac(self, x: np.ndarray, b: np.ndarray) -> np.ndarray:
        """칸 b 의 질량 중 x 아래 비율."""
        fu = (x - self.edges[b]) / self.w[b]
        if self.xs is None:
            return np.clip(fu, 0.0, 1.0)
        fs = (np.interp(x, self.xs, self.cs) - self.fe0[b]) / np.where(self.use[b], self.mass[b], 1.0)
        return np.clip(np.where(self.use[b], fs, fu), 0.0, 1.0)

    def density(self, x: np.ndarray, b: np.ndarray) -> np.ndarray:
        """칸 b 안에서의 정규화 밀도(칸 위에서 적분하면 1)."""
        u = 1.0 / self.w[b]
        if self.xs is None:
            return u
        i = np.clip(np.searchsorted(self.xs, x, side="right") - 1, 0, len(self.dens) - 1)
        return np.where(self.use[b], self.dens[i] / np.where(self.use[b], self.mass[b], 1.0), u)

    def bin_of(self, x: np.ndarray) -> np.ndarray:
        return np.clip(np.searchsorted(self.edges, x, side="right") - 1, 0, len(self.w) - 1)

    def coverage(self, v: np.ndarray) -> np.ndarray:
        """(n, K): 칸별 'v 아래 질량 비율'. 완전히 아래 1, 걸치면 비율, 위 0."""
        v = np.asarray(v, dtype=float)
        b = self.bin_of(v)
        out = (np.arange(len(self.w))[None, :] < b[:, None]).astype(float)
        out[np.arange(len(v)), b] = self.frac(v, b)
        out[v <= self.edges[0]] = 0.0
        out[v >= self.edges[-1]] = 1.0
        return out


@dataclass
class CompetitorModel:
    """경쟁사 가정 사정율 분포 G (조각별 균등 밀도)."""

    band: float
    edges: np.ndarray
    base_logmass: np.ndarray
    h: np.ndarray
    n_obs: int = 0
    label: str = ""
    loglik: float = float("nan")
    converged: bool = True
    #: 적합에 쓴 평활 강도. None 이면 귀무모형(경쟁사 = 이론 사정율 분포, h = 0)
    lam: float | None = None
    _pi: np.ndarray = field(init=False, repr=False)
    _cum: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.edges = np.asarray(self.edges, dtype=float)
        self.base_logmass = np.asarray(self.base_logmass, dtype=float)
        self.h = np.asarray(self.h, dtype=float)
        self._pi = _softmax(self.base_logmass + self.h)
        self._cum = np.r_[0.0, np.cumsum(self._pi)]
        self._shape = _BinShape(self.edges, self.band)

    @property
    def pi(self) -> np.ndarray:
        return self._pi

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.edges)

    def cdf_left(self, x) -> np.ndarray:
        """P(C < x). 칸 질량은 pi, 칸 안 모양은 기준 분포를 따른다(_BinShape)."""
        x = np.asarray(x, dtype=float)
        b = self._shape.bin_of(x)
        out = self._cum[b] + self._pi[b] * self._shape.frac(x, b)
        return np.where(x <= self.edges[0], 0.0, np.where(x >= self.edges[-1], 1.0, out))

    def pdf(self, x) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        b = self._shape.bin_of(x)
        inside = (x >= self.edges[0]) & (x < self.edges[-1])
        return np.where(inside, self._pi[b] * self._shape.density(x, b), 0.0)

    def density_ratio(self) -> np.ndarray:
        """칸별 '경쟁사 밀도 / 기준(자연) 밀도'. 1보다 작으면 경쟁사가 드문 구간."""
        base = _softmax(self.base_logmass)
        return self._pi / base

    def table(self) -> pd.DataFrame:
        base = _softmax(self.base_logmass)
        return pd.DataFrame({
            "lo": self.edges[:-1], "hi": self.edges[1:],
            "base_mass": base, "competitor_mass": self._pi,
            "ratio": self._pi / base,
        })

    # -- 관측 우도 ---------------------------------------------------------
    def loglik_terms(self, y, x1, n) -> np.ndarray:
        """관측별 로그우도(간격 모델)."""
        y = np.asarray(y, dtype=float)
        x1 = np.asarray(x1, dtype=float)
        n = np.asarray(n, dtype=float)
        gy = self.cdf_left(y)
        dg = np.clip(self.cdf_left(x1) - gy, 0.0, 1 - 1e-12)
        dens = np.maximum(self.pdf(x1), 1e-300)
        cond = -np.log1p(-np.exp(np.minimum(n * np.log(np.maximum(gy, 1e-300)), -1e-12)))
        return np.log(n) + np.log(dens) + (n - 1) * np.log1p(-dg) + cond

    def gap_pit(self, y, x1, n) -> np.ndarray:
        """P(1순위 <= x1 | y, N). 모형이 맞으면 U(0,1)."""
        y = np.asarray(y, dtype=float)
        n = np.asarray(n, dtype=float)
        gy = self.cdf_left(y)
        dg = np.clip(self.cdf_left(x1) - gy, 0.0, 1.0)
        exist = 1.0 - gy ** n
        return np.clip((1.0 - (1.0 - dg) ** n) / np.maximum(exist, 1e-300), 0.0, 1.0)


# ---------------------------------------------------------------------------
# 적합
# ---------------------------------------------------------------------------
def _coverage(edges: np.ndarray, v: np.ndarray) -> np.ndarray:
    """각 값 v 에 대해 칸별 '아래쪽 포함 비율' 행렬 (n, K): 완전히 아래 1, 걸치면 비율, 위 0."""
    lo, hi = edges[:-1], edges[1:]
    return np.clip((v[:, None] - lo[None, :]) / (hi - lo)[None, :], 0.0, 1.0)


def fit_competitor_model(
    y: Sequence[float],
    x1: Sequence[float],
    n: Sequence[float],
    band: float,
    *,
    base: RateDistribution | None = None,
    bins: int = 80,
    lam: float = 20.0,
    mu: float = 0.5,
    prior_h: np.ndarray | None = None,
    max_iter: int = 400,
    tol: float = 1e-7,
    label: str = "",
) -> CompetitorModel:
    """벌점 최대우도로 G 를 적합한다(L-BFGS)."""
    band = float(band)
    edges, a = _edges_and_base(band, bins, base)
    k = len(a)
    prior = np.zeros(k) if prior_h is None else np.asarray(prior_h, dtype=float)

    y = np.asarray(y, dtype=float)
    x1 = np.asarray(x1, dtype=float)
    n = np.asarray(n, dtype=float)
    ok = np.isfinite(y) & np.isfinite(x1) & np.isfinite(n) & (n >= 1) & (x1 > y)
    ok &= y >= edges[0]
    y, x1, n = y[ok], np.minimum(x1[ok], edges[-1] - 1e-9), n[ok]
    if len(y) == 0:
        return CompetitorModel(band, edges, a, prior.copy(), 0, label or f"band{band:g}", float("nan"))

    objective = _make_objective(edges, a, prior, y, x1, n, lam, mu, n_smooth=bins, shape=_BinShape(edges, band))
    h, fval, converged = _lbfgs(objective, prior.copy(), max_iter=max_iter, tol=tol)
    model = CompetitorModel(band, edges, a, h, int(len(y)), label or f"band{band:g}", float("nan"), converged, lam)
    model.loglik = float(np.sum(model.loglik_terms(y, x1, n)))
    return model



def _make_objective(edges: np.ndarray, a: np.ndarray, prior: np.ndarray, y: np.ndarray, x1: np.ndarray,
                    n: np.ndarray, lam: float, mu: float, n_smooth: int | None = None, shape: "_BinShape | None" = None):
    """음의 벌점 로그우도와 기울기를 돌려주는 함수(최소화용).

    인접 칸 차이 벌점은 앞쪽 n_smooth 개 칸(정규 칸)에만 준다(상단 초과 칸은 수축 벌점만).
    """
    bins = len(a)
    ns = bins if n_smooth is None else int(n_smooth)
    shape = shape or _BinShape(edges, None)
    cy = shape.coverage(y)
    dmat = shape.coverage(x1) - cy
    bx = shape.bin_of(x1)
    logw = -np.log(shape.density(x1, bx))  # 칸 안 밀도(상수항: h 와 무관)
    cnt_bx = np.bincount(bx, minlength=bins).astype(float)
    lg_n = np.log(n)

    def objective(h: np.ndarray) -> tuple[float, np.ndarray]:
        pi = _softmax(a + h)
        # 행렬곱 대신 곱셈합(멀티스레드 BLAS 동기화 비용 회피)
        dg = np.clip((dmat * pi).sum(axis=1), 0.0, 1 - 1e-12)
        gy = np.clip((cy * pi).sum(axis=1), 1e-300, 1 - 1e-12)
        gyn = np.exp(n * np.log(gy))
        ll = (np.sum(np.log(pi[bx]) - logw + lg_n + (n - 1) * np.log1p(-dg))
              - np.sum(np.log1p(-np.minimum(gyn, 1 - 1e-12))))
        g_pi = cnt_bx / pi
        g_pi -= (dmat * ((n - 1) / (1 - dg))[:, None]).sum(axis=0)
        g_pi += (cy * (n * gyn / gy / (1 - np.minimum(gyn, 1 - 1e-12)))[:, None]).sum(axis=0)
        g_h = pi * (g_pi - np.dot(pi, g_pi))
        dh = np.diff(h[:ns])
        pen = lam * np.sum(dh * dh) + mu * np.sum((h - prior) ** 2)
        g_pen = np.zeros_like(h)
        g_pen[:ns - 1] -= 2 * lam * dh
        g_pen[1:ns] += 2 * lam * dh
        g_pen += 2 * mu * (h - prior)
        return -(ll - pen), -(g_h - g_pen)  # 최소화 형태

    return objective

def _lbfgs(fun, x0: np.ndarray, *, max_iter: int = 400, tol: float = 1e-7, m: int = 10):
    """작은 문제용 L-BFGS(백트래킹 선탐색). 반환: (x, f, 수렴여부)."""
    x = x0.astype(float)
    f, g = fun(x)
    s_hist: list[np.ndarray] = []
    y_hist: list[np.ndarray] = []
    for _ in range(max_iter):
        q = g.copy()
        alphas = []
        for s, yv in zip(reversed(s_hist), reversed(y_hist)):
            rho = 1.0 / max(np.dot(yv, s), 1e-300)
            al = rho * np.dot(s, q)
            alphas.append((rho, al))
            q -= al * yv
        if y_hist:
            gamma = np.dot(s_hist[-1], y_hist[-1]) / max(np.dot(y_hist[-1], y_hist[-1]), 1e-300)
            q *= gamma
        else:
            q *= 1.0 / max(np.linalg.norm(g), 1.0)
        for (rho, al), s, yv in zip(reversed(alphas), s_hist, y_hist):
            be = rho * np.dot(yv, q)
            q += s * (al - be)
        d = -q
        if np.dot(d, g) >= 0:  # 하강 방향이 아니면 초기화
            d = -g / max(np.linalg.norm(g), 1.0)
            s_hist.clear()
            y_hist.clear()
        step = 1.0
        for _ls in range(40):
            xn = x + step * d
            fn, gn = fun(xn)
            if np.isfinite(fn) and fn <= f + 1e-4 * step * np.dot(g, d):
                break
            step *= 0.5
        else:
            return x, f, False
        s, yv = xn - x, gn - g
        if np.dot(s, yv) > 1e-12:
            s_hist.append(s)
            y_hist.append(yv)
            if len(s_hist) > m:
                s_hist.pop(0)
                y_hist.pop(0)
        converged = abs(f - fn) <= tol * max(1.0, abs(f)) and np.linalg.norm(gn, ord=np.inf) < 1e-3
        x, f, g = xn, fn, gn
        if converged:
            return x, f, True
    return x, f, False


# ---------------------------------------------------------------------------
# 예가변동폭 x 업종군 계층 모형
# ---------------------------------------------------------------------------
@dataclass
class CompetitorModelSet:
    """예가변동폭별 모형 + (예가변동폭, 업종군)별 모형(예가변동폭 모형 쪽으로 수축)."""

    by_band: dict[float, CompetitorModel]
    by_segment: dict[tuple[float, str], CompetitorModel]
    #: 예가변동폭별 lambda 선택 근거(검증 우도)
    selection: pd.DataFrame | None = None

    def get(self, band: float, industry_group: str | None = None) -> CompetitorModel | None:
        key = round(float(band), 4)
        if industry_group is not None and (key, industry_group) in self.by_segment:
            return self.by_segment[(key, industry_group)]
        return self.by_band.get(key)

    def summary(self) -> pd.DataFrame:
        rows = []
        items = [((k, "(전체)"), m) for k, m in sorted(self.by_band.items())] + sorted(self.by_segment.items())
        for (k, s), m in items:
            rows.append({"model": m.label, "band": k, "segment": s, "n_obs": m.n_obs, "converged": m.converged,
                         "lam": "귀무(경쟁사=자연분포)" if m.lam is None else m.lam,
                         "min_ratio": float(m.density_ratio()[:-1].min()), "max_ratio": float(m.density_ratio()[:-1].max()),
                         "above_band_mass": float(m.pi[-1])})
        out = pd.DataFrame(rows)
        if self.selection is not None and len(self.selection):
            out.attrs["lambda_selection"] = self.selection
        return out


class DistributionCompetitors:
    """경쟁사 유효 상한 사정율 분포 = 주어진 사정율 분포(효율적 시장의 정확한 귀무).

    귀무모형이 선택된 예가변동폭에서는 공고 자신의 사정율 분포를 그대로 경쟁사 분포로 쓴다.
    (예가변동폭 공통 이론분포를 쓰면 국방·한전처럼 규칙이 다른 기관에서 가짜 '빈 구간'이 생긴다.)
    """

    lam = None

    def __init__(self, dist: RateDistribution, step: float = 0.0005, min_bw: float = 0.01, label: str = "귀무(경쟁사=공고 사정율 분포)"):
        from .bid import rate_density_grid

        yc, wc, st = rate_density_grid(dist, step, min_bw=min_bw)
        self._x = np.r_[yc - st / 2, yc[-1] + st / 2]
        self._c = np.r_[0.0, np.cumsum(wc)]
        self.label = label
        self.n_obs = 0
        self.converged = True

    def cdf_left(self, x) -> np.ndarray:
        return np.interp(np.asarray(x, dtype=float), self._x, self._c, left=0.0, right=1.0)


#: 자동 선택 후보. None = 귀무모형(경쟁사 = 이론 사정율 분포)
LAMBDA_GRID: tuple = (2.0, 5.0, 20.0, 80.0, None)


def null_competitor_model(band: float, bins: int = 80, n_obs: int = 0) -> CompetitorModel:
    """귀무모형: 경쟁사 가정 사정율이 그 예가변동폭의 이론 사정율 분포와 같다(시장 효율)."""
    band = float(band)
    edges, a = _edges_and_base(band, bins)
    return CompetitorModel(band, edges, a, np.zeros(len(a)), n_obs, f"±{band:g}(귀무)", float("nan"), True, None)


def select_lambda(g: pd.DataFrame, band: float, *, grid=LAMBDA_GRID, holdout: float = 0.3, mu: float = 0.5,
                  bins: int = 80, min_valid: int = 60, default: float | None = None,
                  min_z: float = 2.0) -> tuple[float | None, pd.DataFrame]:
    """시간순으로 앞 70% 로 적합하고 뒤 30% 의 1순위 간격 로그우도로 lambda 를 고른다(귀무모형 포함).

    구조 있는 후보는 귀무모형 대비 '행별 로그우도 차이'의 평균이 표준오차의 min_z 배를 넘을 때만 채택한다
    (여러 후보 중 최댓값을 그냥 고르면 구조가 없는 시장에서도 절반쯤 '구조 있음'을 고르기 때문).
    낙찰 결과가 아니라 '간격 예측'이라는 모형 자체의 목표로 고르므로 전략 성과를 보고 고르는 과적합을 피한다.
    검증 표본이 min_valid 보다 적으면 default(기본: 귀무모형)를 쓴다.
    """
    g = g.sort_values("date", kind="mergesort")
    k = int(len(g) * (1 - holdout))
    tr, va = g.iloc[:k], g.iloc[k:]
    if len(va) < min_valid or len(tr) < min_valid:
        return default, pd.DataFrame([{"band": band, "lam": "null" if default is None else default,
                                       "note": f"검증 표본 부족({len(va)}건) -> 기본값"}])
    null = null_competitor_model(band, bins)
    ll0 = null.loglik_terms(va["rate"], np.minimum(va["winner_rate"], 2 * band - 1e-9), va["n_bidders"])
    rows = [{"band": band, "lam": "null", "n_train": len(tr), "n_valid": len(va),
             "valid_mean_loglik": float(np.mean(ll0)), "gain_vs_null": 0.0, "z_vs_null": 0.0}]
    best, best_z = None, -np.inf
    for lam in grid:
        if lam is None:
            continue
        m = fit_competitor_model(tr["rate"], tr["winner_rate"], tr["n_bidders"], band, lam=lam, mu=mu, bins=bins)
        ll = m.loglik_terms(va["rate"], np.minimum(va["winner_rate"], 2 * band - 1e-9), va["n_bidders"])
        d = ll - ll0
        se = float(np.std(d, ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.inf
        z = float(np.mean(d) / se) if se > 0 else 0.0
        rows.append({"band": band, "lam": lam, "n_train": len(tr), "n_valid": len(va),
                     "valid_mean_loglik": float(np.mean(ll)), "gain_vs_null": float(np.sum(d)), "z_vs_null": z})
        if z > best_z:
            best, best_z = lam, z
    t = pd.DataFrame(rows)
    chosen = best if best_z >= min_z else None
    t["chosen"] = [(r == "null" and chosen is None) or (r == chosen) for r in t["lam"]]
    return chosen, t


def fit_model_set(
    gaps: pd.DataFrame,
    *,
    lam: float | str | None = "auto",
    mu: float = 0.5,
    min_band_obs: int = 50,
    min_segment_obs: int = 150,
    bins: int = 80,
) -> CompetitorModelSet:
    """gaps: 컬럼 band, industry_group, rate, winner_rate, n_bidders, date (cbf.gap_sample_mask 통과 행).

    lam="auto" 이면 예가변동폭마다 select_lambda 로 고른다(귀무모형이 이기면 귀무모형 사용).
    """
    by_band: dict[float, CompetitorModel] = {}
    by_seg: dict[tuple[float, str], CompetitorModel] = {}
    sel = []
    for band, g in gaps.groupby(gaps["band"].round(4)):
        band = float(band)
        if len(g) < min_band_obs:
            continue
        if lam == "auto":
            use, t = select_lambda(g, band, mu=mu, bins=bins) if "date" in g else (20.0, pd.DataFrame())
            sel.append(t)
        else:
            use = lam
        if use is None:
            by_band[band] = null_competitor_model(band, bins, n_obs=len(g))
            continue
        mb = fit_competitor_model(g["rate"], g["winner_rate"], g["n_bidders"], band, lam=use, mu=mu, bins=bins,
                                  label=f"±{band:g}")
        by_band[band] = mb
        for seg, gs in g.groupby("industry_group"):
            if len(gs) < min_segment_obs:
                continue
            by_seg[(band, str(seg))] = fit_competitor_model(
                gs["rate"], gs["winner_rate"], gs["n_bidders"], band, lam=use, mu=mu * 4, bins=bins,
                prior_h=mb.h, label=f"±{band:g}/{seg}")
    selection = pd.concat([t for t in sel if len(t)], ignore_index=True) if any(len(t) for t in sel) else None
    return CompetitorModelSet(by_band, by_seg, selection)


def _local_ratio_mle(df_: np.ndarray, n: np.ndarray) -> float:
    """국소 배율 c 의 최대우도: sum[log c + (N-1) log(1 - c dF)] (dF = 자연분포 기준 간격 질량)."""
    df_ = np.asarray(df_, dtype=float)
    n = np.asarray(n, dtype=float)
    if len(df_) == 0 or df_.max() <= 0:
        return float("nan")
    hi = min(1.0 / df_.max() - 1e-9, 50.0)
    lo = 1e-3

    def score(c: float) -> float:
        return len(df_) / c - float(np.sum((n - 1) * df_ / (1 - c * df_)))

    if score(hi) > 0:
        return hi
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if score(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def gap_diagnostics(gaps: pd.DataFrame, models: CompetitorModelSet | None = None,
                    region_edges: Sequence[float] = (-1.01, -0.5, -0.25, -0.1, 0.0, 0.1, 0.25, 0.5, 1.01)) -> pd.DataFrame:
    """'경쟁사 = 자연 분포' 귀무가설 대비 지역(y/예가변동폭)별 진단.

    mean_pit_null : 귀무모형의 정확한 간격 PIT 평균(유한 N, 1순위 존재 조건 반영). 0.5 보다 작으면 경쟁사가 몰리고,
                    크면 드문 구간이다. 귀무가설이 맞으면 어느 지역이든 0.5.
    local_ratio   : 경쟁사 밀도 / 자연 밀도의 국소 최대우도 추정. 꼬리 잘림이 무시 가능한 행
                    (N x (1-F(y)) >= 10)만 쓴다(n_ratio).
    mean_pit_model: models 를 주면 그 모형의 PIT 평균. 같은 자료로 적합했다면 표본 내 값이다.
    """
    rows = []
    for band, g in gaps.groupby(gaps["band"].round(4)):
        band = float(band)
        null = null_competitor_model(band)
        base = _base_for(band)
        y = g["rate"].to_numpy(float)
        x1 = g["winner_rate"].to_numpy(float)
        n = g["n_bidders"].to_numpy(float)
        x1c = np.minimum(x1, 2 * band - 1e-9)
        pit_null = null.gap_pit(y, x1c, n)  # 상단 초과 1순위는 PIT ~ 1 (그 위로 경쟁사가 없었다)
        dF = base.cdf(x1) - base.cdf(y)
        tail_ok = n * (1 - base.cdf(y)) >= 10
        m = models.get(band) if models is not None else None
        pit_model = m.gap_pit(y, x1c, n) if m is not None else np.full(len(g), np.nan)
        reg = pd.cut(y / band, list(region_edges))
        d = pd.DataFrame({"region": reg, "pit_null": pit_null, "pit_model": pit_model, "dF": dF, "n": n, "ok": tail_ok, "x1": x1})
        for r, h in d.groupby("region", observed=True):
            hh = h[h["ok"]]
            rows.append({
                "band": band, "region(y/band)": str(r), "n": len(h), "n_winner_above_band": int((h["x1"] >= band).sum()),
                "mean_pit_null": float(np.nanmean(h["pit_null"])),
                "local_ratio": _local_ratio_mle(hh["dF"].to_numpy(), hh["n"].to_numpy()) if len(hh) >= 10 else np.nan,
                "n_ratio": int(len(hh)),
                "mean_pit_model": float(np.nanmean(h["pit_model"])) if m is not None else np.nan,
            })
    return pd.DataFrame(rows)
