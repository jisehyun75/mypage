"""복수예가(15개 예비가격) 생성 구조와 사정율 예측 분포.

용어
----
* 사정율(0기준, %) : (예정가격 / 기초금액 - 1) * 100. 예: 99.87% -> -0.13
* 복수예가 15개     : 기초금액 대비 편차(%)로 표현한 15개 예비가격
* 예정가격          : 입찰자들이 가장 많이 고른 4개 예비가격의 산술평균

실데이터(충북권 66개 기관, 40,973건) 검증 결과 요약
------------------------------------------------
* 15개 예가는 구간을 나눠 칸마다 1개씩 균등난수로 뽑는다.
  - 지자체·교육청(±3%) : 음수 8칸(-3~0) + 양수 7칸(0~+3)
  - 조달청·공기업 일부(±2%, ±2.5%) : 15개 등간격 칸
* 4개 선택은 사실상 무작위다(1365조합 내 실제 예정가격 위치가 균등분포).
* 따라서 다음 공고의 사정율은 "생성 구조 + 무작위 4개 평균"으로 분포가 정해지며,
  최근 흐름·직전값으로는 더 맞힐 수 없다.

이 모듈은 그 분포를 직접 계산한다.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np

N_PRICES = 15
N_PICK = 4

#: 15개 중 4개를 고르는 모든 조합 (1365, 4)
COMBOS = np.array(list(itertools.combinations(range(N_PRICES), N_PICK)), dtype=np.int64)
N_COMBOS = len(COMBOS)

_COMBO_MEAN_MATRIX = np.zeros((N_COMBOS, N_PRICES))
_COMBO_MEAN_MATRIX[np.arange(N_COMBOS)[:, None], COMBOS] = 1.0 / N_PICK


def combo_means(prices: Sequence[float] | np.ndarray) -> np.ndarray:
    """15개 예가(%) -> 1365개 4개조합 평균(%).

    ``prices`` 가 (15,) 이면 (1365,), (n, 15) 이면 (n, 1365) 를 돌려준다.
    입력 순서와 무관하도록 내부에서 정렬한다.
    """
    x = np.sort(np.asarray(prices, dtype=float), axis=-1)
    if x.shape[-1] != N_PRICES:
        raise ValueError(f"복수예가는 {N_PRICES}개여야 합니다: shape={x.shape}")
    return x @ _COMBO_MEAN_MATRIX.T


# ---------------------------------------------------------------------------
# 생성 구조(scheme)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Scheme:
    """15개 칸 경계(16개)로 정의되는 복수예가 생성 규칙. 각 칸에서 균등난수 1개."""

    name: str
    edges: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.edges) != N_PRICES + 1:
            raise ValueError("edges 는 16개여야 합니다")
        if any(b <= a for a, b in zip(self.edges, self.edges[1:])):
            raise ValueError("edges 는 증가해야 합니다")

    @property
    def half_range(self) -> float:
        return max(abs(self.edges[0]), abs(self.edges[-1]))

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        e = np.asarray(self.edges)
        return e[:-1] + rng.random((n, N_PRICES)) * (e[1:] - e[:-1])

    def fits(self, prices: np.ndarray, tol: float = 6e-4) -> np.ndarray:
        """각 행(정렬된 15개)이 이 규칙의 칸 배치와 일치하는지."""
        x = np.sort(np.atleast_2d(np.asarray(prices, dtype=float)), axis=1)
        e = np.asarray(self.edges)
        return ((x >= e[:-1] - tol) & (x <= e[1:] + tol)).all(axis=1)

    def theoretical(self, n_sim: int = 200_000, seed: int = 0) -> "RateDistribution":
        """무작위 4개 선택을 가정한 이론 사정율 분포(몬테카를로)."""
        rng = np.random.default_rng(seed)
        x = self.sample(n_sim, rng)
        pick = np.argsort(rng.random((n_sim, N_PRICES)), axis=1)[:, :N_PICK]
        y = np.take_along_axis(x, pick, axis=1).mean(axis=1)
        return RateDistribution.from_samples(y, source=f"theory:{self.name}", n_history=0)


def equal_bins(half_range: float) -> Scheme:
    """-R ~ +R 를 15등분."""
    return Scheme(f"EQ15_{half_range:g}", tuple(np.linspace(-half_range, half_range, N_PRICES + 1)))


def split_bins(half_range: float, n_negative: int) -> Scheme:
    """-R~0 을 n_negative 칸, 0~+R 을 (15-n_negative) 칸으로 나눔."""
    neg = np.linspace(-half_range, 0.0, n_negative + 1)
    pos = np.linspace(0.0, half_range, N_PRICES - n_negative + 1)[1:]
    return Scheme(f"SPLIT{n_negative}_{N_PRICES - n_negative}_{half_range:g}", tuple(np.r_[neg, pos]))


#: 실데이터에서 확인된 규칙들
KNOWN_SCHEMES: tuple[Scheme, ...] = (
    equal_bins(2.0),
    equal_bins(2.5),
    equal_bins(3.0),
    split_bins(2.0, 8),
    split_bins(2.0, 7),
    split_bins(3.0, 8),
    split_bins(3.0, 7),
)


def identify_scheme(prices: np.ndarray, schemes: Iterable[Scheme] = KNOWN_SCHEMES, tol: float = 6e-4) -> np.ndarray:
    """각 행에 맞는 규칙 이름. 여러 개가 맞으면 'AMBIGUOUS', 없으면 'UNKNOWN'."""
    x = np.atleast_2d(np.asarray(prices, dtype=float))
    schemes = tuple(schemes)
    hits = np.stack([s.fits(x, tol) for s in schemes], axis=1)
    names = np.array([s.name for s in schemes], dtype=object)
    out = np.full(len(x), "UNKNOWN", dtype=object)
    one = hits.sum(axis=1) == 1
    out[one] = names[hits[one].argmax(axis=1)]
    out[hits.sum(axis=1) > 1] = "AMBIGUOUS"
    return out


def mechanism_signature(prices: np.ndarray) -> np.ndarray:
    """생성 구조 변경 감지용 간단 서명: '±범위/음수개수'. 예: '3.0/8'."""
    x = np.sort(np.atleast_2d(np.asarray(prices, dtype=float)), axis=1)
    half = np.round(np.abs(x).max(axis=1) * 2) / 2  # 0.5 단위
    neg = (x < 0).sum(axis=1)
    return np.array([f"{h:.1f}/{n}" for h, n in zip(half, neg)], dtype=object)


# ---------------------------------------------------------------------------
# 예측 분포
# ---------------------------------------------------------------------------
@dataclass
class RateDistribution:
    """사정율(0기준 %) 예측 분포. 가중 표본으로 표현한다."""

    values: np.ndarray
    weights: np.ndarray
    source: str = ""
    n_history: int = 0
    _cum: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        v = np.asarray(self.values, dtype=float).ravel()
        w = np.asarray(self.weights, dtype=float).ravel()
        if v.size == 0 or v.size != w.size:
            raise ValueError("values/weights 크기가 잘못되었습니다")
        if np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
            raise ValueError("weights 는 0 이상 유한값이어야 합니다")
        order = np.argsort(v, kind="mergesort")
        self.values = v[order]
        self.weights = w[order] / w.sum()
        self._cum = np.cumsum(self.weights)

    # -- 생성자 ------------------------------------------------------------
    @classmethod
    def from_samples(cls, samples: np.ndarray, weights: np.ndarray | None = None, *, source: str = "", n_history: int = 0) -> "RateDistribution":
        s = np.asarray(samples, dtype=float).ravel()
        w = np.ones_like(s) if weights is None else np.asarray(weights, dtype=float).ravel()
        return cls(s, w, source=source, n_history=n_history)

    @classmethod
    def from_prebid_history(cls, history: np.ndarray, recent: int | None = 60) -> "RateDistribution":
        """메커니즘 부트스트랩: 과거 공고들의 15개 예가 x 1365조합 평균의 혼합분포.

        생성 규칙을 몰라도(국방·한전 등 비표준 포함) 해당 기관의 최근 실제 예가로
        분포를 만들기 때문에 기관별 규칙 차이와 규칙 변경을 자동 반영한다.
        """
        h = np.atleast_2d(np.asarray(history, dtype=float))
        if h.shape[1] != N_PRICES or len(h) == 0:
            raise ValueError("history 는 (n, 15) 이어야 합니다")
        if recent:
            h = h[-int(recent):]
        return cls.from_samples(combo_means(h).ravel(), source=f"mechanism_bootstrap(recent={len(h)})", n_history=len(h))

    @classmethod
    def from_prices(cls, prices: Sequence[float]) -> "RateDistribution":
        """15개 예가가 공개된 뒤(개찰 후) 가능한 1365개 예정가격 분포."""
        return cls.from_samples(combo_means(prices), source="post_reveal_1365", n_history=1)

    # -- 질의 --------------------------------------------------------------
    def cdf(self, x: float | np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.values, np.asarray(x, dtype=float), side="right")
        return np.where(idx > 0, self._cum[np.maximum(idx - 1, 0)], 0.0)

    def quantile(self, q: float | np.ndarray) -> np.ndarray:
        q = np.clip(np.asarray(q, dtype=float), 0.0, 1.0)
        idx = np.searchsorted(self._cum, q, side="left")
        return self.values[np.minimum(idx, len(self.values) - 1)]

    def prob_between(self, lo: float, hi: float) -> float:
        """P(lo <= Y < hi)."""
        a = np.searchsorted(self.values, lo, side="left")
        b = np.searchsorted(self.values, hi, side="left")
        return float(self.weights[a:b].sum())

    @property
    def mean(self) -> float:
        return float(np.dot(self.values, self.weights))

    @property
    def std(self) -> float:
        m = self.mean
        return float(np.sqrt(np.dot((self.values - m) ** 2, self.weights)))

    @property
    def prob_positive(self) -> float:
        """P(Y > 0): '양수 부호' 확률."""
        return 1.0 - float(self.cdf(0.0))

    def interval(self, level: float = 0.8) -> tuple[float, float]:
        a = (1.0 - level) / 2.0
        lo, hi = self.quantile([a, 1.0 - a])
        return float(lo), float(hi)

    def bucket_table(self, width: float = 0.1) -> list[dict]:
        """부호를 구분한 width 폭 구간별 확률(내림차순).

        -0.0x 와 +0.0x 는 서로 다른 구간이다(부호구분 1d 버킷).
        """
        v = self.values
        k = np.floor(np.abs(v) / width + 1e-12).astype(np.int64)
        neg = v < 0
        key = np.where(neg, -(k + 1), k + 1)  # 0 을 피해서 부호 보존
        uniq, inv = np.unique(key, return_inverse=True)
        probs = np.bincount(inv, weights=self.weights)
        rows = []
        for kk, p in zip(uniq, probs):
            m = abs(int(kk)) - 1
            if kk < 0:
                lo, hi, label = -(m + 1) * width, -m * width, f"-{m * width:.{_dec(width)}f}x"
            else:
                lo, hi, label = m * width, (m + 1) * width, f"+{m * width:.{_dec(width)}f}x"
            rows.append({"bucket": label, "lo": round(lo, 10), "hi": round(hi, 10), "prob": float(p)})
        rows.sort(key=lambda r: -r["prob"])
        return rows

    def bucket_of(self, y: float, width: float = 0.1) -> str:
        m = int(np.floor(abs(y) / width + 1e-12))
        return (f"-{m * width:.{_dec(width)}f}x" if y < 0 else f"+{m * width:.{_dec(width)}f}x")

    def summary(self) -> dict:
        lo80, hi80 = self.interval(0.8)
        lo50, hi50 = self.interval(0.5)
        return {
            "source": self.source,
            "n_history": self.n_history,
            "mean": self.mean,
            "std": self.std,
            "prob_positive": self.prob_positive,
            "q50": float(self.quantile(0.5)),
            "interval50": (lo50, hi50),
            "interval80": (lo80, hi80),
        }


def _dec(width: float) -> int:
    s = f"{width:.6f}".rstrip("0")
    return max(1, len(s.split(".")[1]) if "." in s else 0)
