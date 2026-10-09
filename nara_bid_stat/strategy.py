"""공고별 투찰 사정율 추천과 시간순 전략 백테스트.

구성
----
1. 사정율 분포 F (RateSourceResolver)
   복수예가 이력(같은 발주기관, 최근 60건 x 1365조합) -> (±3% 이면) 음수8/양수7 이론분포
   -> CBF 의 같은 기관 실제 사정율 이력(평활) -> 예가변동폭 이론분포
2. 경쟁사 분포 G (competition.CompetitorModelSet) : 1순위 간격 모델
3. 경쟁 업체 수 N : 비슷한 과거 공고의 업체수 표본(기관·업종 -> 업종·변동폭·금액대 -> 변동폭)
4. P(낙찰 | x) 를 계산해 x 를 고른다(bid.win_probability_curve)

백테스트는 실제 1순위 금액과 비교해 '내가 그 금액을 냈다면 1순위였는가'를 판정한다.
    내 금액 >= 실제 낙찰하한가(및 순공사원가 기준)  그리고  내 금액 < 실제 1순위 금액  ->  낙찰
모든 추정(F, G, N)은 대상 개찰일 이전 데이터만 쓴다.

추천의 목적함수는 '다른 유효 투찰이 있을 때의 낙찰확률'(contested)이다. 경쟁사가 모두 하한 미달이라
나만 유효해지는 경우(sole)는 개찰결과 데이터에 남지 않아 검증할 수 없으므로 별도로만 보고한다.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd

from . import scoring
from .bid import (
    AwardRule,
    bid_for_assumed_rate,
    lower_limit,
    net_cost_floor,
    null_win_prob,
    rate_density_grid,
    win_probability_curve,
)
from .cbf import eval_sample_mask, gap_sample_mask, org_key
from .competition import CompetitorModelSet, DistributionCompetitors, _base_for, fit_model_set, null_competitor_model
from .data import P_COLS
from .mechanism import KNOWN_SCHEMES, RateDistribution, identify_scheme


# ---------------------------------------------------------------------------
# 사정율 분포 F
# ---------------------------------------------------------------------------
class RateSourceResolver:
    """공고 한 건의 사정율 분포를 '대상 개찰일 이전' 데이터로 만든다."""

    def __init__(self, prebid: pd.DataFrame | None = None, cbf: pd.DataFrame | None = None, *,
                 recent: int = 60, min_history: int = 30, cbf_recent: int = 200, allow_parent: bool = False,
                 aliases: dict[str, str] | None = None, use_known_scheme: bool = True, scheme_share: float = 0.8):
        self.recent, self.min_history, self.cbf_recent, self.allow_parent = recent, min_history, cbf_recent, allow_parent
        self.use_known_scheme, self.scheme_share = use_known_scheme, scheme_share
        self._theory: dict[str, RateDistribution] = {}
        self.aliases = dict(aliases) if aliases is not None else (
            learn_org_aliases(cbf, prebid) if cbf is not None and prebid is not None else {})
        self._prebid: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        if prebid is not None and len(prebid):
            pb = prebid.assign(_k=prebid["org"].map(org_key)).sort_values("date", kind="mergesort")
            for k, g in pb.groupby("_k"):
                self._prebid[k] = (g["date"].to_numpy("datetime64[ns]"), g[P_COLS].to_numpy(float))
        self._cbf: dict[tuple[str, float], tuple[np.ndarray, np.ndarray]] = {}
        if cbf is not None and len(cbf):
            c = cbf[cbf["status"].eq("COMPLETED") & cbf["rate"].notna() & cbf["band"].notna()].sort_values("date", kind="mergesort")
            for (k, b), g in c.groupby(["org_key", c["band"].round(4)]):
                self._cbf[(k, float(b))] = (g["date"].to_numpy("datetime64[ns]"), g["rate"].to_numpy(float))

    def _prebid_key(self, key: str) -> tuple[str | None, str]:
        if key in self._prebid:
            return key, "복수예가(동일기관)"
        if self.aliases.get(key) in self._prebid:
            return self.aliases[key], "복수예가(동일공고로 확인한 별칭)"
        # CBF 는 '충청북도교육청 충청북도청주교육지원청', 복수예가 파일은 '청주교육지원청' 처럼 뒷부분만 쓰는 경우
        suff = [k for k in self._prebid if len(k) >= 5 and key.endswith(k)]
        if suff:
            return max(suff, key=len), "복수예가(기관명 뒷부분 일치)"
        if self.allow_parent:
            cands = [k for k in self._prebid if key.startswith(k)]
            if cands:
                return max(cands, key=len), "복수예가(상위기관)"
        return None, ""

    def distribution(self, org: str, band: float, asof) -> tuple[RateDistribution, str]:
        if pd.isna(asof):
            raise ValueError("기준일(개찰일)이 없습니다")
        asof = np.datetime64(pd.Timestamp(asof).normalize(), "ns")
        key = org_key(org)
        pk, label = self._prebid_key(key)
        if pk is not None:
            dates, x = self._prebid[pk]
            n = int(np.searchsorted(dates, asof, side="left"))
            if n >= self.min_history:
                hist = x[:n]
                if np.isfinite(band):  # 예가변동폭이 다른 과거 공고는 제외(규칙 변경 대비)
                    keep = np.abs(np.abs(hist).max(axis=1) - band) <= 0.26
                    if keep.sum() >= self.min_history:
                        hist = hist[keep]
                recent_hist = hist[-self.recent:]
                if self.use_known_scheme:
                    # 최근 이력이 알려진 생성 규칙 하나로 확인되면 그 규칙의 정확한 이론분포를 쓴다.
                    # (60건 부트스트랩의 표본 잡음을 최적화기가 '기회'로 착각하지 않도록)
                    names = pd.Series(identify_scheme(recent_hist)).value_counts()
                    top = str(names.index[0])
                    if top not in ("UNKNOWN", "AMBIGUOUS") and names.iloc[0] / len(recent_hist) >= self.scheme_share:
                        if top not in self._theory:
                            sch = next(sc for sc in KNOWN_SCHEMES if sc.name == top)
                            self._theory[top] = sch.theoretical(n_sim=300_000, seed=11)
                        return self._theory[top], f"{label} 규칙확인 {top} 이론분포(최근 {len(recent_hist)}건)"
                return RateDistribution.from_prebid_history(hist, recent=self.recent), f"{label} {len(recent_hist)}건"
        if np.isfinite(band) and abs(float(band) - 3.0) < 1e-9:
            # ±3% 는 실데이터 99.9% 가 음수8/양수7 규칙 -> 소표본 이력보다 이론분포가 정확
            return _base_for(3.0), "이론분포(±3, 음수8/양수7)"
        if np.isfinite(band) and (key, round(float(band), 4)) in self._cbf:
            dates, r = self._cbf[(key, round(float(band), 4))]
            n = int(np.searchsorted(dates, asof, side="left"))
            if n >= self.min_history:
                y = r[:n][-self.cbf_recent:]
                return RateDistribution.from_rate_history(y, source="cbf_history"), f"CBF 동일기관 사정율 {len(y)}건"
        if np.isfinite(band):
            return _base_for(float(band)), f"이론분포(±{band:g})"
        return _base_for(3.0), "이론분포(변동폭 미상, ±3 가정)"


def learn_org_aliases(cbf: pd.DataFrame, prebid: pd.DataFrame, *, min_matches: int = 3, min_share: float = 0.8) -> dict[str, str]:
    """같은 공고(기초금액·예정가격·개찰일 일치)를 이용해 CBF 기관명 -> 복수예가 파일 기관명 대응표를 만든다.

    기관 이름 표기만 맞추는 메타데이터이며 사정율 결과는 쓰지 않는다.
    """
    if cbf is None or prebid is None or not len(cbf) or not len(prebid):
        return {}
    c = cbf[cbf["expected"].notna() & cbf["base"].notna() & cbf["date"].notna()]
    c = pd.DataFrame({"ck": c["org_key"], "b": c["base"].round(0), "e": c["expected"].round(0), "d": c["date"].dt.normalize()})
    p = pd.DataFrame({"pk": prebid["org"].map(org_key), "b": prebid["base"].round(0), "e": prebid["expected"].round(0),
                      "d": pd.to_datetime(prebid["date"]).dt.normalize()})
    m = c.merge(p, on=["b", "e", "d"], how="inner")
    out: dict[str, str] = {}
    for ck, g in m.groupby("ck"):
        vc = g["pk"].value_counts()
        if vc.iloc[0] >= min_matches and vc.iloc[0] / len(g) >= min_share and vc.index[0] != ck:
            out[ck] = vc.index[0]
    return out


# ---------------------------------------------------------------------------
# 업체 수 N
# ---------------------------------------------------------------------------
class BidderCountSampler:
    """비슷한 과거 공고의 업체수 표본.

    가장 비슷한 수준(기관·업종 -> 업종·변동폭·금액대 -> 업종·변동폭 -> 변동폭)의 최근 k건에,
    업종·변동폭·금액대 수준의 최근 pool_k건을 pool_weight 비중으로 섞는다. 최근 30건만 쓰면 분포가 좁아
    업체수가 예상보다 훨씬 적은(낙찰확률이 큰) 공고를 놓친다. 실데이터 2,360건(2025-07 ~ 2026-09) 비교:
        예측 낙찰확률 5분위별 실제/예측  30건만 [2.04, 1.93, 1.20, 0.82, 1.09] -> 혼합 [1.34, 1.14, 1.05, 0.90, 1.21]
        Brier 13.38e-4 -> 13.28e-4, 실제 업체수의 PIT 상위 10% 비율 13.6% -> 11.4%
    같은 예가변동폭 전체를 섞으면 합계는 맞아 보여도 공고별 정확도가 나빠진다(Brier 14.4e-4).
    합계 기준으로는 여전히 약 11% 낮게 예측한다(업체 2~5곳 공고가 드물고 미리 알기 어려움). 백테스트 요약의
    n_sampler_vs_actual_ratio 로 확인한다.
    """

    def __init__(self, cbf: pd.DataFrame, *, k: int = 30, min_n: int = 10, pool_k: int = 200, pool_weight: float = 0.3):
        c = cbf[cbf["status"].eq("COMPLETED") & cbf["n_bidders"].ge(1) & cbf["date"].notna()].copy()
        c["_lb"] = np.log10(c["base"].clip(lower=1))
        self.c = c.sort_values("date", kind="mergesort").reset_index(drop=True)
        self.dates = self.c["date"].to_numpy("datetime64[ns]")
        self.k, self.min_n = k, min_n
        self.pool_k, self.pool_weight = pool_k, float(pool_weight)

    def sample(self, org: str, industry_group: str, band: float, base: float, asof) -> tuple[np.ndarray, str]:
        n = int(np.searchsorted(self.dates, np.datetime64(pd.Timestamp(asof).normalize(), "ns"), side="left"))
        past = self.c.iloc[:n]
        key = org_key(org)
        lb = math.log10(base) if base and base > 0 else np.nan
        same_band = past["band"].round(4) == round(band, 4)
        ind_band = (past["industry_group"] == industry_group) & same_band
        levels = [
            ("동일기관·업종", (past["org_key"] == key) & (past["industry_group"] == industry_group)),
            ("업종·변동폭·금액대", ind_band & ((past["_lb"] - lb).abs() <= 0.3)),
            ("업종·변동폭", ind_band),
            ("변동폭", same_band),
            ("전체", pd.Series(True, index=past.index)),
        ]
        local = None
        for label, m in levels:
            v = past.loc[m, "n_bidders"].to_numpy(float)
            if len(v) >= self.min_n:
                local, llabel, lname = v[-self.k:], f"{label} {min(len(v), self.k)}건", label
                break
        if local is None:
            v = past["n_bidders"].to_numpy(float)
            return (v[-self.k:] if len(v) else np.array([50.0])), "기본값"
        if self.pool_weight <= 0:
            return local, llabel
        for label, m in levels[1:4]:
            pool = past.loc[m, "n_bidders"].to_numpy(float)
            if len(pool) >= self.min_n:
                pool = pool[-self.pool_k:]
                # 표본 비중 = 지역 (1 - pool_weight) : 넓은 수준 pool_weight 가 되도록 지역 표본을 반복
                rep = max(1, int(round(len(pool) * (1.0 - self.pool_weight) / (self.pool_weight * len(local)))))
                wide = "같은 수준" if label == lname else label
                return np.r_[np.repeat(local, rep), pool], f"{llabel} + {wide} 최근 {len(pool)}건(비중 {self.pool_weight:.0%})"
        return local, llabel


# ---------------------------------------------------------------------------
# 추천
# ---------------------------------------------------------------------------
def _rule_for(row) -> AwardRule | None:
    lr = row.get("lower_rate")
    if lr is None or not np.isfinite(lr):
        return None
    a = row.get("a_value")
    nc = row.get("net_cost")
    est = row.get("est_price")
    nc_ok = nc is not None and np.isfinite(nc) and nc > 0 and not (est is not None and np.isfinite(est) and est >= 1e10)
    return AwardRule(float(lr), float(a) if a is not None and np.isfinite(a) else 0.0, float(nc) if nc_ok else None)


def _bid(base: float, x: float, rule: AwardRule) -> int:
    """가정 사정율 x 의 투찰금액 = max(하한가, 순공사원가 98% x (1+x)). 실제 사정율 <= x 이면 유효."""
    return bid_for_assumed_rate(base, x, rule)["bid"]


def competitors_for(model, dist: RateDistribution, band: float):
    """recommend 에 넘길 경쟁사 분포.

    귀무모형이면 이 공고의 사정율 분포(효율적 시장의 정확한 귀무). 구조 있는 모형은 그 예가변동폭의 표준 규칙
    (이론분포)을 기준으로 학습했으므로 공고의 사정율 분포가 바로 그 규칙일 때만 쓰고, 아니면 귀무로 바꾼다
    (국방·한전처럼 규칙이 다른 기관에 적용하면 분포가 어긋난 구간이 가짜 '빈 구간'으로 보인다).
    """
    if model is None or model.lam is None:
        return DistributionCompetitors(dist)
    if not np.isfinite(band) or dist.source != _base_for(float(band)).source:
        return DistributionCompetitors(dist, label="귀무(사정율 규칙이 모형 기준과 달라 경쟁사 모형 미적용)")
    return model


def efficient_market_probs(n_samples, f_x: float | None = None) -> dict:
    """효율적 시장(경쟁사 = 사정율 분포)에서 업체수 표본 평균: 중앙값 (1-2^-N)/N, 무작위 선택 1/(N+2),
    f_x(=F(x))를 주면 그 x 의 값("at")도."""
    ns = np.asarray(n_samples, dtype=float)
    ns = ns[np.isfinite(ns) & (ns >= 0)]
    out = {"median": float(np.mean(null_win_prob(0.5, ns))), "random": float(np.mean(1.0 / (ns + 2.0)))}
    if f_x is not None:
        out["at"] = float(np.mean(null_win_prob(f_x, ns)))
    return out


def alternative_candidate(candidates: pd.DataFrame, median_rate: float, min_distance: float = 0.05):
    """중앙값에서 min_distance 이상 떨어진 최상위 후보(없으면 None)."""
    alt = candidates[(candidates["assumed_rate"] - median_rate).abs() >= min_distance]
    return alt.iloc[0] if len(alt) else None


def selection_bias(row, dist: RateDistribution, model, refits: Sequence, n_samples, *, top_k: int = 3,
                   min_distance: float = 0.05) -> dict:
    """대안 추천값의 선택 편향(부트스트랩).

    추천은 적합된 곡선의 최댓값 위치를 고르고 같은 곡선으로 그 값을 보고하므로 낙찰확률이 위로 치우친다.
    부트스트랩 모형 m_b 로 같은 절차를 돌려 고른 x_b 에서 r_b = (m_b 로 본 값) / (원래 모형으로 본 값)을 구하면
    mean(r_b) 가 과대 배율의 추정치다. 보정값 = 보고값 / mean(r_b).
    (lambda 선택 단계에서 생기는 추가 편향은 포함하지 않는다.)
    """
    nvals = np.atleast_1d(np.asarray(n_samples, dtype=float))
    dens = rate_density_grid(dist, float(np.clip(0.02 / (np.nanmax(nvals) + 1.0), 2e-5, 2e-3)))
    ratios = []
    for mb in refits:
        rb = recommend(row, dist, mb, n_samples, top_k=top_k)
        a = alternative_candidate(rb["candidates"], rb["median_rate"], min_distance)
        if a is None:
            continue
        pm = float(win_probability_curve(dist, model, n_samples, grid=[float(a["assumed_rate"])], density=dens)["cond_prob"].iloc[0])
        if pm > 0:
            ratios.append(float(a["cond_prob"]) / pm)
    r = np.asarray(ratios, dtype=float)
    if not len(r):
        return {"factor": float("nan"), "n": 0, "lo": float("nan"), "hi": float("nan")}
    return {"factor": float(r.mean()), "n": int(len(r)), "lo": float(np.quantile(r, 0.1)), "hi": float(np.quantile(r, 0.9))}


def recommend(row: pd.Series | dict, dist: RateDistribution, competitors, n_samples: np.ndarray, *,
              top_k: int = 5, step: float = 0.02, refine: float = 0.002, min_separation: float = 0.05,
              n_y: int | None = None, tie_margin: float = 0.01) -> dict:
    """한 공고의 낙찰확률 곡선과 추천 사정율.

    목적함수: cond_prob = P(낙찰 | 경쟁사 1곳 이상 유효). 단독 유효(sole) 확률은 별도 보고.
    competitors 가 귀무모형(lam=None)이면 경쟁사 분포 = 이 공고의 사정율 분포로 바꿔 계산한다.
    반환: curve, candidates, best_rate, best_cond_prob, random_cond_prob, median_rate, median_cond_prob,
          p_contest, lift(best/random), n_expected
    """
    row = dict(row)
    base = row.get("base")
    rule = _rule_for(row)
    if getattr(competitors, "lam", "x") is None and not isinstance(competitors, DistributionCompetitors):
        # 귀무모형(효율적 시장)이 선택되면 경쟁사 분포 = 이 공고의 사정율 분포
        competitors = DistributionCompetitors(dist)
    nvals = np.atleast_1d(np.asarray(n_samples, dtype=float))
    step_fine = float(np.clip(0.02 / (np.nanmax(nvals) + 1.0), 2e-5, 2e-3))
    dens = rate_density_grid(dist, step_fine)
    lo, hi = dist.quantile([0.001, 0.999])
    grid = np.round(np.arange(np.floor(lo / step) * step, hi + step, step), 6)
    curve = win_probability_curve(dist, competitors, n_samples, grid=grid, density=dens)
    p_contest = curve.attrs["p_contest"]
    # 무작위 기준선: x 를 사정율 분포대로 뽑을 때
    cell = np.r_[grid[0] - step / 2, (grid[:-1] + grid[1:]) / 2, grid[-1] + step / 2]
    mass = np.diff(dist.smooth_cdf(cell))
    random_cp = float((curve["cond_prob"].to_numpy() * mass).sum() / max(mass.sum(), 1e-12))
    # 상위 후보 주변을 촘촘히 다시 계산(격자 해상도 보정일 뿐, 같은 곡선에서 고르고 같은 곡선으로 보고하므로
    # best_cond_prob 는 선택 편향만큼 위로 치우친다 -> selection_bias 로 보정)
    order = curve.sort_values("contested_prob", ascending=False)["assumed_rate"].to_numpy()
    seeds: list[float] = []
    for x in order:
        if all(abs(x - s0) >= min_separation for s0 in seeds):
            seeds.append(float(x))
        if len(seeds) >= top_k:
            break
    fine = np.unique(np.round(np.concatenate([np.arange(s0 - step, s0 + step + 1e-9, refine) for s0 in seeds]), 6))
    fc = win_probability_curve(dist, competitors, n_samples, grid=fine, density=dens)
    picks = []
    for s0 in seeds:
        w = fc[(fc["assumed_rate"] - s0).abs() <= step + 1e-9]
        picks.append(w.loc[w["contested_prob"].idxmax()])
    cand = pd.DataFrame(picks).sort_values("contested_prob", ascending=False).reset_index(drop=True)
    cand["rank"] = np.arange(1, len(cand) + 1)
    cand["lift_vs_random"] = cand["cond_prob"] / random_cp if random_cp > 0 else np.nan
    if rule is not None and base is not None and np.isfinite(base):
        info = [bid_for_assumed_rate(base, x, rule) for x in cand["assumed_rate"]]
        cand["bid_amount"] = [d["bid"] for d in info]
        cand["binding"] = [d["binding"] for d in info]
    med = float(dist.quantile(0.5))
    mc = win_probability_curve(dist, competitors, n_samples, grid=[med], density=dens)
    if float(mc["cond_prob"].iloc[0]) * (1.0 + tie_margin) >= float(cand["cond_prob"].iloc[0]):
        # 곡선이 사실상 평평하면(중앙값과 차이 tie_margin 이내) 임의의 최댓값 대신 중앙값을 1순위로
        mrow = mc.iloc[0].copy()
        mrow["rank"] = 0
        mrow["lift_vs_random"] = mrow["cond_prob"] / random_cp if random_cp > 0 else np.nan
        if "bid_amount" in cand.columns:
            d = bid_for_assumed_rate(base, med, rule)
            mrow["bid_amount"], mrow["binding"] = d["bid"], d["binding"]
        cand = pd.concat([pd.DataFrame([mrow]), cand], ignore_index=True)
        cand = cand[(cand.index == 0) | ((cand["assumed_rate"] - med).abs() >= min_separation)].reset_index(drop=True)
        cand = cand.head(top_k)
        cand["rank"] = np.arange(1, len(cand) + 1)
        cand["note"] = ["중앙값(곡선이 평평해 차이 1% 이내)"] + [""] * (len(cand) - 1)
    best = cand.iloc[0]
    return {
        "curve": curve, "candidates": cand, "p_contest": p_contest,
        "best_rate": float(best["assumed_rate"]), "best_cond_prob": float(best["cond_prob"]),
        "best_sole_prob": float(best["sole_prob"]),
        "random_cond_prob": random_cp, "median_rate": med, "median_cond_prob": float(mc["cond_prob"].iloc[0]),
        "lift": float(best["cond_prob"] / random_cp) if random_cp > 0 else np.nan,
        "n_expected": float(np.median(nvals)),
    }


# ---------------------------------------------------------------------------
# 시간순 백테스트
# ---------------------------------------------------------------------------
def _would_win(row, x: float) -> bool:
    """가정 사정율 x 의 금액으로 냈다면 실제 1순위보다 낮은 유효 투찰이었는가(금액 기준)."""
    rule = _rule_for(row)
    base, wb, y = row["base"], row["winner_bid"], row["rate"]
    if rule is None or not np.isfinite(base) or not np.isfinite(wb):
        return bool(row["rate"] <= x < row["winner_rate"])
    mine = _bid(base, x, rule)
    floor = row.get("floor")
    actual_floor = int(math.ceil(float(floor) - 1e-6)) if floor is not None and np.isfinite(floor) else lower_limit(base, y, rule)
    if mine < actual_floor:
        return False
    ncf = net_cost_floor(rule, y)
    if ncf is not None and mine < ncf:
        return False
    return mine < wb


def _band_strategy(curve: pd.DataFrame, y: float, x1: float, frac: float = 0.9) -> tuple[float, float, np.ndarray]:
    """'최적 근처' 전략: contested_prob >= frac x 최댓값 인 칸들 중에서 균등하게 x 를 고른다.

    모형이 맞으면 기대 낙찰확률이 최적의 frac 배 이상이므로, 실현값(연속)으로 모형이 고른 구간이
    실제로 유리했는지를 점 전략보다 작은 분산으로 판정할 수 있다.
    반환: (실현 낙찰확률, 모형 예측 조건부 낙찰확률, 선택된 x 들)
    """
    g = curve["assumed_rate"].to_numpy(float)
    c = curve["contested_prob"].to_numpy(float)
    if len(g) < 2 or c.max() <= 0:
        return float("nan"), float("nan"), np.array([])
    step = float(np.median(np.diff(g)))
    sel = c >= frac * c.max()
    lo, hi = g[sel] - step / 2, g[sel] + step / 2
    overlap = np.clip(np.minimum(hi, x1) - np.maximum(lo, y), 0, None) / step
    return float(overlap.mean()), float(curve["cond_prob"].to_numpy(float)[sel].mean()), g[sel]


def strategy_backtest(
    cbf: pd.DataFrame,
    prebid: pd.DataFrame | None = None,
    *,
    start: str = "2025-07-01",
    end: str | None = None,
    lam: float | str | None = "auto",
    mu: float = 0.5,
    n_k: int = 30,
    max_cases: int | None = None,
    seed: int = 0,
    progress: Callable[[str], None] | None = None,
    spy: Callable[[pd.Timestamp, pd.DataFrame], None] | None = None,
) -> pd.DataFrame:
    """월 단위로 경쟁사 모형을 다시 적합하며 공고마다 전략을 평가한다.

    평가 대상은 개찰 전에 알 수 있는 조건과 자료 일관성만으로 고른다(cbf.eval_sample_mask).
    전략
        model  : 조건부 낙찰확률 최대 x 한 점 (F, G, N 모두 과거 데이터)
        band90 : 조건부 낙찰확률이 최댓값의 90% 이상인 구간에서 균등하게 x 선택(실현값이 연속)
        median : 사정율 분포의 중앙값 (효율적 시장에서 1순위가 있는 공고 기준 최적)
        mode   : 사정율 분포 최빈 0.1 구간의 가운데
        random : x 를 사정율 분포대로 뽑을 때 실제로 이겼을 확률 = F(1순위) - F(실제)
    각 전략의 'null' 은 효율적 시장(경쟁사 = 사정율 분포)에서 그 x 의 기대 낙찰확률(1순위 존재 조건부)이다.
    경쟁사 모형을 학습할 간격 자료가 부족한 예가변동폭은 귀무모형으로, 공고의 사정율 분포가 모형 기준 규칙과
    다르면 그 공고의 분포를 경쟁사 분포로 써서(competitors_for) 평가 대상에서 빼지 않는다(lam 컬럼에 표시).
    model_pred 는 고른 x 에서 같은 모형으로 읽은 값이라 선택 편향만큼 위로 치우친다(model_calib_p 로 확인).
    """
    df = cbf.copy()
    target = df[eval_sample_mask(df)].copy()
    target = target[target["date"] >= pd.Timestamp(start)]
    if end is not None:
        target = target[target["date"] <= pd.Timestamp(end)]
    if max_cases is not None and len(target) > max_cases:
        target = target.sample(n=max_cases, random_state=seed)
    target = target.sort_values("open_dt", kind="mergesort")
    # 기관명 별칭은 평가 시작 이전 자료로만 학습(미래 공고 정보 미사용)
    t0 = pd.Timestamp(start)
    aliases = learn_org_aliases(df[df["date"] < t0], None if prebid is None else prebid[pd.to_datetime(prebid["date"]) < t0])
    resolver = RateSourceResolver(prebid, df, aliases=aliases)
    sampler = BidderCountSampler(df, k=n_k)
    gaps_all = df[gap_sample_mask(df)]
    rows = []
    for month, tm in target.groupby(target["date"].dt.to_period("M")):
        cutoff = month.start_time
        train = gaps_all[gaps_all["date"] < cutoff]
        if spy is not None:
            spy(cutoff, train)
        models = fit_model_set(train, lam=lam, mu=mu)
        if progress:
            progress(f"{month}: 학습 {len(train)}건, 평가 {len(tm)}건")
        for _, r in tm.iterrows():
            m = models.get(r["band"], r["industry_group"])
            no_model = m is None
            if no_model:  # 학습 간격 자료가 min_band_obs 미만인 예가변동폭: 귀무(효율적 시장)로 평가
                m = null_competitor_model(float(r["band"]))
            dist, src = resolver.distribution(r["org"], r["band"], r["date"])
            ns, nsrc = sampler.sample(r["org"], r["industry_group"], r["band"], r["base"], r["date"])
            comp = competitors_for(m, dist, float(r["band"]))
            rec = recommend(r, dist, comp, ns, top_k=3)
            y, x1 = float(r["rate"]), float(r["winner_rate"])
            band_real, band_pred, band_xs = _band_strategy(rec["curve"], y, x1)
            top_bucket = dist.bucket_table(0.1)[0]
            x_mode = (top_bucket["lo"] + top_bucket["hi"]) / 2
            n_act = float(r["n_bidders"])  # 평가용(실제 업체수). 의사결정에는 쓰지 않는다.
            fx = dist.smooth_cdf(np.array([rec["best_rate"], rec["median_rate"], x_mode]))
            nul = null_win_prob(fx, n_act)
            band_null = float(np.mean(null_win_prob(dist.smooth_cdf(band_xs), n_act))) if len(band_xs) else np.nan
            rows.append({
                "notice": r["notice"], "date": r["date"], "org": r["org"], "industry_group": r["industry_group"],
                "band": r["band"], "n_bidders": n_act, "n_expected": rec["n_expected"],
                "rate": y, "winner_rate": x1, "rate_source": src, "n_source": nsrc, "model": m.label,
                "lam": ("null(학습자료 부족)" if no_model else "null") if m.lam is None else (
                    m.lam if comp is m else "null(규칙 불일치)"),
                "model_rate": rec["best_rate"], "median_rate": rec["median_rate"], "mode_rate": x_mode,
                "model_pred": rec["best_cond_prob"], "median_pred": rec["median_cond_prob"],
                "random_pred": rec["random_cond_prob"], "model_sole_prob": rec["best_sole_prob"],
                "win_model": _would_win(r, rec["best_rate"]),
                "win_median": _would_win(r, rec["median_rate"]),
                "win_mode": _would_win(r, x_mode),
                "band90_realized": band_real, "band90_pred": band_pred, "band90_null": band_null,
                "random_realized": float(np.clip(np.diff(dist.smooth_cdf(np.array([y, x1])))[0], 0.0, 1.0)),
                "null_model": float(nul[0]), "null_median": float(nul[1]), "null_mode": float(nul[2]),
                "null_random": 1.0 / (n_act + 2.0), "null_best": float(null_win_prob(0.5, n_act)),
                # 같은 효율적 시장 공식을 '업체수 표본'으로 평균(예측에 쓰는 N 분포의 보정 점검용)
                "null_median_sampled_n": efficient_market_probs(ns)["median"],
            })
    return pd.DataFrame(rows)


def summarize_backtest(cases: pd.DataFrame, by: Sequence[str] = ()) -> pd.DataFrame:
    """전략별 낙찰 수와 비교 검정.

    *_vs_null : 관측 낙찰 수 vs 효율적 시장 기대(같은 x, 1순위 존재 조건부) — 정확 포아송-이항 양측검정.
                '경쟁사 구조를 이용한 우위'의 직접 검정이다.
    *_calib   : 관측 낙찰 수 vs 모형 예측확률 — 정확 포아송-이항 양측검정(예측이 과대/과소인지).
    *_lift_vs_random : 분포대로 무작위로 고를 때의 실현 기대값 대비 배수(설명용, 검정 아님).
    band90 은 실현값이 연속이라 정규근사(분산 sum p(1-p), 보수적)를 쓴다.
    detectable_lift : 무작위 기대값 기준 검정력 80%·양측 5% 로 검출 가능한 최소 배수(대략).
    """
    if cases is None or len(cases) == 0:
        return pd.DataFrame([{"n": 0, "note": "평가 가능한 공고가 없습니다(기간·학습자료 확인)"}])
    groups = [((), cases)] if not by else list(cases.groupby(list(by)))
    out = []
    for key, g in groups:
        n = len(g)
        base_exp = float(g["random_realized"].sum())
        rec = {"n": n, "n_bidders_median": float(g["n_bidders"].median()) if n else np.nan,
               "random_expected_wins": base_exp, "random_null_expected": float(g["null_random"].sum()),
               "efficient_market_best_wins": float(g["null_best"].sum())}
        if "null_median_sampled_n" in g:
            # 업체수 표본 보정: 같은 효율적 시장 공식을 표본 N 과 실제 N 으로 계산한 기대 낙찰 수
            rec["n_sampler_expected_median_wins"] = float(g["null_median_sampled_n"].sum())
            rec["n_sampler_vs_actual_ratio"] = (rec["n_sampler_expected_median_wins"] / float(g["null_median"].sum())
                                                if float(g["null_median"].sum()) > 0 else np.nan)
        if by:
            rec.update(dict(zip(by, key if isinstance(key, tuple) else (key,))))
        rec["detectable_lift"] = 1.0 + 2.8 / math.sqrt(base_exp) if base_exp > 0 else np.nan
        for s in ("model", "median", "mode"):
            w = g[f"win_{s}"].astype(bool)
            k = int(w.sum())
            lo, hi = scoring.wilson_ci(k, n)
            tn = scoring.poisson_binomial_test(k, g[f"null_{s}"].to_numpy(float))
            rec.update({f"{s}_wins": k, f"{s}_ci": f"[{lo:.4f},{hi:.4f}]",
                        f"{s}_lift_vs_random": k / base_exp if base_exp > 0 else np.nan,
                        f"{s}_null_expected": tn["expected"],
                        f"{s}_edge_vs_null": k / tn["expected"] if tn["expected"] > 0 else np.nan,
                        f"{s}_p_vs_null": tn["p_two_sided"]})
            if f"{s}_pred" in g:
                tc = scoring.poisson_binomial_test(k, g[f"{s}_pred"].to_numpy(float))
                rec.update({f"{s}_predicted_wins": tc["expected"], f"{s}_calib_p": tc["p_two_sided"]})
        if "band90_realized" in g:
            br = g["band90_realized"].astype(float)
            for ref, lab in (("band90_null", "null"), ("band90_pred", "calib")):
                pv = g[ref].astype(float)
                var = float((pv * (1 - pv)).sum())
                z = float((br.sum() - pv.sum()) / math.sqrt(var)) if var > 0 else np.nan
                rec[f"band90_p_vs_{lab}" if lab == "null" else "band90_calib_p"] = (
                    2 * scoring.normal_sf(abs(z)) if np.isfinite(z) else np.nan)
            rec.update({"band90_expected_wins": float(br.sum()), "band90_null_expected": float(g["band90_null"].sum()),
                        "band90_edge_vs_null": float(br.sum() / g["band90_null"].sum()) if g["band90_null"].sum() > 0 else np.nan,
                        "band90_predicted_wins": float(g["band90_pred"].sum())})
        from .walkforward import mcnemar_exact

        b = int((g["win_model"] & ~g["win_median"]).sum())
        c = int((~g["win_model"] & g["win_median"]).sum())
        rec["model_vs_median_mcnemar_p"] = mcnemar_exact(b, c)
        out.append(rec)
    return pd.DataFrame(out)
