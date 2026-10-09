"""테스트용 합성 CBF(V95 형식 원본 컬럼).

* 사정율: 예가변동폭 규칙(±3 음수8/양수7, ±2 15등분)으로 15개 생성 후 무작위 4개 평균
* 경쟁사: 이론 사정율 분포에서 뽑되 x < sparse_below 구간은 sparse_keep 비율만 남김(경쟁사가 드문 구간)
* 1순위: 실제 사정율 이상인 경쟁사 중 최소, 금액은 하한 산식(절상)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nara_bid_stat.bid import AwardRule, bid_for_assumed_rate, lower_limit
from nara_bid_stat.mechanism import N_PICK, equal_bins, split_bins


def make_raw_cbf(n: int = 1200, *, seed: int = 0, sparse_below: float = -1.0, sparse_keep: float = 0.4,
                 n_range=(5, 60), n_pending: int = 3, net_cost_share: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    schemes = {3.0: split_bins(3.0, 8), 2.0: equal_bins(2.0)}
    banks = {}
    for b, sc in schemes.items():
        x = sc.sample(60_000, rng)
        pick = np.argsort(rng.random(x.shape), axis=1)[:, :N_PICK]
        bank = np.take_along_axis(x, pick, axis=1).mean(axis=1)
        keep = (bank >= sparse_below) | (rng.random(len(bank)) < sparse_keep)
        banks[b] = bank[keep]
    dates = pd.Timestamp("2025-01-02") + pd.to_timedelta(np.sort(rng.integers(0, 540, n + n_pending)), unit="D")
    rows = []
    for i in range(n + n_pending):
        band = 3.0 if rng.random() < 0.6 else 2.0
        org = "충청북도 가상시" if band == 3.0 else "조달청 가상지방지방조달청"
        sc = schemes[band]
        prices = sc.sample(1, rng)[0]
        y = float(np.mean(rng.choice(prices, N_PICK, replace=False)))
        base = float(rng.integers(50_000_000, 500_000_000))
        a_val = round(base * 0.03)
        net_cost = round(base * 0.9) if rng.random() < net_cost_share else None
        rule = AwardRule(87.745, a_val, net_cost)  # 순공사원가 98%(사정율 반영) 포함
        nb = int(rng.integers(*n_range))
        comp = rng.choice(banks[band], nb)  # 업체별 '유효 상한 사정율'
        valid = comp[comp >= y]
        pending = i >= n
        expected = base * (1 + y / 100)
        rec = {
            "공고번호": f"R26BK{i:08d}-000", "발주기관": org, "업종": "전기" if rng.random() < 0.7 else "통신",
            "지역": "충북", "추정가격": base / 1.1, "기초금액": base, "낙찰하한율": 87.745, "A값": a_val,
            "순공사원가": net_cost, "예가변동폭": f"-{band:g}/+{band:g}", "개찰일시": dates[i] + pd.Timedelta(hours=11),
            "개찰일": dates[i] + pd.Timedelta(hours=11), "낙찰자선정방법_표준": "QUALIFICATION_REVIEW",
            "낙찰하한율상태": "FIXED_LOWER_LIMIT", "quality_status": "READY",
        }
        if pending or len(valid) == 0:
            rec.update({"record_status": "PENDING" if pending else "BID_ONLY_REVIEW", "예정가격": None, "예가/기초(0%)": None,
                        "낙찰하한가": None, "1순위투찰금액": None, "1순위사정율(0%)": None, "업체수": None if pending else nb})
        else:
            xw = float(valid.min())
            wb = bid_for_assumed_rate(base, xw, rule)["bid"]
            floor = lower_limit(base, y, rule)
            rec.update({"record_status": "COMPLETED", "예정가격": expected, "예가/기초(0%)": round(y, 4),
                        "낙찰하한가": floor, "1순위투찰금액": wb, "1순위사정율(0%)": round(xw, 4), "업체수": nb})
        rows.append(rec)
    return pd.DataFrame(rows)
