"""nara_bid_stat — 나라장터 복수예가·사정율 통계 검증 도구.

기존 예측 엔진을 대체하기보다, 그 위에 '검증 가능한 기준선'과
'통일된 투찰 산식', '낙찰확률 최적화'를 제공하는 것을 목표로 한다.
"""
from .bid import AwardRule, best_rates, bid_for_assumed_rate, implied_rate, lower_limit, win_probability_curve
from .mechanism import KNOWN_SCHEMES, RateDistribution, Scheme, combo_means, equal_bins, identify_scheme, split_bins

__all__ = [
    "AwardRule",
    "KNOWN_SCHEMES",
    "RateDistribution",
    "Scheme",
    "best_rates",
    "bid_for_assumed_rate",
    "combo_means",
    "equal_bins",
    "identify_scheme",
    "implied_rate",
    "lower_limit",
    "split_bins",
    "win_probability_curve",
]

__version__ = "0.1.0"
