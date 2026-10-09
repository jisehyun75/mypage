"""명령행 사용법.

    python -m nara_bid_stat audit    --prebid-dir 복수예가 --out audit_out
    python -m nara_bid_stat forecast --prebid-dir 복수예가 --org "충청북도 청주시" [--asof 2026-10-01]
    python -m nara_bid_stat bid      --base 123456000 --lower-rate 87.745 --a-value 0 --rate -0.12
    python -m nara_bid_stat winprob  --prebid-dir 복수예가 --org "충청북도 청주시" --competitors comp.csv --n 120
    python -m nara_bid_stat nulltest --cases 60 --prior 0.44
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd


def _history(args):
    from .data import load_prebid_folder

    df = load_prebid_folder(args.prebid_dir)
    h = df[df["org"] == args.org]
    if h.empty:
        orgs = ", ".join(sorted(df["org"].unique())[:20])
        sys.exit(f"발주기관 '{args.org}' 이력이 없습니다. 예: {orgs} ...")
    if args.asof:
        h = h[h["date"] < pd.Timestamp(args.asof)]
    return h


def cmd_audit(args) -> None:
    from .audit import audit_prebid, write_audit
    from .data import load_prebid_folder

    df = load_prebid_folder(args.prebid_dir)
    print(f"[load] {len(df):,}건 / 기관 {df['org'].nunique()}곳 / {df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d}")
    tables = audit_prebid(df, oos_start=args.oos_start, max_cases=args.max_cases)
    p = write_audit(tables, args.out)
    print(f"[ok] {p}")
    for name in ("selection_randomness", "theory_vs_empirical", "walkforward_summary", "null_tuning"):
        print(f"\n== {name}")
        print(tables[name].to_string(index=False))


def cmd_forecast(args) -> None:
    from .data import P_COLS
    from .mechanism import RateDistribution, mechanism_signature

    h = _history(args)
    dist = RateDistribution.from_prebid_history(h[P_COLS].to_numpy(dtype=float), recent=args.recent)
    sig = pd.Series(mechanism_signature(h[P_COLS].to_numpy(dtype=float)[-args.recent:])).value_counts()
    s = dist.summary()
    print(f"발주기관: {args.org} | 사용 이력 {s['n_history']}건 | 생성구조 서명 {dict(sig)}")
    print(f"평균 {s['mean']:+.4f}  표준편차 {s['std']:.4f}  P(양수) {s['prob_positive']:.3f}")
    print(f"50% 구간 [{s['interval50'][0]:+.3f}, {s['interval50'][1]:+.3f}]  80% 구간 [{s['interval80'][0]:+.3f}, {s['interval80'][1]:+.3f}]")
    print(f"\n상위 {args.top}개 0.1 구간(부호 구분):")
    cum = 0.0
    for i, r in enumerate(dist.bucket_table(0.1)[: args.top], 1):
        cum += r["prob"]
        print(f"  {i:2d}. {r['bucket']:>6s} [{r['lo']:+.1f},{r['hi']:+.1f})  {r['prob']:.3f}  누적 {cum:.3f}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump({**s, "buckets": dist.bucket_table(0.1)}, f, ensure_ascii=False, indent=2, default=float)


def cmd_bid(args) -> None:
    from .bid import AwardRule, bid_for_assumed_rate

    rule = AwardRule(args.lower_rate, args.a_value, args.net_cost, rounding=args.rounding,
                     net_cost_scales_with_rate=args.net_cost_scaled)
    print(json.dumps(bid_for_assumed_rate(args.base, args.rate, rule), ensure_ascii=False, indent=2))


def cmd_winprob(args) -> None:
    from .bid import best_rates, win_probability_curve
    from .data import P_COLS
    from .mechanism import RateDistribution

    h = _history(args)
    dist = RateDistribution.from_prebid_history(h[P_COLS].to_numpy(dtype=float), recent=args.recent)
    comp = pd.read_csv(args.competitors)
    col = "rate" if "rate" in comp.columns else comp.columns[0]
    curve = win_probability_curve(dist, comp[col].to_numpy(dtype=float), args.n)
    print(f"경쟁사 표본 {len(comp)}개, 참여사 수 가정 {args.n}")
    print(best_rates(curve, k=args.top).to_string(index=False))
    print(f"\n참고: 무작위 선택 시 기대 낙찰확률 ~ 1/(N+1) = {1 / (args.n + 1):.4f}")
    if args.out:
        curve.to_csv(args.out, index=False, encoding="utf-8-sig")


def cmd_nulltest(args) -> None:
    from .audit import null_tuning_simulation, strict_gate, utility_gate

    for name, gate, n in (("utility_gate(소표본 튜닝 방식)", utility_gate(), args.cases),
                          ("strict_gate(권장)", strict_gate(args.prior), max(args.cases, 600))):
        r = null_tuning_simulation(gate, n_cases=n, prior_positive=args.prior, trials=args.trials)
        print(f"{name:32s} 사례 {n:4d}건: 무정보 모델 통과율 {r['fire_rate']:.1%}, 통과 시 보고 정확도 {r['mean_reported_accuracy_when_fired']:.3f}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="nara_bid_stat", description="나라장터 복수예가·사정율 통계 검증 도구")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("audit", help="복수예가 데이터 전체 감사")
    a.add_argument("--prebid-dir", required=True)
    a.add_argument("--out", default="audit_out")
    a.add_argument("--oos-start", default="2024-01-01")
    a.add_argument("--max-cases", type=int, default=4000)
    a.set_defaults(fn=cmd_audit)

    f = sub.add_parser("forecast", help="발주기관 사정율 예측 분포")
    f.add_argument("--prebid-dir", required=True)
    f.add_argument("--org", required=True)
    f.add_argument("--asof", default=None, help="이 날짜 이전 이력만 사용(YYYY-MM-DD)")
    f.add_argument("--recent", type=int, default=60)
    f.add_argument("--top", type=int, default=10)
    f.add_argument("--json", default=None)
    f.set_defaults(fn=cmd_forecast)

    b = sub.add_parser("bid", help="가정 사정율의 투찰금액")
    b.add_argument("--base", type=float, required=True, help="기초금액(원)")
    b.add_argument("--lower-rate", type=float, required=True, help="낙찰하한율(%%)")
    b.add_argument("--a-value", type=float, default=0.0)
    b.add_argument("--net-cost", type=float, default=None, help="순공사원가(원)")
    b.add_argument("--net-cost-scaled", action="store_true", help="순공사원가 98%%에 사정율을 곱함(공고·예규 확인 후)")
    b.add_argument("--rounding", default="CEIL", choices=["CEIL", "HALF_UP", "FLOOR"])
    b.add_argument("--rate", type=float, required=True, help="가정 사정율(0기준 %%)")
    b.set_defaults(fn=cmd_bid)

    w = sub.add_parser("winprob", help="경쟁사 분포를 반영한 낙찰확률 최적 사정율")
    w.add_argument("--prebid-dir", required=True)
    w.add_argument("--org", required=True)
    w.add_argument("--asof", default=None)
    w.add_argument("--recent", type=int, default=60)
    w.add_argument("--competitors", required=True, help="경쟁사 가정 사정율 CSV(컬럼 rate)")
    w.add_argument("--n", type=int, required=True, help="예상 참여업체 수(본인 제외)")
    w.add_argument("--top", type=int, default=5)
    w.add_argument("--out", default=None)
    w.set_defaults(fn=cmd_winprob)

    n = sub.add_parser("nulltest", help="튜닝 절차의 우연 통과율")
    n.add_argument("--cases", type=int, default=60)
    n.add_argument("--prior", type=float, default=0.44)
    n.add_argument("--trials", type=int, default=3000)
    n.set_defaults(fn=cmd_nulltest)

    args = ap.parse_args(argv)
    np.set_printoptions(suppress=True)
    args.fn(args)


if __name__ == "__main__":
    main()
