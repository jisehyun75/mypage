"""명령행 사용법.

    python -m nara_bid_stat audit    --prebid-dir 복수예가 --out audit_out
    python -m nara_bid_stat forecast --prebid-dir 복수예가 --org "충청북도 청주시" [--asof 2026-10-01]
    python -m nara_bid_stat bid      --base 123456000 --lower-rate 87.745 --a-value 0 --rate -0.12
    python -m nara_bid_stat winprob  --prebid-dir 복수예가 --org "충청북도 청주시" --competitors comp.csv --n 120
    python -m nara_bid_stat nulltest --cases 60 --prior 0.44

    # CBF(입찰·낙찰 통합 DB) 연동
    python -m nara_bid_stat cbf-audit --cbf CBF.xlsx --out cbf_audit
    python -m nara_bid_stat backtest  --cbf CBF.xlsx --prebid-dir 복수예가 --start 2025-07-01 --out backtest_out
    python -m nara_bid_stat consult   --cbf CBF.xlsx --prebid-dir 복수예가 --backtest backtest_out --out consult_out
    python -m nara_bid_stat run-all   --cbf CBF.xlsx --prebid-dir 복수예가 --out 결과      # 위 셋을 한 번에

    # 공고 하나(또는 몇 개) 분석: BEST10 + 최종 추천 사정율·투찰금액·낙찰확률
    python -m nara_bid_stat analyze --cbf CBF.xlsx --prebid-dir 복수예가 --notice R26BK01740091-000
    python -m nara_bid_stat analyze --cbf CBF.xlsx --list                                  # 개찰 전 공고 목록
    python -m nara_bid_stat analyze --cbf CBF.xlsx --prebid-dir 복수예가 --notice 새공고 \
        --org "충청북도 청주시" --industry 전기 --base 312450000 --lower-rate 89.745 --a-value 12340000 \
        --band 3 --date "2026-10-12 11:00"                                            # CBF 에 없는 공고 직접 입력
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
                     net_cost_scales_with_rate=not args.net_cost_unscaled)
    print(json.dumps(bid_for_assumed_rate(args.base, args.rate, rule), ensure_ascii=False, indent=2))


def read_rate_csv(path, max_abs: float = 5.0) -> tuple[np.ndarray, int]:
    """경쟁사 사정율 CSV(컬럼 rate, 없으면 첫 컬럼) -> (사정율 배열, 제외한 행 수). UTF-8/cp949.

    숫자가 아니거나(소수점 쉼표 '0,5' 포함) |값| > max_abs 인 행(금액·% 단위 착오)은 제외하고 개수를 돌려준다.
    범위 밖 값을 그대로 쓰면 그 경쟁사는 결코 더 낮게 투찰하지 않으므로 낙찰확률이 크게 부풀려진다.
    """
    try:
        comp = pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:  # Excel 에서 저장한 한글 CSV
        comp = pd.read_csv(path, encoding="cp949")
    col = "rate" if "rate" in comp.columns else comp.columns[0]
    rates = pd.to_numeric(comp[col], errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(rates) & (np.abs(rates) <= max_abs)
    return rates[ok], int((~ok).sum())


def cmd_winprob(args) -> None:
    from .bid import best_rates, win_probability_curve
    from .data import P_COLS
    from .mechanism import RateDistribution

    h = _history(args)
    dist = RateDistribution.from_prebid_history(h[P_COLS].to_numpy(dtype=float), recent=args.recent)
    rates, n_bad = read_rate_csv(args.competitors)
    if not len(rates) and args.n > 0:
        raise SystemExit(f"경쟁사 CSV 에 숫자 사정율이 없습니다: {args.competitors}")
    curve = win_probability_curve(dist, rates, args.n)
    print(f"경쟁사 표본 {len(rates)}개(숫자가 아니거나 ±5 를 넘는 값 {n_bad}개 제외), 참여사 수 가정 {args.n}")
    print("(정렬 기준: contested_prob = 다른 유효 투찰이 있을 때의 낙찰확률. sole_prob 는 나만 유효한 경우로 검증 불가)")
    print(best_rates(curve, k=args.top).to_string(index=False))
    n = args.n
    print(f"\n참고(효율적 시장): 무작위 선택 1/(N+2) = {1 / (n + 2):.4f}, 1순위가 있는 공고에서 중앙값 (1-2^-N)/N = {(1 - 2.0 ** -n) / max(n, 1):.4f}")
    if args.out:
        curve.to_csv(args.out, index=False, encoding="utf-8-sig")


def cmd_nulltest(args) -> None:
    from .audit import null_tuning_simulation, strict_gate, utility_gate

    for name, gate, n in (("utility_gate(소표본 튜닝 방식)", utility_gate(), args.cases),
                          ("strict_gate(권장)", strict_gate(args.prior), max(args.cases, 600))):
        r = null_tuning_simulation(gate, n_cases=n, prior_positive=args.prior, trials=args.trials)
        print(f"{name:32s} 사례 {n:4d}건: 무정보 모델 통과율 {r['fire_rate']:.1%}, 통과 시 보고 정확도 {r['mean_reported_accuracy_when_fired']:.3f}")


def _nonneg_int(v: str) -> int:
    x = int(v)
    if x < 0:
        raise argparse.ArgumentTypeError("0 이상의 정수여야 합니다")
    return x


def _lam_arg(v: str):
    v = str(v).strip().lower()
    if v == "auto":
        return "auto"
    if v in ("null", "none"):
        return None
    x = float(v)
    if not np.isfinite(x) or x <= 0:
        raise argparse.ArgumentTypeError("--lam 은 auto, null 또는 양의 유한한 숫자여야 합니다")
    return x


def _load_cbf(path: str, sheet: str, cache: bool = True):
    from pathlib import Path

    from .cache import cached_frame
    from .cbf import load_cbf

    print(f"[load] CBF {path} (시트 {sheet}) ...", flush=True)
    if not Path(path).exists():
        sys.exit(f"CBF 파일을 찾을 수 없습니다: {path}")
    build = lambda: load_cbf(path, sheet=sheet)  # noqa: E731
    df = cached_frame("cbf", f"{Path(path).resolve()}|{sheet}", [path], build,
                      log=lambda m: print(m, flush=True)) if cache else build()
    print(f"[load] {len(df):,}건, 완료 {int(df['status'].eq('COMPLETED').sum()):,} / 개찰 전 {int(df['status'].eq('PENDING').sum())}", flush=True)
    return df


def _load_prebid_opt(path, cache: bool = True):
    if not path:
        return None
    from pathlib import Path

    from .cache import cached_frame
    from .data import load_prebid_folder

    folder = Path(path)
    if not folder.is_dir():
        sys.exit(f"복수예가 폴더를 찾을 수 없습니다: {path}")
    files = sorted(p for p in folder.glob("*.xlsx") if not p.name.startswith("~$"))
    build = lambda: load_prebid_folder(path)  # noqa: E731
    pb = cached_frame("prebid", str(folder.resolve()), files, build,
                      log=lambda m: print(m, flush=True)) if cache and files else build()
    print(f"[load] 복수예가 {len(pb):,}건 / 기관 {pb['org'].nunique()}곳", flush=True)
    return pb


def _save_tables(tables: dict, out: str) -> None:
    from pathlib import Path

    o = Path(out)
    o.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(o / f"{name}.csv", index=False, encoding="utf-8-sig")
    print(f"[ok] {o.resolve()}")


def cmd_cbf_audit(args, df=None) -> None:
    from .cbf import gap_sample_mask, quality_report, verify_floor_formula, verify_net_cost_rule
    from .competition import fit_model_set, gap_diagnostics

    df = _load_cbf(args.cbf, args.sheet, not args.no_cache) if df is None else df
    gaps = df[gap_sample_mask(df)]
    models = fit_model_set(gaps, lam=args.lam)
    tables = {
        "data_quality": quality_report(df),
        "floor_formula": verify_floor_formula(df),
        "net_cost_rule": verify_net_cost_rule(df),
        "competitor_models": models.summary(),
        "competitor_diagnostics": gap_diagnostics(gaps, models),
    }
    for band, m in sorted(models.by_band.items()):
        tables[f"competitor_density_{band:g}"] = m.table()
    if models.selection is not None and len(models.selection):
        tables["competitor_lambda_selection"] = models.selection
    for k in ("data_quality", "floor_formula", "net_cost_rule", "competitor_models", "competitor_diagnostics"):
        print(f"\n== {k}")
        print(tables[k].to_string(index=False))
    _save_tables(tables, args.out)


def cmd_backtest(args, df=None, pb=None):
    from .strategy import strategy_backtest, summarize_backtest

    df = _load_cbf(args.cbf, args.sheet, not args.no_cache) if df is None else df
    pb = _load_prebid_opt(args.prebid_dir, not args.no_cache) if pb is None else pb
    cases = strategy_backtest(df, pb, start=args.start, end=args.end, lam=args.lam, max_cases=args.max_cases,
                              progress=lambda m: print("  " + m, flush=True))
    summ = summarize_backtest(cases)
    tables = {"cases": cases, "summary": summ}
    if len(cases):
        for by in ("band", "industry_group"):
            tables[f"summary_by_{by}"] = summarize_backtest(cases, by=[by])
        cases = cases.assign(n_bucket=pd.cut(cases["n_bidders"], [0, 30, 100, 300, 1000, 1e9]).astype(str))
        tables["summary_by_n_bucket"] = summarize_backtest(cases, by=["n_bucket"])
    else:
        print("[warn] 평가 가능한 공고가 없습니다(기간·학습자료를 확인하십시오).")
    print(summ.T.to_string())
    _save_tables(tables, args.out)
    return summ


def cmd_consult(args, df=None, pb=None, bt=None) -> None:
    from pathlib import Path

    from .consult import consult, write_consult

    df = _load_cbf(args.cbf, args.sheet, not args.no_cache) if df is None else df
    pb = _load_prebid_opt(args.prebid_dir, not args.no_cache) if pb is None else pb
    if bt is None and args.backtest:
        p = Path(args.backtest)
        p = p / "summary.csv" if p.is_dir() else p
        bt = pd.read_csv(p, encoding="utf-8-sig")
    tables = consult(df, pb, notices=args.notice, lam=args.lam, backtest_summary=bt, n_boot=args.n_boot)
    paths = write_consult(tables, args.out)
    cols = [c for c in ("공고번호", "발주기관", "예상업체수(중앙값)", "기본추천_사정율", "기본추천_투찰금액", "기본추천_낙찰확률",
                        "대안_사정율", "대안_중앙값대비(보정)", "비고")
            if c in tables["요약"].columns]
    print(tables["요약"][cols].to_string(index=False))
    print(f"[ok] {paths['xlsx']}\n[ok] {paths['md']}")


_OVERRIDE_COLS = ("org", "industry", "base", "lower_rate", "a_value", "net_cost", "band", "date", "est_price")


def _safe_name(text: str) -> str:
    import re

    return re.sub(r'[\\/:*?"<>|\s]+', "_", text).strip("_")[:80] or "분석"


def cmd_analyze(args) -> None:
    """공고번호(또는 직접 입력한 공고)를 분석해 BEST10 과 최종 추천 사정율을 출력·저장한다."""
    from pathlib import Path

    from .cbf import industry_group, manual_notice_row, org_key, parse_band
    from .consult import consult, notice_cards, write_consult

    df = _load_cbf(args.cbf, args.sheet, not args.no_cache)
    if args.list:
        p = df[df["status"].eq("PENDING")].sort_values("open_dt", kind="mergesort")
        cols = ["notice", "org", "industry", "open_dt", "base", "lower_rate", "a_value", "band"]
        print(f"개찰 전 공고 {len(p)}건" + ("" if len(p) else " (CBF 에 PENDING 공고가 없습니다)"))
        if len(p):
            p = p.assign(**{c: p[c].map(lambda v: f"{v:,.0f}" if pd.notna(v) else "-") for c in ("base", "a_value")})
            print(p[cols].rename(columns={"notice": "공고번호", "org": "발주기관", "industry": "업종", "open_dt": "개찰일시",
                                          "base": "기초금액", "lower_rate": "하한율", "a_value": "A값", "band": "변동폭"}
                                 ).to_string(index=False))
        return
    manual = {k: getattr(args, k) for k in _OVERRIDE_COLS if getattr(args, k) is not None}
    notices = [n.strip() for n in (args.notice or []) if n.strip()]
    if not notices and not manual:
        if sys.stdin is not None and sys.stdin.isatty():
            notices = input("공고번호를 입력하세요(여러 개는 띄어쓰기로 구분): ").split()
        if not notices:
            sys.exit("공고번호(--notice) 또는 직접 입력 항목(--org --base --lower-rate --band --date)이 필요합니다. "
                     "개찰 전 공고 목록은 --list 로 볼 수 있습니다.")
    known = set(df["notice"].astype(str))
    if manual:
        if len(notices) > 1:
            sys.exit("직접 입력 항목(--org, --base 등)은 공고 하나에만 쓸 수 있습니다.")
        nid = notices[0] if notices else "직접입력"
        notices = [nid]
        if nid in known:  # CBF 에 있는 공고의 일부 값을 바꿔서 분석(예: 기초금액 공개 후)
            m = df["notice"].astype(str).eq(nid)
            df = df.copy()
            for k, v in manual.items():
                if k == "date":
                    odt = pd.to_datetime(v, errors="coerce")
                    if pd.isna(odt):
                        sys.exit(f"개찰일을 읽을 수 없습니다: {v}")
                    df.loc[m, "open_dt"] = odt
                    df.loc[m, "date"] = odt.normalize()
                elif k == "org":
                    df.loc[m, "org"] = v
                    df.loc[m, "org_key"] = org_key(v)
                elif k == "industry":
                    df.loc[m, "industry"] = v
                    df.loc[m, "industry_group"] = industry_group(v)
                else:
                    df.loc[m, k] = float(v)
            print(f"[입력] {nid}: CBF 값 중 {', '.join(manual)} 을(를) 입력값으로 바꿔 분석합니다.")
        else:
            need = [k for k in ("org", "base", "lower_rate", "band", "date") if k not in manual]
            if need:
                flags = ", ".join("--" + k.replace("_", "-") for k in need)
                sys.exit(f"CBF 에 없는 공고 '{nid}' 를 직접 입력하려면 {flags} 도 필요합니다.")
            try:
                row = manual_notice_row(df.columns, notice=nid, org=manual["org"], base=manual["base"],
                                        lower_rate=manual["lower_rate"], band=parse_band(manual["band"]),
                                        open_dt=manual["date"], a_value=manual.get("a_value"),
                                        net_cost=manual.get("net_cost"), industry=manual.get("industry", ""),
                                        est_price=manual.get("est_price"))
            except ValueError as e:
                sys.exit(str(e))
            df = pd.concat([df, row], ignore_index=True)
            print(f"[입력] {nid}: CBF 에 없는 공고를 직접 입력값으로 분석합니다.")
    else:
        missing = [n for n in notices if n not in known]
        if missing:
            import difflib

            for n in missing:
                near = difflib.get_close_matches(n, sorted(known), n=5, cutoff=0.75)
                hint = f" 비슷한 번호: {', '.join(near)}" if near else ""
                print(f"[없음] '{n}' 은(는) CBF 에 없습니다.{hint}")
            notices = [n for n in notices if n in known]
            if not notices:
                sys.exit("CBF 에 없는 공고는 --org --base --lower-rate --band --date (필요하면 --a-value --net-cost --industry)로 "
                         "직접 입력해 분석할 수 있습니다.")
    pb = _load_prebid_opt(args.prebid_dir, not args.no_cache)
    print(f"[분석] {', '.join(notices)} (대안 보정 부트스트랩 {args.n_boot}회) ...", flush=True)
    tables = consult(df, pb, notices=notices, lam=args.lam, n_boot=args.n_boot)
    for card in notice_cards(tables):
        print(card)
    name = "분석_" + (_safe_name("_".join(notices)) if len(notices) <= 3 else f"{_safe_name(notices[0])}_외{len(notices) - 1}건")
    try:
        paths = write_consult(tables, args.out, name=name)
    except PermissionError as e:  # 같은 이름의 결과 파일이 Excel 에서 열려 있음
        sys.exit(f"결과 파일을 저장하지 못했습니다({e.filename}). Excel 에서 열려 있으면 닫고 다시 실행하십시오. "
                 "(분석 결과는 위 화면 출력과 같습니다)")
    print("=" * 72)
    print(f"[저장] {Path(paths['xlsx']).resolve()}\n[저장] {Path(paths['md']).resolve()}")


def cmd_run_all(args) -> None:
    """CBF·복수예가를 한 번만 읽고 감사 -> (백테스트) -> 컨설팅 보고서를 순서대로 만든다."""
    import copy
    from pathlib import Path

    out = Path(args.out)
    df = _load_cbf(args.cbf, args.sheet, not args.no_cache)
    pb = _load_prebid_opt(args.prebid_dir, not args.no_cache)
    a = copy.copy(args)
    a.out = str(out / "1_cbf_audit")
    cmd_cbf_audit(a, df)
    bt = None
    if not args.skip_backtest:
        a = copy.copy(args)
        a.out, a.end, a.max_cases = str(out / "2_backtest"), None, args.max_cases
        bt = cmd_backtest(a, df, pb)
    a = copy.copy(args)
    a.out, a.notice, a.backtest = str(out / "3_consult"), args.notice, None
    cmd_consult(a, df, pb, bt)


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
    b.add_argument("--net-cost-unscaled", action="store_true",
                   help="순공사원가 98%%에 사정율을 곱하지 않음(기본은 곱함: 실데이터 1순위 27건 중 26건이 사정율 반영 기준 준수)")
    b.add_argument("--rounding", default="CEIL", choices=["CEIL", "HALF_UP", "FLOOR"])
    b.add_argument("--rate", type=float, required=True, help="가정 사정율(0기준 %%)")
    b.set_defaults(fn=cmd_bid)

    w = sub.add_parser("winprob", help="경쟁사 분포를 반영한 낙찰확률 최적 사정율")
    w.add_argument("--prebid-dir", required=True)
    w.add_argument("--org", required=True)
    w.add_argument("--asof", default=None)
    w.add_argument("--recent", type=int, default=60)
    w.add_argument("--competitors", required=True, help="경쟁사 가정 사정율 CSV(컬럼 rate)")
    w.add_argument("--n", type=_nonneg_int, required=True, help="예상 참여업체 수(본인 제외)")
    w.add_argument("--top", type=int, default=5)
    w.add_argument("--out", default=None)
    w.set_defaults(fn=cmd_winprob)

    n = sub.add_parser("nulltest", help="튜닝 절차의 우연 통과율")
    n.add_argument("--cases", type=int, default=60)
    n.add_argument("--prior", type=float, default=0.44)
    n.add_argument("--trials", type=int, default=3000)
    n.set_defaults(fn=cmd_nulltest)

    c = sub.add_parser("cbf-audit", help="CBF 데이터 품질·산식 검증·경쟁사 모형")
    c.add_argument("--cbf", required=True)
    c.add_argument("--sheet", default="통합데이터")
    c.add_argument("--lam", type=_lam_arg, default="auto", help="경쟁사 모형 평활 강도: auto(기본, 시간순 검증으로 선택) / null / 숫자")
    c.add_argument("--no-cache", action="store_true", help="저장된 표를 쓰지 않고 원본을 다시 읽음")
    c.add_argument("--out", default="cbf_audit")
    c.set_defaults(fn=cmd_cbf_audit)

    t = sub.add_parser("backtest", help="시간순 전략 백테스트(실제 1순위 금액과 비교)")
    t.add_argument("--cbf", required=True)
    t.add_argument("--sheet", default="통합데이터")
    t.add_argument("--prebid-dir", default=None)
    t.add_argument("--start", default="2025-07-01")
    t.add_argument("--end", default=None)
    t.add_argument("--lam", type=_lam_arg, default="auto", help="경쟁사 모형 평활 강도: auto(기본, 시간순 검증으로 선택) / null / 숫자")
    t.add_argument("--max-cases", type=int, default=None)
    t.add_argument("--no-cache", action="store_true", help="저장된 표를 쓰지 않고 원본을 다시 읽음")
    t.add_argument("--out", default="backtest_out")
    t.set_defaults(fn=cmd_backtest)

    k = sub.add_parser("consult", help="개찰 전 공고 컨설팅 보고서(Excel/Markdown)")
    k.add_argument("--cbf", required=True)
    k.add_argument("--sheet", default="통합데이터")
    k.add_argument("--prebid-dir", default=None)
    k.add_argument("--notice", nargs="*", default=None, help="공고번호(생략 시 개찰 전 전체)")
    k.add_argument("--backtest", default=None, help="backtest 출력 폴더 또는 summary.csv")
    k.add_argument("--lam", type=_lam_arg, default="auto", help="경쟁사 모형 평활 강도: auto(기본, 시간순 검증으로 선택) / null / 숫자")
    k.add_argument("--n-boot", type=_nonneg_int, default=30, help="대안(경쟁사 모형) 낙찰확률 선택 편향 보정 부트스트랩 횟수(0: 대안 생략)")
    k.add_argument("--no-cache", action="store_true", help="저장된 표를 쓰지 않고 원본을 다시 읽음")
    k.add_argument("--out", default="consult_out")
    k.set_defaults(fn=cmd_consult)

    an = sub.add_parser("analyze", help="공고 하나(또는 몇 개) 분석: BEST10 + 최종 추천 사정율·투찰금액·낙찰확률")
    an.add_argument("--cbf", required=True)
    an.add_argument("--sheet", default="통합데이터")
    an.add_argument("--prebid-dir", default=None)
    an.add_argument("--notice", nargs="*", default=None, help="공고번호(여러 개 가능). 생략하면 물어봄")
    an.add_argument("--list", action="store_true", help="CBF 의 개찰 전 공고 목록만 출력")
    an.add_argument("--org", default=None, help="[직접 입력] 발주기관명(복수예가 파일명과 같게)")
    an.add_argument("--industry", default=None, help="[직접 입력] 업종(예: 전기, 통신, 소방)")
    an.add_argument("--base", type=float, default=None, help="[직접 입력] 기초금액(원)")
    an.add_argument("--lower-rate", type=float, default=None, help="[직접 입력] 낙찰하한율(%%, 예: 89.745)")
    an.add_argument("--a-value", type=float, default=None, help="[직접 입력] A값(원, 없으면 0)")
    an.add_argument("--net-cost", type=float, default=None, help="[직접 입력] 순공사원가(원, 없으면 생략)")
    an.add_argument("--est-price", type=float, default=None, help="[직접 입력] 추정가격(원, 100억 이상이면 순공사원가 기준 미적용)")
    an.add_argument("--band", default=None, help="[직접 입력] 예가변동폭(3, 2, 2.5. '-3/+3' 형식은 --band=-3/+3 로)")
    an.add_argument("--date", default=None, help="[직접 입력] 개찰일시(예: 2026-10-12 또는 '2026-10-12 11:00')")
    an.add_argument("--lam", type=_lam_arg, default="auto", help="경쟁사 모형 평활 강도: auto(기본) / null / 숫자")
    an.add_argument("--n-boot", type=_nonneg_int, default=30, help="대안 낙찰확률 선택 편향 보정 부트스트랩 횟수(0: 대안 생략, 더 빠름)")
    an.add_argument("--no-cache", action="store_true", help="저장된 표를 쓰지 않고 원본을 다시 읽음")
    an.add_argument("--out", default="분석결과", help="결과 파일 폴더")
    an.set_defaults(fn=cmd_analyze)

    ra = sub.add_parser("run-all", help="감사 + 백테스트 + 컨설팅 보고서 한 번에")
    ra.add_argument("--cbf", required=True)
    ra.add_argument("--sheet", default="통합데이터")
    ra.add_argument("--prebid-dir", default=None)
    ra.add_argument("--notice", nargs="*", default=None)
    ra.add_argument("--start", default="2025-07-01")
    ra.add_argument("--lam", type=_lam_arg, default="auto", help="경쟁사 모형 평활 강도: auto(기본, 시간순 검증으로 선택) / null / 숫자")
    ra.add_argument("--max-cases", type=int, default=None)
    ra.add_argument("--skip-backtest", action="store_true", help="백테스트 생략(수 분 단축)")
    ra.add_argument("--n-boot", type=_nonneg_int, default=30, help="대안(경쟁사 모형) 낙찰확률 선택 편향 보정 부트스트랩 횟수(0: 대안 생략)")
    ra.add_argument("--no-cache", action="store_true", help="저장된 표를 쓰지 않고 원본을 다시 읽음")
    ra.add_argument("--out", default="nara_bid_stat_out")
    ra.set_defaults(fn=cmd_run_all)

    args = ap.parse_args(argv)
    np.set_printoptions(suppress=True)
    args.fn(args)


if __name__ == "__main__":
    main()
