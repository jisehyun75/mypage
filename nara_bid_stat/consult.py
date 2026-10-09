"""개찰 전 공고 컨설팅 보고서(Excel + Markdown).

공고마다
* 사정율 분포(출처, 평균, P(양수), 50%/80% 구간)
* 예상 경쟁 업체 수
* 경쟁사 모형 기반 낙찰확률 상위 후보(사정율, 투찰금액, 낙찰확률, 무작위 대비 배수)
를 계산하고 산식·데이터 검증과 백테스트 근거를 함께 싣는다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .cbf import gap_sample_mask, quality_report, verify_floor_formula, verify_net_cost_rule
from .competition import bootstrap_refits, fit_model_set, gap_diagnostics, null_competitor_model
from .bid import bid_for_assumed_rate
from .strategy import (
    BidderCountSampler,
    RateSourceResolver,
    _rule_for,
    alternative_candidate,
    competitors_for,
    efficient_market_probs,
    recommend,
    selection_bias,
)

GUIDE = [
    ("이 보고서의 성격", "사정율은 복수예가 추첨으로 정해지는 확률변수입니다. 아래 수치는 '정답'이 아니라 과거 데이터로 보정한 확률입니다."),
    ("사정율 분포", "같은 발주기관의 최근 복수예가 60건을 확인해, 알려진 생성 규칙(±3% 음수8/양수7, ±2% 15등분 등)이면 그 규칙의 정확한 이론분포, 국방·한전 같은 비표준 규칙이면 60건 x 1365조합 분포. 복수예가가 없으면 ±3%는 이론분포, 그 외는 CBF 동일기관 사정율(평활) 또는 예가변동폭 이론분포."),
    ("경쟁사 모형", "과거 개찰결과의 '1순위 가정 사정율 - 실제 사정율' 간격과 업체수로 추정한 경쟁사 분포. 개찰일 이전 데이터만 사용."),
    ("기본 추천 = 중앙값", "실데이터(2,360건 시간순 백테스트)에서 시장은 효율적이었고(무작위 선택 실제 39.0건 vs 이론 38.1건), 효율적 시장의 최적값인 사정율 분포 중앙값이 51건(효율적 시장 기대 44.8건)으로 가장 좋았음. 그래서 중앙값을 기본 추천으로 함."),
    ("대안(경쟁사 모형)", "±3% 기관에서 경쟁사가 드문 구간(낮은 사정율)을 노리는 값. 1순위 간격 예측에서는 귀무모형보다 낫지만, 낙찰 수로는 아직 확인되지 않은 가설(검출 가능한 최소 우위 약 1.45배). 공고의 사정율 분포가 모형 기준 규칙과 다르면(국방·한전 등) 제시하지 않음."),
    ("대안 낙찰확률(보정)", "모형 곡선의 최댓값을 고른 뒤 같은 모형으로 확률을 읽으면 위로 치우침(선택 편향). 학습 자료를 복원추출해 다시 적합한 모형들로 같은 절차를 반복해 과대 배율을 재고 그만큼 나눈 값. lambda 선택 단계의 편향은 빠져 있어 여전히 다소 낙관적일 수 있음. '모형내최댓값'은 보정 전 값."),
    ("낙찰확률", "그 사정율의 금액으로 투찰했을 때 1순위(유효 최저가)가 될 확률. 다른 유효 투찰이 있는 경우 기준(과거 개찰결과로 검증 가능한 부분). '효율적 시장' 열은 경쟁사가 사정율 분포대로 투찰한다는 가정(기본 추천의 근거), '경쟁사 모형' 열은 과거 1순위 간격으로 추정한 경쟁사 분포 기준. 대안은 경쟁사 모형이 맞을 때만 유리하고 효율적 시장이면 중앙값보다 불리함. 업체수를 개찰 전에 모르므로 비슷한 과거 공고(가장 비슷한 수준 최근 30건 + 업종·변동폭·금액대 최근 200건을 30% 비중)의 업체수 분포로 평균한 값이며, 수치의 불확실성 대부분이 업체수에서 옴. 백테스트(2,360건)에서 이 값의 합계는 실제 업체수로 계산한 값보다 약 11% 낮았음(40.0 vs 44.8건, 실제 중앙값 낙찰 51건). 차이는 대부분 업체가 예상보다 훨씬 적게(2~5곳) 들어온 소수 공고에서 생기므로 개별 수치는 하한 쪽 추정으로 보는 것이 안전함. 동률과 적격심사 비가격 점수 탈락은 반영하지 않음."),
    ("단독유효 확률", "경쟁사가 모두 하한 미달이라 나만 유효해지는 확률. 개찰결과 자료에 남지 않아 검증되지 않으므로 추천 기준에 넣지 않고 참고로만 표시."),
    ("무작위 대비 배수", "사정율 분포대로 아무 값이나 골랐을 때의 낙찰확률 대비 배수. 효율적 시장에서도 중앙값 부근이 무작위보다 기계적으로 유리하므로, 배수가 곧 '경쟁사 구조 우위'는 아님(백테스트 시트의 *_vs_null 참고)."),
    ("투찰금액", "max((예정가격-A값)x하한율+A값, 순공사원가x98%x(1+사정율)), 원 미만 절상. 실제 투찰 전 공고문 확인 필수."),
    ("검증 근거", "'백테스트' 시트: 과거 공고에 같은 절차를 적용해 실제 1순위 금액과 비교한 결과(월별 재학습, 미래 데이터 미사용)."),
]


def _fmt_interval(t: tuple[float, float]) -> str:
    return f"[{t[0]:+.3f}, {t[1]:+.3f}]"


def consult(
    cbf: pd.DataFrame,
    prebid: pd.DataFrame | None = None,
    *,
    notices: Iterable[str] | None = None,
    lam: float | str | None = "auto",
    mu: float = 0.5,
    top_k: int = 5,
    backtest_summary: pd.DataFrame | None = None,
    n_boot: int = 30,
) -> dict[str, pd.DataFrame]:
    """개찰 전 공고별 분석표들을 만든다(파일 저장은 write_consult).

    n_boot: 대안(경쟁사 모형) 낙찰확률의 선택 편향 보정용 부트스트랩 횟수(0 이면 보정하지 않고 대안도 내지 않음).
    """
    df = cbf
    if notices:
        want = {str(n).strip() for n in notices}
        targets = df[df["notice"].isin(want)]
    else:
        targets = df[df["status"].eq("PENDING")]
    targets = targets.sort_values("open_dt", kind="mergesort")
    resolver = RateSourceResolver(prebid, df)
    sampler = BidderCountSampler(df)
    gaps = df[gap_sample_mask(df)]
    summary, cands, curves = [], [], []
    model_cache: dict = {}
    boot_cache: dict = {}
    for _, r in targets.iterrows():
        day = r["date"]
        base_row = {"공고번호": r["notice"], "발주기관": r["org"], "업종": r["industry"], "개찰일시": r["open_dt"],
                    "기초금액": r["base"], "낙찰하한율": r["lower_rate"], "A값": r["a_value"], "순공사원가": r["net_cost"]}
        if pd.isna(day):
            summary.append({**base_row, "비고": "개찰일이 없어 분석 제외(기준일을 정할 수 없음)"})
            continue
        if day not in model_cache:
            model_cache[day] = fit_model_set(gaps[gaps["date"] < day], lam=lam, mu=mu)
        models = model_cache[day]
        band_known = bool(np.isfinite(r["band"]))
        band = float(r["band"]) if band_known else 3.0
        dist, src = resolver.distribution(r["org"], band if band_known else np.nan, day)
        s = dist.summary()
        row = {
            **base_row,
            "예가변동폭": f"±{band:g}" if band_known else "미상(±3 가정)", "사정율분포_출처": src,
            "사정율_평균": s["mean"], "P(양수)": s["prob_positive"],
            "50%구간": _fmt_interval(s["interval50"]), "80%구간": _fmt_interval(s["interval80"]),
        }
        lls = r.get("lower_limit_status")
        lls = "FIXED_LOWER_LIMIT" if lls is None or (isinstance(lls, float) and np.isnan(lls)) or str(lls) in ("", "nan", "None") else str(lls)
        fixed = lls == "FIXED_LOWER_LIMIT" and np.isfinite(r["lower_rate"])
        m = models.get(band, r["industry_group"])
        if not fixed:
            row["비고"] = "고정 하한율이 아니어서(종합심사·하한율 미상) 사정율 분포만 제공"
            summary.append(row)
            continue
        if not band_known:
            row["비고"] = "예가변동폭 미상 -> 사정율 분포만 제공(±3 가정 금액은 신뢰할 수 없음)"
            summary.append(row)
            continue
        notes = []
        if m is None:
            m = null_competitor_model(band)
            notes.append("이 예가변동폭은 경쟁사 모형 학습 자료가 부족해 효율적 시장(귀무)으로 계산")
        ns, nsrc = sampler.sample(r["org"], r["industry_group"], band, r["base"], day)
        comp = competitors_for(m, dist, band)
        structured = comp is m
        rec = recommend(r, dist, comp, ns, top_k=top_k)
        eff = efficient_market_probs(ns)
        lam_txt = "귀무(경쟁사=자연분포)" if m.lam is None else f"lambda={m.lam:g}"
        if m.lam is not None and not structured:
            lam_txt += ", 이 공고는 사정율 규칙이 모형 기준과 달라 귀무로 계산"
        row.update({"예상업체수(중앙값)": rec["n_expected"], "업체수_출처": nsrc,
                    "경쟁사모형": f"{m.label} (학습 {m.n_obs}건, {lam_txt})",
                    "무작위_낙찰확률": eff["random"]})
        c = rec["candidates"]
        rule = _rule_for(r)
        med = rec["median_rate"]
        base_ok = r["base"] is not None and pd.notna(r["base"]) and np.isfinite(r["base"]) and r["base"] > 0
        if rule is None or not base_ok:
            notes.append("기초금액·하한율이 없어 투찰금액 미제공")
        row.update({"기본추천_사정율": med,
                    "기본추천_투찰금액": bid_for_assumed_rate(r["base"], med, rule)["bid"] if rule is not None and base_ok else np.nan,
                    "기본추천_낙찰확률": eff["median"],
                    "기본추천_근거": "사정율 분포 중앙값 = 효율적 시장의 최적값(1순위가 있는 공고 기준). 백테스트로 확인된 기본값"})
        if structured:
            row.update({"경쟁사모형_중앙값_낙찰확률": rec["median_cond_prob"],
                        "경쟁사모형_무작위_낙찰확률": rec["random_cond_prob"]})
        a0 = alternative_candidate(c, med)
        if structured and a0 is not None and n_boot > 0:
            seg = r["industry_group"] if (round(band, 4), r["industry_group"]) in models.by_segment else None
            train = gaps[gaps["date"] < day]
            bkey = (len(train), round(band, 4), seg, m.lam)
            if bkey not in boot_cache:
                boot_cache[bkey] = bootstrap_refits(train, band, seg, lam=m.lam, mu=mu, n_boot=n_boot)
            sb = selection_bias(r, dist, comp, boot_cache[bkey], ns)
            raw = float(a0["cond_prob"])
            adj = raw / sb["factor"] if np.isfinite(sb["factor"]) and sb["factor"] > 0 else np.nan
            ratio = adj / rec["median_cond_prob"] if rec["median_cond_prob"] > 0 else np.nan
            if np.isfinite(ratio) and ratio >= 1.01:
                fx = float(dist.smooth_cdf(np.array([float(a0["assumed_rate"])]))[0])
                row.update({"대안_사정율": a0["assumed_rate"],
                            "대안_낙찰확률(효율적시장)": efficient_market_probs(ns, fx)["at"],
                            "대안_투찰금액": a0.get("bid_amount", np.nan) if base_ok else np.nan,
                            "대안_낙찰확률(보정)": adj, "대안_중앙값대비(보정)": ratio,
                            "대안_모형내최댓값": raw, "대안_선택편향배율": sb["factor"],
                            "대안_근거": "경쟁사 모형상 경쟁사가 드문 구간(선택 편향 보정, 경쟁사 모형 기준 중앙값 대비). "
                                       "시간순 백테스트로는 아직 우위가 확인되지 않은 가설"})
            else:
                notes.append(f"경쟁사 모형 대안은 선택 편향 보정 후 중앙값 대비 {ratio:.3f}배로 이점 없음")
        row["단독유효확률(참고)"] = float(c["sole_prob"].iloc[0]) if "sole_prob" in c else np.nan
        if not m.converged:
            notes.append("경쟁사 모형 적합이 수렴하지 않음 - 결과 주의")
        row["비고"] = "; ".join(notes)
        summary.append(row)
        c = c.assign(공고번호=r["notice"], 발주기관=r["org"],
                     확률기준="경쟁사모형(선택 편향 보정 전)" if structured else "효율적 시장(경쟁사 = 공고 사정율 분포)")
        cands.append(c)
        cv = rec["curve"].iloc[::2].assign(공고번호=r["notice"])
        curves.append(cv)
    tables: dict[str, pd.DataFrame] = {
        "요약": pd.DataFrame(summary),
        "추천후보": pd.concat(cands, ignore_index=True) if cands else pd.DataFrame(),
        "낙찰확률곡선": pd.concat(curves, ignore_index=True) if curves else pd.DataFrame(),
    }
    model_cache = {k: v for k, v in model_cache.items() if not pd.isna(k)}
    if model_cache:
        last = model_cache[max(model_cache)]
        tables["경쟁사모형"] = last.summary()
        mt = []
        for band, mm in sorted(last.by_band.items()):
            mt.append(mm.table().assign(model=mm.label))
        tables["경쟁사밀도표"] = pd.concat(mt, ignore_index=True) if mt else pd.DataFrame()
        tables["경쟁사진단"] = gap_diagnostics(gaps[gaps["date"] < max(model_cache)], last)
        if last.selection is not None and len(last.selection):
            tables["경쟁사모형_lambda선택"] = last.selection
    tables["산식검증_하한가"] = verify_floor_formula(df)
    tables["산식검증_순공사원가"] = verify_net_cost_rule(df)
    tables["데이터품질"] = quality_report(df)
    if backtest_summary is not None:
        tables["백테스트"] = backtest_summary
    tables["안내"] = pd.DataFrame(GUIDE, columns=["항목", "설명"])
    return tables


def write_consult(tables: dict[str, pd.DataFrame], out_dir: str | Path, name: str = "컨설팅_보고서") -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    xlsx = out / f"{name}.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as w:
        for sheet, t in tables.items():
            (t if len(t) else pd.DataFrame({"내용": ["(해당 없음)"]})).to_excel(w, sheet_name=sheet[:31], index=False)
    md = out / f"{name}.md"
    md.write_text(_markdown(tables), encoding="utf-8")
    return {"xlsx": xlsx, "md": md}


def _markdown(tables: dict[str, pd.DataFrame]) -> str:
    s = tables.get("요약", pd.DataFrame())
    lines = ["# 개찰 전 공고 컨설팅 보고서", ""]
    lines += [f"- {k}: {v}" for k, v in tables["안내"].itertuples(index=False)] + [""]

    def won(v):
        return f"{v:,.0f}원" if v is not None and pd.notna(v) else "-"

    def pct(v):
        return f"{v:.4%}" if v is not None and pd.notna(v) else "-"

    for _, r in s.iterrows():
        lines.append(f"## {r['공고번호']} · {r['발주기관']} · {r['업종']}")
        base = r.get("기초금액")
        lines.append(f"- 개찰 {r['개찰일시']}, 기초금액 {won(base)}, 하한율 {r['낙찰하한율']}%, 예가변동폭 {r.get('예가변동폭', '-')}")
        if pd.notna(r.get("사정율_평균", np.nan)):
            lines.append(f"- 사정율 분포: {r['사정율분포_출처']} / 평균 {r['사정율_평균']:+.3f} / P(양수) {r['P(양수)']:.3f} / 50% {r['50%구간']} / 80% {r['80%구간']}")
        if isinstance(r.get("비고"), str) and r.get("비고"):
            lines.append(f"- 비고: {r['비고']}")
        if pd.notna(r.get("기본추천_사정율", np.nan)):
            lines.append(f"- 예상 업체수 {r['예상업체수(중앙값)']:.0f} ({r['업체수_출처']}), 경쟁사 모형 {r['경쟁사모형']}")
            lines.append("")
            lines.append("| 구분 | 사정율 | 투찰금액 | 낙찰확률(효율적 시장) | 낙찰확률(경쟁사 모형) | 비고 |")
            lines.append("|---|---|---|---|---|---|")
            lines.append(f"| **기본 추천(중앙값)** | {r['기본추천_사정율']:+.3f} | {won(r['기본추천_투찰금액'])} | "
                         f"{pct(r['기본추천_낙찰확률'])} | {pct(r.get('경쟁사모형_중앙값_낙찰확률'))} | 효율적 시장 최적값 |")
            if pd.notna(r.get("대안_사정율", np.nan)):
                lines.append(f"| 대안(경쟁사 모형) | {r['대안_사정율']:+.3f} | {won(r['대안_투찰금액'])} | "
                             f"{pct(r['대안_낙찰확률(효율적시장)'])} | {pct(r['대안_낙찰확률(보정)'])} | "
                             f"경쟁사 모형 기준 중앙값의 {r['대안_중앙값대비(보정)']:.2f}배(선택 편향 보정, 보정 전 "
                             f"{pct(r['대안_모형내최댓값'])}). 효율적 시장이면 불리. 미검증 가설 |")
            lines.append(f"| 참고: 무작위 선택 | - | - | {pct(r['무작위_낙찰확률'])} | {pct(r.get('경쟁사모형_무작위_낙찰확률'))} | 분포대로 아무 값 |")
        lines.append("")
    bt = tables.get("백테스트")
    if bt is not None and len(bt):
        lines += ["## 백테스트 근거", "", bt.T.to_string(), ""]
    return "\n".join(lines)
