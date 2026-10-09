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
from .competition import fit_model_set, gap_diagnostics
from .bid import bid_for_assumed_rate
from .strategy import BidderCountSampler, RateSourceResolver, _rule_for, recommend

GUIDE = [
    ("이 보고서의 성격", "사정율은 복수예가 추첨으로 정해지는 확률변수입니다. 아래 수치는 '정답'이 아니라 과거 데이터로 보정한 확률입니다."),
    ("사정율 분포", "같은 발주기관의 최근 복수예가 60건을 확인해, 알려진 생성 규칙(±3% 음수8/양수7, ±2% 15등분 등)이면 그 규칙의 정확한 이론분포, 국방·한전 같은 비표준 규칙이면 60건 x 1365조합 분포. 복수예가가 없으면 ±3%는 이론분포, 그 외는 CBF 동일기관 사정율(평활) 또는 예가변동폭 이론분포."),
    ("경쟁사 모형", "과거 개찰결과의 '1순위 가정 사정율 - 실제 사정율' 간격과 업체수로 추정한 경쟁사 분포. 개찰일 이전 데이터만 사용."),
    ("기본 추천 = 중앙값", "실데이터(2,326건 시간순 백테스트)에서 시장은 효율적이었고(무작위 선택 실제 38.9건 vs 이론 38.0건), 효율적 시장의 최적값인 사정율 분포 중앙값이 51건(효율적 시장 기대 44.6건)으로 가장 좋았음. 그래서 중앙값을 기본 추천으로 함."),
    ("대안(경쟁사 모형)", "±3% 기관에서 경쟁사가 드문 구간(낮은 사정율)을 노리는 값. 1순위 간격 예측에서는 귀무모형보다 낫지만, 낙찰 수로는 아직 확인되지 않은 가설(검출 가능한 최소 우위 약 1.45배)."),
    ("낙찰확률", "그 사정율의 금액으로 투찰했을 때 1순위(유효 최저가)가 될 확률. 다른 유효 투찰이 있는 경우 기준(과거 개찰결과로 검증 가능한 부분). 업체수를 개찰 전에 모르므로 비슷한 과거 공고의 업체수 분포로 평균한 값이며, 수치의 불확실성 대부분이 업체수에서 옴. 동률과 적격심사 비가격 점수 탈락은 반영하지 않음."),
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
) -> dict[str, pd.DataFrame]:
    """개찰 전 공고별 분석표들을 만든다(파일 저장은 write_consult)."""
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
        if m is None:
            row["비고"] = "경쟁사 모형 학습 데이터 부족"
            summary.append(row)
            continue
        if not band_known:
            row["비고"] = "예가변동폭 미상 -> 사정율 분포만 제공(±3 가정 금액은 신뢰할 수 없음)"
            summary.append(row)
            continue
        ns, nsrc = sampler.sample(r["org"], r["industry_group"], band, r["base"], day)
        rec = recommend(r, dist, m, ns, top_k=top_k)
        lam_txt = "귀무(경쟁사=자연분포)" if m.lam is None else f"lambda={m.lam:g}"
        row.update({"예상업체수(중앙값)": rec["n_expected"], "업체수_출처": nsrc,
                    "경쟁사모형": f"{m.label} (학습 {m.n_obs}건, {lam_txt})",
                    "무작위_낙찰확률": rec["random_cond_prob"], "중앙값_사정율": rec["median_rate"],
                    "중앙값_낙찰확률": rec["median_cond_prob"]})
        c = rec["candidates"]
        rule = _rule_for(r)
        med = rec["median_rate"]
        row.update({"기본추천_사정율": med,
                    "기본추천_투찰금액": bid_for_assumed_rate(r["base"], med, rule)["bid"] if rule is not None else np.nan,
                    "기본추천_낙찰확률": rec["median_cond_prob"],
                    "기본추천_근거": "사정율 분포 중앙값 = 효율적 시장의 최적값(1순위가 있는 공고 기준). 백테스트로 확인된 기본값"})
        alt = c[(c["assumed_rate"] - med).abs() >= 0.05]
        if m.lam is not None and len(alt) and float(alt["cond_prob"].iloc[0]) >= 1.01 * rec["median_cond_prob"]:
            a0 = alt.iloc[0]
            row.update({"대안_사정율": a0["assumed_rate"], "대안_투찰금액": a0.get("bid_amount", np.nan),
                        "대안_낙찰확률": a0["cond_prob"], "대안_중앙값대비": a0["cond_prob"] / rec["median_cond_prob"],
                        "대안_근거": "경쟁사 모형상 경쟁사가 드문 구간. 시간순 백테스트로는 아직 우위가 확인되지 않은 가설"})
        row["단독유효확률(참고)"] = float(c["sole_prob"].iloc[0]) if "sole_prob" in c else np.nan
        row["비고"] = "" if m.converged else "경쟁사 모형 적합이 수렴하지 않음 - 결과 주의"
        summary.append(row)
        c = c.assign(공고번호=r["notice"], 발주기관=r["org"])
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
            lines.append("| 구분 | 사정율 | 투찰금액 | 낙찰확률 | 비고 |")
            lines.append("|---|---|---|---|---|")
            lines.append(f"| **기본 추천(중앙값)** | {r['기본추천_사정율']:+.3f} | {won(r['기본추천_투찰금액'])} | {r['기본추천_낙찰확률']:.4%} | 효율적 시장 최적값 |")
            if pd.notna(r.get("대안_사정율", np.nan)):
                lines.append(f"| 대안(경쟁사 모형) | {r['대안_사정율']:+.3f} | {won(r['대안_투찰금액'])} | {r['대안_낙찰확률']:.4%} | 중앙값 대비 {r['대안_중앙값대비']:.2f}배, 미검증 가설 |")
            lines.append(f"| 참고: 무작위 선택 | - | - | {r['무작위_낙찰확률']:.4%} | 분포대로 아무 값 |")
        lines.append("")
    bt = tables.get("백테스트")
    if bt is not None and len(bt):
        lines += ["## 백테스트 근거", "", bt.T.to_string(), ""]
    return "\n".join(lines)
