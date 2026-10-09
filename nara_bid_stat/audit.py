"""데이터·방법론 감사(audit).

1. 복수예가 데이터 감사: 생성 규칙, 4개 선택의 무작위성, 부호 기저율, 규칙 변경, 자기상관
2. 튜닝 절차 감사: '정보가 전혀 없는 모델'로도 자동튜닝이 통과되는지(귀무 시뮬레이션)
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

from . import scoring
from .data import P_COLS
from .mechanism import KNOWN_SCHEMES, combo_means, identify_scheme, mechanism_signature
from .walkforward import history_rate_forecaster, mechanism_forecaster, recent_normal_forecaster, summarize, walk_forward

# ---------------------------------------------------------------------------
# 튜닝 절차 귀무 시뮬레이션
# ---------------------------------------------------------------------------
DEFAULT_THRESHOLDS = (0.54, 0.56, 0.58, 0.59, 0.60, 0.62, 0.64, 0.66, 0.68, 0.70)

Gate = Callable[[np.ndarray, np.ndarray], "tuple[bool, float]"]


def utility_gate(thresholds: Sequence[float] = DEFAULT_THRESHOLDS, min_share: float = 0.20, min_n: int = 10, min_acc: float = 0.54) -> Gate:
    """'선택건수 >= max(10, 20%) 이고 정확도 >= 0.54 인 임계값 중 (acc-0.5)*sqrt(n) 최대' 방식.

    소표본 부호 임계값 자동튜닝에서 흔히 쓰는 규칙을 그대로 구현한 것이다.
    """

    def gate(y: np.ndarray, p: np.ndarray) -> tuple[bool, float]:
        n = len(y)
        need = max(min_n, math.ceil(n * min_share))
        best = None
        conf = np.maximum(p, 1 - p)
        for th in thresholds:
            sel = conf >= th
            k = int(sel.sum())
            if k < need:
                continue
            acc = float(((p[sel] > 0.5).astype(int) == y[sel]).mean())
            if acc < min_acc:
                continue
            u = (acc - 0.5) * math.sqrt(k)
            if best is None or u > best[0]:
                best = (u, acc)
        return (best is not None, best[1] if best else float("nan"))

    return gate


def strict_gate(prior_positive: float, thresholds: Sequence[float] = DEFAULT_THRESHOLDS, alpha: float = 0.05, min_n: int = 200) -> Gate:
    """권장 게이트: 구조적 기저율(항상 다수부호 찍기)보다 유의하게 높아야 통과.

    * 비교 기준이 0.5 가 아니라 max(prior, 1-prior) (예: ±3% 8/7 규칙이면 0.551)
    * 임계값 후보 수만큼 Bonferroni 보정
    * 최소 표본 200
    """
    base = max(prior_positive, 1 - prior_positive)

    def gate(y: np.ndarray, p: np.ndarray) -> tuple[bool, float]:
        conf = np.maximum(p, 1 - p)
        best = None
        for th in thresholds:
            sel = conf >= th
            k = int(sel.sum())
            if k < min_n:
                continue
            hits = int(((p[sel] > 0.5).astype(int) == y[sel]).sum())
            pv = scoring.binom_test_greater(hits, k, base)
            if pv < alpha / len(thresholds) and (best is None or pv < best[0]):
                best = (pv, hits / k)
        return (best is not None, best[1] if best else float("nan"))

    return gate


def null_tuning_simulation(gate: Gate, *, n_cases: int = 60, prior_positive: float = 0.44, trials: int = 5000, seed: int = 0) -> dict:
    """실제 부호와 무관한 확률(p_pos)을 내는 모델로 gate 를 반복 실행한다.

    통과율이 alpha(예: 5%)보다 훨씬 크면 그 튜닝 절차는 우연을 '성능'으로 오인한다.
    """
    rng = np.random.default_rng(seed)
    fired, accs = 0, []
    for _ in range(trials):
        y = (rng.random(n_cases) < prior_positive).astype(int)
        p = rng.beta(4, 4, n_cases)  # 정보 없음: y 와 독립
        ok, acc = gate(y, p)
        if ok:
            fired += 1
            accs.append(acc)
    return {
        "trials": trials,
        "n_cases": n_cases,
        "fire_rate": fired / trials,
        "mean_reported_accuracy_when_fired": float(np.mean(accs)) if accs else float("nan"),
    }


# ---------------------------------------------------------------------------
# 복수예가 데이터 감사
# ---------------------------------------------------------------------------
def selection_pit(prices: np.ndarray, rates: np.ndarray, tol: float = 2.5e-4, chunk: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    """공개된 15개 예가의 1365조합 평균 중 실제 예정가격의 위치(PIT)와 일치 조합 수.

    4개 선택이 무작위라면 PIT 는 U(0,1) 을 따른다. 메모리를 아끼려고 chunk 행씩 계산한다.
    """
    x = np.atleast_2d(np.asarray(prices, dtype=float))
    y_all = np.asarray(rates, dtype=float)
    pit = np.empty(len(x))
    eq = np.empty(len(x), dtype=np.int64)
    for s in range(0, len(x), chunk):
        m = combo_means(x[s:s + chunk])
        y = y_all[s:s + chunk, None]
        e = (np.abs(m - y) <= tol).sum(axis=1)
        pit[s:s + chunk] = ((m < y - tol).sum(axis=1) + 0.5 * e) / m.shape[1]
        eq[s:s + chunk] = e
    return pit, eq


def scheme_table(df: pd.DataFrame) -> pd.DataFrame:
    x = df[P_COLS].to_numpy(dtype=float)
    t = df[["org"]].copy()
    t["scheme"] = identify_scheme(x)
    return t.groupby(["org", "scheme"]).size().unstack(fill_value=0)


def regime_changes(df: pd.DataFrame, freq: str = "Q", min_share: float = 0.8, min_n: int = 5) -> pd.DataFrame:
    """기관별 분기 대표 생성규칙이 바뀐 지점.

    판별된 규칙명(EQ15_2 등)을 우선 쓰고, 판별되지 않으면 '±범위/음수개수' 서명을 쓴다.
    (15등분 규칙은 0을 걸친 칸 때문에 음수 개수가 7/8로 자연스럽게 바뀌므로 서명만 쓰면 오탐이 난다.)
    공고가 min_n 건 미만이거나 한 규칙이 min_share 미만인 분기는 건너뛰고,
    안정된 분기끼리 규칙이 다를 때만 변경으로 기록한다.
    """
    x = df[P_COLS].to_numpy(dtype=float)
    sch = identify_scheme(x)
    sig = mechanism_signature(x)
    t = df[["org", "date"]].copy()
    t["label"] = np.where(np.isin(sch, ["UNKNOWN", "AMBIGUOUS"]), sig, sch)
    t["period"] = t["date"].dt.to_period(freq)
    rows = []
    for org, g in t.groupby("org"):
        prev, prev_per = None, None
        for per, h in g.groupby("period"):
            if len(h) < min_n:
                continue
            vc = h["label"].value_counts()
            if vc.iloc[0] / len(h) < min_share:
                continue
            label = vc.index[0]
            if prev is not None and label != prev:
                rows.append({"org": org, "last_stable_period": str(prev_per), "period": str(per), "from": prev, "to": label, "n_in_period": len(h)})
            prev, prev_per = label, per
    return pd.DataFrame(rows, columns=["org", "last_stable_period", "period", "from", "to", "n_in_period"])


def theory_vs_empirical(df: pd.DataFrame, n_sim: int = 200_000) -> pd.DataFrame:
    x = df[P_COLS].to_numpy(dtype=float)
    sch = identify_scheme(x)
    rows = []
    for s in KNOWN_SCHEMES:
        mask = sch == s.name
        n = int(mask.sum())
        if n == 0:
            continue
        th = s.theoretical(n_sim=n_sim)
        y = df.loc[mask, "rate"].to_numpy(dtype=float)
        k = int((y > 0).sum())
        lo, hi = scoring.wilson_ci(k, n)
        rows.append({
            "scheme": s.name, "n": n,
            "theory_p_pos": th.prob_positive, "empirical_p_pos": k / n, "empirical_p_pos_ci": f"[{lo:.3f},{hi:.3f}]",
            "theory_mean": th.mean, "empirical_mean": float(y.mean()),
            "theory_sd": th.std, "empirical_sd": float(y.std(ddof=1)) if n > 1 else float("nan"),
        })
    return pd.DataFrame(rows)


def autocorrelation_table(df: pd.DataFrame, min_n: int = 100) -> pd.DataFrame:
    rows = []
    for org, g in df.groupby("org"):
        if len(g) < min_n:
            continue
        y = g.sort_values("date")["rate"]
        ac = float(y.autocorr(1))
        rows.append({"org": org, "n": len(g), "lag1_autocorr": ac, "flag_beyond_2se": abs(ac) > 2 / math.sqrt(len(g))})
    return pd.DataFrame(rows).sort_values("lag1_autocorr")


def audit_prebid(df: pd.DataFrame, *, oos_start: str = "2024-01-01", max_cases: int | None = 4000, seed: int = 0) -> dict[str, pd.DataFrame]:
    """전체 감사. 반환값은 이름 -> 표."""
    out: dict[str, pd.DataFrame] = {}
    x = df[P_COLS].to_numpy(dtype=float)
    pit_v, nmatch = selection_pit(x, df["rate"].to_numpy(dtype=float))
    sch = identify_scheme(x)
    sig = mechanism_signature(x)
    grp = np.array([f"UNKNOWN({g})" if s == "UNKNOWN" else s for s, g in zip(sch, sig)], dtype=object)
    rows = []
    for g in sorted(set(grp)):
        m = (grp == g) & (nmatch > 0)
        if m.sum() < 30:
            continue
        ks = scoring.ks_uniform(pit_v[m])
        rows.append({"group": g, "n": int(m.sum()), "mean_pit": float(pit_v[m].mean()), "ks_p": ks["p"],
                     "decile_shares": " ".join(f"{v:.3f}" for v in scoring.pit_histogram(pit_v[m]))})
    out["selection_randomness"] = pd.DataFrame(rows)
    out["scheme_by_org"] = scheme_table(df).reset_index()
    out["theory_vs_empirical"] = theory_vs_empirical(df)
    out["regime_changes"] = regime_changes(df)
    out["autocorrelation"] = autocorrelation_table(df)
    fc = {
        "mechanism60": mechanism_forecaster(60),
        "history_all": history_rate_forecaster(None),
        "recent_normal20": recent_normal_forecaster(20),
    }
    wf = walk_forward(df, fc, start=oos_start, max_cases=max_cases, seed=seed)
    out["walkforward_cases"] = wf
    out["walkforward_summary"] = summarize(wf, baseline="mechanism60") if len(wf) else pd.DataFrame()
    nulls = [
        {"gate": "utility_gate(소표본 튜닝 방식)", **null_tuning_simulation(utility_gate(), n_cases=60, trials=3000, seed=seed)},
        {"gate": "strict_gate(권장)", **null_tuning_simulation(strict_gate(0.44), n_cases=600, trials=1000, seed=seed)},
    ]
    out["null_tuning"] = pd.DataFrame(nulls)
    return out


def write_audit(tables: dict[str, pd.DataFrame], out_dir: str | Path) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, t in tables.items():
        t.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
    lines = ["# 복수예가 데이터 감사 요약", ""]
    for name in ("selection_randomness", "theory_vs_empirical", "walkforward_summary", "null_tuning", "regime_changes"):
        t = tables.get(name)
        if t is None or t.empty:
            continue
        lines += [f"## {name}", "", _md_table(t), ""]
    p = out / "AUDIT_SUMMARY.md"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def _md_table(t: pd.DataFrame, max_rows: int = 60) -> str:
    t = t.head(max_rows)
    cols = list(t.columns)

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.4f}"
        return str(v)

    head = "| " + " | ".join(map(str, cols)) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in t.itertuples(index=False)]
    return "\n".join([head, sep, *body])
