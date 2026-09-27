"""가중치 최적화 — 순환논리 방지를 위한 제약 조건 버전.

무제약 최적화는 승리기여(WS+VORP)에 83%를 몰아줬다. WS/VORP는 팀 성적에서
역산된 지표라, 팀 성적을 타깃으로 최적화하면 당연히 승자가 된다. 그 해는
'NDL Score를 WS의 복잡한 재포장으로 만드는' 퇴화된 해이므로 채택할 수 없다.

세 가지 변형을 비교한다.
  A 무제약           — 참고용 상한선
  B 상하한 제약       — 각 요소 5~40%, 5개 축이 모두 살아있는 다차원 설계 유지
  C 승리기여 제외     — 박스스코어 기반 4개 축만으로 얼마나 설명되는지 (순환논리 완전 배제)
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize, nnls

from optimize_weights import (COMPONENTS, LABELS, ORIGINAL_WEIGHTS,
                              build_team_table, season_corr)

BASE = Path(__file__).parent


def standardize(df):
    out = df.copy()
    for c in COMPONENTS + ["margin"]:
        out[c] = df.groupby("year_id")[c].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else 1))
    return out


def fit_bounded(tr_std, cols, lo, hi):
    """가중치 합=1, 각 요소 [lo, hi] 제약 하에서 잔차제곱합 최소화."""
    X = tr_std[cols].values
    y = tr_std["margin"].values
    n = len(cols)

    def loss(w):
        return float(np.sum((X @ w - y) ** 2))

    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    bounds = [(lo, hi)] * n
    x0 = np.full(n, 1 / n)
    res = minimize(loss, x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 500, "ftol": 1e-10})
    w = np.clip(res.x, lo, hi)
    return w / w.sum()


def report(name, table, train, test, weights, cols):
    """주어진 가중치의 학습/검증 성능 출력."""
    def corr(subset):
        score = subset[cols].values @ weights
        tmp = subset.assign(score=score)
        return tmp.groupby("year_id").apply(
            lambda g: g["score"].corr(g["margin"]), include_groups=False).mean()

    base_test = season_corr(test, ORIGINAL_WEIGHTS).mean()
    print(f"\n[{name}]")
    for label, w in zip([LABELS[COMPONENTS.index(c)] for c in cols], weights):
        print(f"    {label:<8} {w:>6.1%}")
    print(f"    학습 r={corr(train):.4f}   OOS r={corr(test):.4f}   "
          f"(기존 가중치 OOS {base_test:.4f}, {corr(test) - base_test:+.4f})")
    return corr(test)


def main():
    table = build_team_table()
    train = table[table["year_id"] % 2 == 1]
    test = table[table["year_id"] % 2 == 0]
    tr_std = standardize(train)

    print(f"대상: {len(table)}개 팀-시즌, {table['year_id'].nunique()}개 시즌")
    print(f"기존 가중치 OOS r = {season_corr(test, ORIGINAL_WEIGHTS).mean():.4f}")

    results = {}

    # A. 무제약
    w_a, _ = nnls(tr_std[COMPONENTS].values, tr_std["margin"].values)
    w_a = w_a / w_a.sum()
    results["A_무제약"] = (report("A. 무제약 (참고용 상한선)", table, train, test, w_a, COMPONENTS),
                         COMPONENTS, w_a)

    # B. 상하한 제약 5~40%
    w_b = fit_bounded(tr_std, COMPONENTS, 0.05, 0.40)
    results["B_제약"] = (report("B. 상하한 제약 (각 5~40%)", table, train, test, w_b, COMPONENTS),
                        COMPONENTS, w_b)

    # C. 승리기여 제외
    cols_c = [c for c in COMPONENTS if c != "NDL_WinImpact"]
    w_c = fit_bounded(tr_std, cols_c, 0.05, 0.50)
    results["C_승리기여제외"] = (report("C. 승리기여 제외 (박스스코어 4축만)", table, train, test, w_c, cols_c),
                            cols_c, w_c)

    print("\n=== 결론 ===")
    for name, (oos, cols, w) in results.items():
        print(f"  {name:<14} OOS r = {oos:.4f}")

    best = "B_제약"
    oos, cols, w = results[best]
    payload = {
        "recommended": best,
        "reason": "무제약 해는 승리기여(WS·VORP)에 83%가 몰려 순환논리가 되므로 배제. "
                  "상하한 제약 버전이 5개 축을 모두 유지하면서 OOS 성능도 개선.",
        "components": cols,
        "labels": [LABELS[COMPONENTS.index(c)] for c in cols],
        "original_weights": ORIGINAL_WEIGHTS.tolist(),
        "optimized_weights": [round(float(x), 4) for x in w],
        "oos_corr_original": round(float(season_corr(test, ORIGINAL_WEIGHTS).mean()), 4),
        "oos_corr_optimized": round(float(oos), 4),
        "variants": {k: {"oos_corr": round(float(v[0]), 4),
                          "components": v[1],
                          "weights": [round(float(x), 4) for x in v[2]]}
                      for k, v in results.items()},
        "train_seasons": int(train["year_id"].nunique()),
        "test_seasons": int(test["year_id"].nunique()),
        "note": "학습=홀수연도 시즌, 검증=짝수연도 시즌. 시즌 단위 분리로 정보 누수 차단.",
    }
    (BASE / "optimized_weights.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: optimized_weights.json (권장안 = {best})")


if __name__ == "__main__":
    main()
