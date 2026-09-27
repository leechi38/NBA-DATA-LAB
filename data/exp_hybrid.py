"""플레이메이킹·수비기여의 비율/절대 조합 4가지를 전부 비교.

앞선 실험에서 두 축을 한꺼번에 바꾸면 성능이 떨어졌는데, 중복도를 보니 원인이 축마다 달랐다.
  수비기여(절대)는 생산성과 상관 0.902 — 사실상 같은 정보를 다시 재는 셈
  플레이메이킹(절대)은 생산성과 상관 0.156 — 오히려 비율판(0.412)보다 독립적
따라서 두 축을 따로 판단해야 한다. 네 조합을 같은 기준으로 학습·검증한다.
"""
import json
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

BASE = Path(__file__).parent
AXES = ["production", "efficiency", "playmaking", "defense", "win_impact"]
LABELS = ["생산성", "효율성", "플레이메이킹", "수비기여", "승리기여"]
ORIGINAL = np.array([0.25, 0.20, 0.15, 0.20, 0.20])

players = pd.read_csv(BASE / "_exp_components.csv")
margins = pd.read_csv(BASE / "team_margins.csv").rename(columns={"team_abbr": "Team"})


def team_table(pm_kind: str, def_kind: str) -> pd.DataFrame:
    cols = {"production": "production", "efficiency": "efficiency",
            "win_impact": "win_impact",
            "playmaking": f"playmaking_{pm_kind}", "defense": f"defense_{def_kind}"}
    d = players.copy()
    for tgt, src in cols.items():
        d[tgt] = d[src] * d["w"]
    agg = d.groupby(["year_id", "Team"]).agg(
        {**{a: "sum" for a in AXES}, "w": "sum"}).reset_index()
    for a in AXES:
        agg[a] = agg[a] / agg["w"].replace(0, np.nan)
    out = agg.merge(margins[["year_id", "Team", "margin"]], on=["year_id", "Team"])
    out = out.dropna(subset=AXES + ["margin"])
    return out[out.groupby("year_id")["Team"].transform("size") >= 8].reset_index(drop=True)


def season_corr(tbl, w):
    return tbl.assign(s=tbl[AXES].values @ w).groupby("year_id").apply(
        lambda g: g["s"].corr(g["margin"]), include_groups=False)


def fit(tbl):
    std = tbl.copy()
    for c in AXES + ["margin"]:
        std[c] = tbl.groupby("year_id")[c].transform(
            lambda s: (s - s.mean()) / (s.std(ddof=0) or 1))
    X, y = std[AXES].values, std["margin"].values
    res = minimize(lambda w: float(np.sum((X @ w - y) ** 2)), np.full(5, 0.2),
                   method="SLSQP", bounds=[(0.05, 0.40)] * 5,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 500, "ftol": 1e-10})
    w = np.clip(res.x, 0.05, 0.40)
    return w / w.sum()


rows = []
for pm, dfk in product(["rate", "abs"], ["rate", "abs"]):
    tbl = team_table(pm, dfk)
    train = tbl[(tbl.year_id <= 2015) & (tbl.year_id % 2 == 1)]
    test = tbl[(tbl.year_id <= 2015) & (tbl.year_id % 2 == 0)]
    hold = tbl[tbl.year_id >= 2016]
    w = fit(train)
    h_old, h_new = season_corr(hold, ORIGINAL), season_corr(hold, w)
    rows.append({
        "플레이메이킹": "비율" if pm == "rate" else "절대",
        "수비기여": "비율" if dfk == "rate" else "절대",
        "학습": season_corr(train, w).mean(),
        "검증": season_corr(test, w).mean(),
        "홀드아웃": h_new.mean(),
        "홀드아웃_개선시즌": f"{int((h_new > h_old).sum())}/{len(h_new)}",
        "weights": w,
    })

res = pd.DataFrame(rows).sort_values("홀드아웃", ascending=False)

print("=" * 74)
print("조합별 성능 (홀드아웃 순)")
print(f"{'플메':>6}{'수비':>6}{'학습':>10}{'검증':>10}{'홀드아웃':>12}{'개선시즌':>10}")
for _, r in res.iterrows():
    print(f"{r['플레이메이킹']:>6}{r['수비기여']:>6}{r['학습']:>10.4f}"
          f"{r['검증']:>10.4f}{r['홀드아웃']:>12.4f}{r['홀드아웃_개선시즌']:>10}")

best = res.iloc[0]
cur = res[(res["플레이메이킹"] == "비율") & (res["수비기여"] == "비율")].iloc[0]
print(f"\n현재 방식 대비 최고 조합의 이득: 검증 {best['검증'] - cur['검증']:+.4f}, "
      f"홀드아웃 {best['홀드아웃'] - cur['홀드아웃']:+.4f}")

print("\n" + "=" * 74)
print("조합별 학습 가중치")
print(f"{'플메':>6}{'수비':>6}" + "".join(f"{l:>12}" for l in LABELS))
for _, r in res.iterrows():
    print(f"{r['플레이메이킹']:>6}{r['수비기여']:>6}" + "".join(f"{x:>11.1%}" for x in r["weights"]))

(BASE / "exp_hybrid.json").write_text(json.dumps([
    {k: (round(float(v), 4) if isinstance(v, (int, float, np.floating)) else
         ([round(float(x), 4) for x in v] if k == "weights" else v))
     for k, v in r.items()} for r in rows], ensure_ascii=False, indent=2), encoding="utf-8")
print("\n저장: exp_hybrid.json")
