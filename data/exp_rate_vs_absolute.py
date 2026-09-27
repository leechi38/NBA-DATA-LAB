"""실험: 플레이메이킹·수비기여를 비율(%)로 쓸 것인가 절대 수치로 쓸 것인가.

배경
  현재 설계는 생산성 축에서 AST/STL/BLK를 절대 수치로 쓰면서, 플레이메이킹·수비기여
  축에서는 같은 스탯을 AST%/STL%/BLK%/DRB%로 쓴다. 한 지표 안에서 같은 스탯이 두 형태로
  들어가 있고, 어느 쪽이 팀 성적을 더 잘 설명하는지 확인한 적이 없다.

  비율은 출전시간·페이스가 이미 보정돼 시대 간 비교에 유리하고,
  절대 수치는 "실제로 얼마나 많이 했는가"를 담아 팀 기여 총량에 가깝다.

방법
  두 변형의 구성요소를 각각 계산해 팀 단위로 집계하고, 팀 실제 득실 마진과의
  상관계수를 비교한다. 구성요소 단독 비교와, 가중치까지 다시 학습한 최종 지표 비교를
  모두 본다. 학습/검증 분리는 기존과 동일(홀수=학습, 짝수=검증, 2016~=홀드아웃).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).parent))
from fetch_all_seasons import (NUM_COLS_ADV, NUM_COLS_PG, REF_MIN_G,
                                dedupe_traded_players, ensure_cols, fetch_html,
                                parse_table, pos_zscore_ref, primary_pos_group,
                                to_num, zscore_ref)

BASE = Path(__file__).parent
CACHE = BASE / "_exp_components.csv"
MARGINS = BASE / "team_margins.csv"
SEASON_START, SEASON_END = 1950, 2026

AXES = ["production", "efficiency", "playmaking", "defense", "win_impact"]
LABELS = {"production": "생산성", "efficiency": "효율성", "playmaking": "플레이메이킹",
          "defense": "수비기여", "win_impact": "승리기여"}
ORIGINAL = np.array([0.25, 0.20, 0.15, 0.20, 0.20])


def season_components(season: int) -> pd.DataFrame | None:
    """캐시된 HTML에서 시즌 하나의 구성요소를 두 변형 모두 계산."""
    try:
        pg_html = fetch_html("", f"per_game_stats_html_{season}")
        adv_html = fetch_html("", f"advanced_html_{season}")
    except Exception:
        return None

    pg = dedupe_traded_players(parse_table(pg_html, "per_game_stats"))
    adv = dedupe_traded_players(parse_table(adv_html, "advanced"))
    pg = ensure_cols(pg, NUM_COLS_PG)
    for c in NUM_COLS_PG:
        pg[c] = to_num(pg[c])
    adv = ensure_cols(adv, NUM_COLS_ADV)
    for c in NUM_COLS_ADV:
        adv[c] = to_num(adv[c])

    df = pg.merge(adv[["Player", "Team"] + NUM_COLS_ADV], on=["Player", "Team"],
                  how="left", suffixes=("", "_adv"))
    df["PosGroup"] = df["Pos"].map(primary_pos_group)
    df["TotalMP"] = df["MP"] * df["G"]
    ref = df["G"] >= REF_MIN_G

    def z(col):
        return zscore_ref(df[col], ref).fillna(0)

    def pz(col):
        return pos_zscore_ref(df, col, ref).fillna(0)

    out = pd.DataFrame({
        "year_id": season,
        "Team": df["Team"],
        "w": df["TotalMP"].fillna(0),
        # 두 변형이 공유하는 축
        "production": z("PTS") + z("TRB") + z("AST") + z("STL") + z("BLK") - z("TOV"),
        "efficiency": pz("TS%") + pz("eFG%"),
        "win_impact": z("WS") + z("VORP"),
        # 변형 A: 비율 (현재 방식)
        "playmaking_rate": pz("AST%") - pz("TOV%"),
        "defense_rate": pz("STL%") + pz("BLK%") + pz("DRB%"),
        # 변형 B: 절대 수치 (경기당)
        "playmaking_abs": pz("AST") - pz("TOV"),
        "defense_abs": pz("STL") + pz("BLK") + pz("DRB"),
    })
    return out[~out["Team"].astype(str).str.contains("TM", na=False)]


def build_cache() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_csv(CACHE)
    frames = []
    for season in range(SEASON_START, SEASON_END + 1):
        d = season_components(season)
        if d is not None and len(d):
            frames.append(d)
            print(f"  {season} ", end="", flush=True)
    print()
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(CACHE, index=False, encoding="utf-8")
    return df


def team_table(players: pd.DataFrame, variant: str) -> pd.DataFrame:
    """선수 → 팀 단위 출전시간 가중 평균 + 실제 마진 결합."""
    cols = ["production", "efficiency", "win_impact",
            f"playmaking_{variant}", f"defense_{variant}"]
    d = players.copy()
    for c in cols:
        d[c] = d[c] * d["w"]
    agg = d.groupby(["year_id", "Team"]).agg(
        {**{c: "sum" for c in cols}, "w": "sum"}).reset_index()
    for c in cols:
        agg[c] = agg[c] / agg["w"].replace(0, np.nan)
    agg = agg.rename(columns={f"playmaking_{variant}": "playmaking",
                               f"defense_{variant}": "defense"})

    margins = pd.read_csv(MARGINS).rename(columns={"team_abbr": "Team"})
    out = agg.merge(margins[["year_id", "Team", "margin"]], on=["year_id", "Team"])
    out = out.dropna(subset=AXES + ["margin"])
    counts = out.groupby("year_id")["Team"].transform("size")
    return out[counts >= 8].reset_index(drop=True)


def season_corr(tbl, weights):
    score = tbl[AXES].values @ weights
    return tbl.assign(s=score).groupby("year_id").apply(
        lambda g: g["s"].corr(g["margin"]), include_groups=False)


def fit_weights(tbl):
    """시즌 내 표준화 후 5~40% 제약으로 학습 (운영 방식과 동일)."""
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


def main():
    print("구성요소 계산 중(캐시된 HTML 사용)...")
    players = build_cache()
    print(f"선수-시즌 {len(players):,}행\n")

    results = {}
    for variant, name in [("rate", "비율 (%, 현재)"), ("abs", "절대 수치")]:
        tbl = team_table(players, variant)
        train = tbl[(tbl.year_id <= 2015) & (tbl.year_id % 2 == 1)]
        test = tbl[(tbl.year_id <= 2015) & (tbl.year_id % 2 == 0)]
        hold = tbl[tbl.year_id >= 2016]

        # 1) 구성요소 단독으로 팀 마진을 얼마나 설명하는가
        solo = {}
        for axis in ["playmaking", "defense"]:
            r = tbl.groupby("year_id").apply(
                lambda g: g[axis].corr(g["margin"]), include_groups=False)
            solo[axis] = r.mean()

        # 2) 가중치까지 다시 학습했을 때 최종 성능
        w = fit_weights(train)
        results[variant] = {
            "name": name, "weights": w, "solo": solo,
            "train": season_corr(train, w).mean(),
            "test": season_corr(test, w).mean(),
            "hold": season_corr(hold, w).mean(),
            "orig_test": season_corr(test, ORIGINAL).mean(),
        }

    print("=" * 66)
    print("1) 구성요소 단독 — 팀 마진과의 상관계수 (전 시즌 평균)")
    print(f"{'':16}{'비율(%)':>12}{'절대 수치':>12}{'차이':>10}")
    for axis in ["playmaking", "defense"]:
        a = results["rate"]["solo"][axis]
        b = results["abs"]["solo"][axis]
        print(f"  {LABELS[axis]:<14}{a:>12.4f}{b:>12.4f}{b - a:>+10.4f}")

    print("\n" + "=" * 66)
    print("2) 가중치 재학습 후 최종 지표 성능")
    print(f"{'':16}{'학습':>10}{'검증':>10}{'홀드아웃':>12}")
    for v in ["rate", "abs"]:
        r = results[v]
        print(f"  {r['name']:<14}{r['train']:>10.4f}{r['test']:>10.4f}{r['hold']:>12.4f}")
    d_test = results["abs"]["test"] - results["rate"]["test"]
    d_hold = results["abs"]["hold"] - results["rate"]["hold"]
    print(f"  {'차이(절대-비율)':<14}{'':>10}{d_test:>+10.4f}{d_hold:>+12.4f}")

    print("\n" + "=" * 66)
    print("3) 학습된 가중치")
    print(f"{'':16}" + "".join(f"{LABELS[a]:>12}" for a in AXES))
    for v in ["rate", "abs"]:
        r = results[v]
        print(f"  {r['name']:<14}" + "".join(f"{x:>11.1%}" for x in r["weights"]))

    (BASE / "exp_rate_vs_absolute.json").write_text(json.dumps({
        v: {"name": r["name"], "weights": [round(float(x), 4) for x in r["weights"]],
            "solo": {k: round(float(x), 4) for k, x in r["solo"].items()},
            "train": round(float(r["train"]), 4), "test": round(float(r["test"]), 4),
            "holdout": round(float(r["hold"]), 4)}
        for v, r in results.items()}, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n저장: exp_rate_vs_absolute.json")


if __name__ == "__main__":
    main()
