"""NDL Score 구성요소 가중치를 실제 팀 성적으로 최적화 (OOS 검증 포함).

기존 가중치(생산성25/효율20/플레이메이킹15/수비20/승리기여20)는 설계자가
임의로 정한 값이다. 이제 팀 실제 득실 마진이라는 외부 정답값이 있으므로,
가중치를 데이터로 학습하고 '학습에 쓰지 않은 시즌'에서 성능을 확인한다.

OOS 설계
  홀수 연도 시즌 = 학습, 짝수 연도 시즌 = 검증. 시즌을 통째로 분리하므로
  같은 시즌 안에서 정보가 새지 않는다. 학습 가중치는 비음수 최소제곱으로
  구해 '음수 가중치' 같은 해석 불가능한 해를 배제한다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

BASE = Path(__file__).parent
SEASONS_DIR = BASE / "seasons"
MARGINS_CSV = BASE / "team_margins.csv"  # fetch_team_margins.py 산출 (1947~2024)

COMPONENTS = ["NDL_Production", "NDL_Efficiency", "NDL_Playmaking",
              "NDL_Defense", "NDL_WinImpact"]
LABELS = ["생산성", "효율성", "플레이메이킹", "수비기여", "승리기여"]
ORIGINAL_WEIGHTS = np.array([0.25, 0.20, 0.15, 0.20, 0.20])
MIN_TEAMS = 8


def load_team_margins() -> pd.DataFrame:
    df = pd.read_csv(MARGINS_CSV)
    return df.rename(columns={"team_abbr": "team_id"})[["year_id", "team_id", "margin"]]


def build_team_table() -> pd.DataFrame:
    """팀-시즌 단위로 5개 구성요소의 출전시간 가중 평균 + 실제 마진을 결합."""
    margins = load_team_margins()
    rows = []
    for f in sorted(SEASONS_DIR.glob("players_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        year = int(f.stem.split("_")[1])
        data = json.loads(f.read_text(encoding="utf-8"))
        recs = []
        for p in data["players"]:
            team, mp, g = p.get("Team"), p.get("MP"), p.get("G")
            if team is None or mp is None or g is None or "TM" in str(team):
                continue
            vals = [p.get(c) for c in COMPONENTS]
            if any(v is None for v in vals):
                continue
            recs.append({"team_id": team, "w": float(mp) * float(g),
                         **{c: float(v) for c, v in zip(COMPONENTS, vals)}})
        if not recs:
            continue
        df = pd.DataFrame(recs)
        for c in COMPONENTS:
            df[c] = df[c] * df["w"]
        agg = df.groupby("team_id").agg({**{c: "sum" for c in COMPONENTS}, "w": "sum"}).reset_index()
        for c in COMPONENTS:
            agg[c] = agg[c] / agg["w"].replace(0, np.nan)
        agg["year_id"] = year
        agg["season"] = data["season"]
        rows.append(agg)

    table = pd.concat(rows, ignore_index=True)
    table = table.merge(margins, on=["year_id", "team_id"], how="inner")
    table = table.dropna(subset=COMPONENTS + ["margin"])
    counts = table.groupby("year_id")["team_id"].transform("size")
    return table[counts >= MIN_TEAMS].reset_index(drop=True)


def season_corr(table: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    """시즌별로 가중합 점수와 실제 마진의 상관계수."""
    score = table[COMPONENTS].values @ weights
    tmp = table.assign(score=score)
    return tmp.groupby("year_id").apply(
        lambda g: g["score"].corr(g["margin"]), include_groups=False)


def main():
    table = build_team_table()
    years = sorted(table["year_id"].unique())
    print(f"분석 대상: {len(table)}개 팀-시즌, {len(years)}개 시즌 "
          f"({min(years)}~{max(years)})\n")

    train = table[table["year_id"] % 2 == 1]
    test = table[table["year_id"] % 2 == 0]
    print(f"학습: {train['year_id'].nunique()}개 시즌 / 검증: {test['year_id'].nunique()}개 시즌\n")

    # 시즌마다 마진 스케일이 다르므로 시즌 내 표준화 후 학습
    def standardize(df):
        out = df.copy()
        for c in COMPONENTS + ["margin"]:
            out[c] = df.groupby("year_id")[c].transform(
                lambda s: (s - s.mean()) / (s.std(ddof=0) if s.std(ddof=0) else 1))
        return out

    tr_std = standardize(train)
    w_fit, _ = nnls(tr_std[COMPONENTS].values, tr_std["margin"].values)
    if w_fit.sum() > 0:
        w_norm = w_fit / w_fit.sum()
    else:
        w_norm = ORIGINAL_WEIGHTS

    print("=== 학습된 가중치 ===")
    for label, o, n in zip(LABELS, ORIGINAL_WEIGHTS, w_norm):
        print(f"  {label:<8} {o:>6.1%}  →  {n:>6.1%}   ({n - o:+.1%})")

    print("\n=== 성능 비교 (시즌별 상관계수 평균) ===")
    for name, subset in [("학습 시즌", train), ("검증 시즌(OOS)", test), ("전체", table)]:
        r_old = season_corr(subset, ORIGINAL_WEIGHTS).mean()
        r_new = season_corr(subset, w_norm).mean()
        print(f"  {name:<14} 기존 {r_old:.4f}  →  최적화 {r_new:.4f}   ({r_new - r_old:+.4f})")

    oos_old = season_corr(test, ORIGINAL_WEIGHTS)
    oos_new = season_corr(test, w_norm)
    improved = (oos_new > oos_old).sum()
    print(f"\n  OOS 검증 시즌 {len(oos_old)}개 중 {improved}개에서 개선 "
          f"({improved / len(oos_old):.0%})")

    (BASE / "optimized_weights.json").write_text(json.dumps({
        "components": COMPONENTS,
        "labels": LABELS,
        "original": ORIGINAL_WEIGHTS.tolist(),
        "optimized": [round(float(x), 4) for x in w_norm],
        "oos_corr_original": round(float(season_corr(test, ORIGINAL_WEIGHTS).mean()), 4),
        "oos_corr_optimized": round(float(season_corr(test, w_norm).mean()), 4),
        "train_seasons": int(train["year_id"].nunique()),
        "test_seasons": int(test["year_id"].nunique()),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: optimized_weights.json")


if __name__ == "__main__":
    main()
