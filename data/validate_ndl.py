"""NDL Score를 실제 팀 성적(득실 마진)으로 검증한다.

문제의식
  NDL Score는 가중치를 사람이 임의로 정한 합성 지표다. "이 점수가 실제로
  무언가를 측정하고 있는가"를 외부 기준으로 확인하지 않으면 근거가 약하다.

검증 설계
  팀에 속한 선수들의 NDL Score를 출전시간 가중 평균한 값이 그 팀의 실제
  경기당 득실 마진을 얼마나 설명하는지 시즌별로 상관계수/R²를 구한다.
  득실 마진은 538 ELO 데이터셋(1947~2015)의 게임별 실제 득점/실점에서 계산한다.

  비교군으로 기존 공인 지표(WS, BPM)도 같은 방식으로 평가해, NDL Score가
  이들 대비 어느 수준인지 상대적으로 판단할 수 있게 한다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
SEASONS_DIR = BASE / "seasons"
MARGINS_CSV = BASE / "team_margins.csv"  # fetch_team_margins.py 산출 (1947~2024)
OUT_JSON = BASE / "validation_results.json"

MIN_TEAMS = 8  # 시즌 내 팀 수가 이보다 적으면 상관계수가 불안정해 제외


def load_team_margins() -> pd.DataFrame:
    """팀-시즌별 경기당 득실 마진 (538 1947~2015 + NBA 팀박스 2016~2024)."""
    df = pd.read_csv(MARGINS_CSV)
    return df.rename(columns={"team_abbr": "team_id"})


def team_aggregate(players: list[dict], value_key: str) -> pd.DataFrame:
    """선수 리스트 → 팀별 출전시간 가중 평균."""
    rows = []
    for p in players:
        team = p.get("Team")
        val = p.get(value_key)
        mp, g = p.get("MP"), p.get("G")
        if team is None or val is None or mp is None or g is None:
            continue
        # 여러 팀을 뛴 선수의 합산 행(2TM/3TM 등)은 특정 팀에 귀속시킬 수 없어 제외
        if "TM" in str(team):
            continue
        rows.append({"team": team, "val": float(val), "w": float(mp) * float(g)})
    if not rows:
        return pd.DataFrame(columns=["team", "agg"])
    df = pd.DataFrame(rows)
    df["wv"] = df["val"] * df["w"]
    out = df.groupby("team").agg(wv=("wv", "sum"), w=("w", "sum")).reset_index()
    out["agg"] = out["wv"] / out["w"].replace(0, np.nan)
    return out[["team", "agg"]].rename(columns={"team": "team_id"})


def main():
    margins = load_team_margins()
    results = []

    for f in sorted(SEASONS_DIR.glob("players_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        year = int(f.stem.split("_")[1])
        data = json.loads(f.read_text(encoding="utf-8"))
        players = data["players"]
        season_margin = margins[margins["year_id"] == year]
        if len(season_margin) < MIN_TEAMS:
            continue

        row = {"season": data["season"], "year": year, "teams": 0}
        for label, key in [("NDL", "NDL_Score"), ("WS", "WS"), ("BPM", "BPM")]:
            agg = team_aggregate(players, key)
            if agg.empty:
                row[f"r_{label}"] = None
                continue
            merged = agg.merge(season_margin[["team_id", "margin"]], on="team_id", how="inner")
            if len(merged) < MIN_TEAMS:
                row[f"r_{label}"] = None
                continue
            r = merged["agg"].corr(merged["margin"])
            row[f"r_{label}"] = None if pd.isna(r) else round(float(r), 4)
            row["teams"] = len(merged)
        results.append(row)

    res = pd.DataFrame(results)
    res.to_json(OUT_JSON, orient="records", force_ascii=False, indent=2)

    print(f"{'season':<9}{'teams':>6}{'r(NDL)':>9}{'r(WS)':>9}{'r(BPM)':>9}")
    for _, r in res.iterrows():
        def fmt(v):
            return f"{v:>9.3f}" if v is not None and not pd.isna(v) else f"{'-':>9}"
        print(f"{r['season']:<9}{int(r['teams']):>6}{fmt(r['r_NDL'])}{fmt(r['r_WS'])}{fmt(r['r_BPM'])}")

    print("\n=== 전체 요약 (시즌별 상관계수의 평균) ===")
    for label in ["NDL", "WS", "BPM"]:
        vals = res[f"r_{label}"].dropna()
        if len(vals):
            print(f"  {label:<4} 평균 r = {vals.mean():.4f}  (중앙값 {vals.median():.4f}, "
                  f"R² 평균 {(vals**2).mean():.4f}, 시즌 {len(vals)}개)")


if __name__ == "__main__":
    main()
