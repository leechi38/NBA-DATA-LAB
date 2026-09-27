"""팀-시즌별 실제 경기당 득실 마진 테이블을 만든다 (검증용 정답값).

두 개의 공개 소스를 합친다.
  1. 538 ELO 게임 로그        : 1947~2015 (nbaallelo.csv)
  2. NBA-Data-2010-2024 팀 박스: 2010~2024 (PLUS_MINUS 보유)

겹치는 2010~2015 구간은 두 소스가 독립적으로 같은 값을 내는지 대조해
데이터 신뢰성을 확인하는 용도로 쓰고, 최종 테이블에는 538 값을 채택한다.

basketball-reference를 직접 긁지 않는 이유: 해당 사이트가 Cloudflare JS
챌린지(cf-mitigated: challenge)를 걸어 비브라우저 클라이언트를 전부 막고 있다.
robots.txt조차 403이라 우회가 아니라 대체 소스를 쓰는 쪽을 택했다.
"""
import urllib.request
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
ELO_CSV = BASE / "nbaallelo.csv"
MODERN_CSV = BASE / "team_totals_2010_2024.csv"
OUT_CSV = BASE / "team_margins.csv"

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
MODERN_URL = ("https://raw.githubusercontent.com/NocturneBear/NBA-Data-2010-2024/main/"
              "regular_season_totals_2010_2024.csv")

# 두 소스의 팀 약자 표기 차이 (NBA 공식 → basketball-reference)
ABBR_FIX = {"PHX": "PHO", "BKN": "BRK", "CHA": "CHO"}


def download_modern():
    if MODERN_CSV.exists():
        return
    print("2010-2024 팀 박스스코어 다운로드 중...")
    req = urllib.request.Request(MODERN_URL, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=300) as r:
        MODERN_CSV.write_bytes(r.read())
    print(f"  저장: {MODERN_CSV.stat().st_size/1024/1024:.1f} MB")


def margins_from_elo() -> pd.DataFrame:
    df = pd.read_csv(ELO_CSV, usecols=["year_id", "lg_id", "is_playoffs",
                                        "team_id", "pts", "opp_pts"])
    df = df[(df["lg_id"] == "NBA") & (df["is_playoffs"] == 0)]
    g = df.groupby(["year_id", "team_id"]).agg(
        games=("pts", "size"), pf=("pts", "sum"), pa=("opp_pts", "sum")).reset_index()
    g["margin"] = (g["pf"] - g["pa"]) / g["games"]
    g["source"] = "538"
    return g.rename(columns={"team_id": "team_abbr"})[
        ["year_id", "team_abbr", "games", "margin", "source"]]


def margins_from_modern() -> pd.DataFrame:
    df = pd.read_csv(MODERN_CSV, usecols=["SEASON_YEAR", "TEAM_ABBREVIATION",
                                           "GAME_ID", "PTS", "PLUS_MINUS"])
    # "2022-23" → 2023 (우리 파이프라인은 시즌 종료 연도를 키로 쓴다)
    df["year_id"] = df["SEASON_YEAR"].str.slice(0, 4).astype(int) + 1
    df["team_abbr"] = df["TEAM_ABBREVIATION"].replace(ABBR_FIX)
    # 같은 경기가 중복 수록된 행이 있을 수 있어 (시즌,팀,경기) 단위로 중복 제거
    df = df.drop_duplicates(subset=["year_id", "team_abbr", "GAME_ID"])
    g = df.groupby(["year_id", "team_abbr"]).agg(
        games=("PTS", "size"), plus_minus=("PLUS_MINUS", "sum")).reset_index()
    g["margin"] = g["plus_minus"] / g["games"]
    g["source"] = "nba2010_2024"
    return g[["year_id", "team_abbr", "games", "margin", "source"]]


def main():
    download_modern()
    elo = margins_from_elo()
    modern = margins_from_modern()

    print(f"\n538      : {elo['year_id'].min()}~{elo['year_id'].max()}, {len(elo)}개 팀-시즌")
    print(f"2010-2024: {modern['year_id'].min()}~{modern['year_id'].max()}, {len(modern)}개 팀-시즌")

    # 겹치는 구간 교차검증
    overlap = elo.merge(modern, on=["year_id", "team_abbr"], suffixes=("_elo", "_mod"))
    if len(overlap):
        diff = (overlap["margin_elo"] - overlap["margin_mod"]).abs()
        print(f"\n[교차검증] 겹치는 {len(overlap)}개 팀-시즌에서 마진 차이")
        print(f"  평균 {diff.mean():.4f}  최대 {diff.max():.4f}  "
              f"0.1 초과 {int((diff > 0.1).sum())}건")
        worst = overlap.loc[diff.idxmax()]
        print(f"  최대 불일치: {int(worst['year_id'])} {worst['team_abbr']} "
              f"538={worst['margin_elo']:.2f} vs modern={worst['margin_mod']:.2f}")

    # 538이 커버하는 연도는 538만 쓰고, 그 이후 시즌만 modern으로 보충한다.
    # 팀 약자로 중복을 거르면 같은 구단의 표기가 소스마다 다른 경우(2013~14
    # 샬럿: 538 "CHA" vs 최신 "CHO")를 못 잡아 같은 팀이 두 줄로 들어간다.
    last_elo_year = int(elo["year_id"].max())
    extra = modern[modern["year_id"] > last_elo_year]
    combined = pd.concat([elo, extra], ignore_index=True).sort_values(["year_id", "team_abbr"])
    combined.to_csv(OUT_CSV, index=False, encoding="utf-8")

    by_year = combined.groupby("year_id").size()
    print(f"\n최종: {len(combined)}개 팀-시즌, {combined['year_id'].min()}~{combined['year_id'].max()}")
    print(f"  소스별: {combined['source'].value_counts().to_dict()}")
    print(f"  저장: {OUT_CSV.name}")
    print(f"  최근 12시즌 팀 수: {by_year.tail(12).to_dict()}")


if __name__ == "__main__":
    main()
