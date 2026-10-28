"""팀 레이팅 예측에 필요한 두 가지를 basketball-reference에서 받는다.

  1. 시즌별 팀 ORtg/DRtg (leagues/NBA_{year}_ratings.html) → team_ratings.csv
     공격·수비 레이팅은 포제션 추정에 턴오버가 필요해 1977-78 시즌부터 나온다.
     그 이전 시즌은 받아도 빈 칸이라 건너뛴다.
  2. 2026-27 시즌 30개 팀 로스터 (teams/{abbr}/2027.html) → rosters_2026_27.csv
     개막 직후라 per_game 표에는 아직 안 뛴 선수(부상자 등)가 빠져 있다.
     팀 페이지의 roster 표는 등록 선수 전원을 담고 있어 이쪽을 쓴다.

팀 약자는 표의 링크(/teams/BOS/2025.html)에서 뽑는다. 선수 스탯 표와 같은
체계라 이름 대신 약자로 바로 맞물린다.

bbref는 분당 20회를 넘기면 한 시간 차단하므로 실제 다운로드 사이에 간격을 둔다.
"""
import io
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
CACHE_DIR = BASE / ".cache"
RATINGS_CSV = BASE / "team_ratings.csv"
ROSTER_CSV = BASE / "rosters_2026_27.csv"

FIRST_RATED = 1978    # ORtg/DRtg가 처음 집계된 시즌(종료 연도)
TARGET = 2027         # 예측할 시즌 2026-27
SLEEP = 3.5

TEAM_LINK = re.compile(r'href="/teams/(\w+)/(\d{4})\.html">([^<]+)</a>')
PLAYER_LINK = re.compile(r"/players/\w/(\w+)\.html")


def fetch_html(url: str, cache_name: str, refresh: bool = False) -> str:
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{cache_name}.html"
    if path.exists() and not refresh:
        return path.read_text(encoding="utf-8")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
    path.write_text(html, encoding="utf-8")
    time.sleep(SLEEP)
    return html


def ratings(year: int, refresh: bool = False) -> pd.DataFrame:
    html = fetch_html(f"https://www.basketball-reference.com/leagues/NBA_{year}_ratings.html",
                      f"ratings_html_{year}", refresh)
    df = pd.read_html(io.StringIO(html), attrs={"id": "ratings"})[0]
    df.columns = [c[1] for c in df.columns]
    df = df[df["Team"] != "Team"].dropna(subset=["Team"])
    abbr = {name: a for a, _, name in TEAM_LINK.findall(html)}
    df["team_abbr"] = df["Team"].map(abbr)
    df["year_id"] = year
    keep = {"Team": "team", "Conf": "conf", "W": "w", "L": "l", "MOV": "mov",
            "ORtg": "ortg", "DRtg": "drtg", "NRtg": "nrtg",
            "ORtg/A": "ortg_adj", "DRtg/A": "drtg_adj", "NRtg/A": "nrtg_adj"}
    df = df.rename(columns=keep)[["year_id", "team_abbr", *keep.values()]]
    for c in ["w", "l", "mov", "ortg", "drtg", "nrtg", "ortg_adj", "drtg_adj", "nrtg_adj"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def injuries(html: str) -> dict:
    """부상자 표는 HTML 주석 안에 들어 있어 주석을 벗겨 읽는다. 선수 ID → 상태 설명."""
    m = re.search(r'<table[^>]*id="injuries".*?</table>', html.replace("<!--", "").replace("-->", ""),
                  re.S)
    if not m:
        return {}
    out = {}
    for row in re.findall(r"<tr.*?</tr>", m.group(0), re.S):
        pid = PLAYER_LINK.search(row)
        note = re.search(r'data-stat="note"[^>]*>(.*?)</td>', row, re.S)
        if pid and note:
            out[pid.group(1)] = re.sub(r"<[^>]+>", "", note.group(1)).strip()
    return out


def roster(abbr: str, refresh: bool = False) -> pd.DataFrame:
    html = fetch_html(f"https://www.basketball-reference.com/teams/{abbr}/{TARGET}.html",
                      f"team_{abbr}_{TARGET}", refresh)
    table = re.search(r'<table[^>]*id="roster".*?</table>', html, re.S).group(0)
    df = pd.read_html(io.StringIO(table))[0]
    ids = [PLAYER_LINK.search(r).group(1) for r in re.findall(r"<tr.*?</tr>", table, re.S)
           if PLAYER_LINK.search(r)]
    df = df[df["Player"] != "Player"].reset_index(drop=True)
    df["two_way"] = df["Player"].str.contains(r"\(TW\)")
    df["player"] = df["Player"].str.replace(r"\s*\(TW\)", "", regex=True)
    df["id"] = ids
    df["team_abbr"] = abbr
    df["exp"] = df["Exp"].astype(str)
    inj = injuries(html)
    df["injury"] = df["id"].map(inj).fillna("")
    return df[["team_abbr", "id", "player", "Pos", "exp", "two_way", "injury"]].rename(
        columns={"Pos": "pos"})


def main(refresh_current: bool = True):
    frames = []
    for y in range(FIRST_RATED, TARGET + 1):
        # 진행 중인 시즌은 캐시가 낡았을 수 있어 새로 받는다
        frames.append(ratings(y, refresh=refresh_current and y == TARGET))
        print(f"  ratings {y}: {len(frames[-1])}팀", flush=True)
    rt = pd.concat(frames, ignore_index=True)
    missing = rt["team_abbr"].isna().sum()
    assert missing == 0, f"팀 약자 매칭 실패 {missing}건"
    rt.to_csv(RATINGS_CSV, index=False)
    print(f"저장 {RATINGS_CSV.name}: {len(rt)}개 팀-시즌")

    teams = sorted(rt.loc[rt["year_id"] == TARGET, "team_abbr"])
    assert len(teams) == 30, teams
    rosters = []
    for t in teams:
        rosters.append(roster(t, refresh=refresh_current))
        print(f"  roster {t}: {len(rosters[-1])}명", flush=True)
    ro = pd.concat(rosters, ignore_index=True)
    ro.to_csv(ROSTER_CSV, index=False, encoding="utf-8")
    print(f"저장 {ROSTER_CSV.name}: {len(ro)}명 / 부상 표기 {(ro['injury'] != '').sum()}명")


if __name__ == "__main__":
    main()
