"""data/seasons/players_{season}.json 77개를 합쳐서 웹용 압축 데이터를 만든다.

1. all_seasons_stats.json   — 시즌 선택형 스탯 페이지용.
2. all_seasons_ndl.json     — 시즌 선택형 NDL Score 페이지용.
3. careers.json             — 선수 개별 페이지(커리어 추이)용. BbrefId로 묶은
   선수별 시즌 로그 배열 (숫자 위주로 압축, 배열 형태로 필드명 반복 제거).
4. all_seasons_scatter.json — 포지셔닝(산점도) 페이지용. 표본이 불안정한 선수를
   제외(G>=15)하고 산점도 축으로 쓸 수 있는 수치 필드만 남긴다.
5. all_seasons_game.json    — 시즌 시뮬레이션 페이지용. 슛 구역별 데이터가 있는
   1996-97 이후 시즌만 포함한다. 엔진이 구역별 슛 선택으로 포제션을 계산하는데,
   그 이전 시즌은 구역 분포가 전부 균등값(0.2)이라 시뮬레이션이 무의미해진다.
   구역별 리그 평균 성공률(BASE_RATES)도 시즌마다 따로 계산한다 — 1996-97과
   2025-26의 리그 평균 3점 성공률이 다른데 하나의 값을 공유하면 왜곡된다.
"""
import json
from pathlib import Path

BASE = Path(__file__).parent
SEASONS_DIR = BASE / "seasons"
OUT_DIR = BASE

STATS_FIELDS = ["Player", "Team", "TeamsPlayed", "Pos", "BbrefId", "G", "MP", "PTS", "TRB",
                "AST", "STL", "BLK", "FG%", "3P%", "TS%", "PER", "WS", "BPM"]
NDL_FIELDS = ["Player", "Team", "Pos", "BbrefId", "G", "MP", "NDL_Score", "NDL_Weight",
              "PTS", "TRB", "AST", "NDL_Production", "NDL_Efficiency",
              "NDL_Playmaking", "NDL_Defense", "NDL_WinImpact"]

# 커리어 로그에 남길 스탯(순서 고정 — careers.json에서 배열 인덱스로 접근).
CAREER_STAT_KEYS = ["G", "MP", "PTS", "TRB", "AST", "STL", "BLK", "FG%", "3P%",
                     "TS%", "PER", "WS", "BPM", "NDL_Score"]

SCATTER_FIELDS = ["Player", "Team", "Pos", "PosGroup", "BbrefId", "MP", "G",
                  "FGA", "FG%", "3PA", "3P%", "eFG%", "TS%", "PTS", "TRB", "AST",
                  "TOV", "USG%", "PER", "NDL_Score"]

GAME_ATTR_KEYS = [
    "Finishing", "Paint", "Midrange", "Corner3", "Arc3", "FreeThrow",
    "Playmaking", "BallHandling", "ShotCreation",
    "OnBallDefense", "RimProtection", "DefRebound", "OffRebound", "DefensiveIQ",
    "Teamwork", "Composure", "WinningMentality", "Experience", "Athleticism", "Stamina",
]
GAME_ZONE_KEYS = ["RA", "Paint", "Midrange", "Corner3", "Arc3"]

MIN_G = 15  # 산점도·시뮬레이션 공통: 표본이 불안정한 선수 제외 기준


def zone_base_rate(field, pool, fallback):
    vals = [p[field] for p in pool if p.get(field) is not None]
    return round(sum(vals) / len(vals), 3) if vals else fallback


def season_base_rates(pool):
    """해당 시즌 리그 평균 구역별 성공률. 시뮬레이션 엔진의 기준선이 된다."""
    paint_vals, arc3_vals = [], []
    for p in pool:
        a, b = p.get("FGpct_3_10"), p.get("FGpct_10_16")
        if a is not None and b is not None:
            paint_vals.append((a + b) / 2)
        corner_share = p.get("Corner3PctOf3PA") or 0
        overall3, corner3 = p.get("FGpct_3P"), p.get("Corner3Pct")
        if overall3 is not None and corner3 is not None and (1 - corner_share) > 0.05:
            arc3_vals.append((overall3 - corner3 * corner_share) / (1 - corner_share))
    return {
        "RA": zone_base_rate("FGpct_0_3", pool, 0.6),
        "Paint": round(sum(paint_vals) / len(paint_vals), 3) if paint_vals else 0.42,
        "Midrange": zone_base_rate("FGpct_16_3P", pool, 0.4),
        "Corner3": zone_base_rate("Corner3Pct", pool, 0.37),
        "Arc3": round(sum(arc3_vals) / len(arc3_vals), 3) if arc3_vals else 0.35,
        "FT": zone_base_rate("FT%", pool, 0.75),
    }


def game_player(p):
    return {
        "Player": p["Player"], "Team": p["Team"], "Pos": p["Pos"],
        "PosGroup": p["PosGroup"], "BbrefId": p["BbrefId"],
        "MP": p["MP"], "USG": p["USG%"], "NDL": p["NDL_Score"],
        "PTS": p["PTS"], "TRB": p["TRB"], "AST": p["AST"],
        "STL": p["STL"], "BLK": p["BLK"],
        "Attr": {k: p[f"Attr_{k}"] for k in GAME_ATTR_KEYS},
        "Zone": {k: p[f"Zone_{k}"] for k in GAME_ZONE_KEYS},
    }


def pick(p, fields):
    return {k: p.get(k) for k in fields}


def main():
    season_files = sorted(SEASONS_DIR.glob("players_*.json"),
                           key=lambda p: int(p.stem.split("_")[1]))

    stats_by_season = {}
    ndl_by_season = {}
    scatter_by_season = {}
    game_by_season = {}
    careers = {}  # bbrefId -> {"name": str, "seasons": [[season, team, *stats], ...]}

    for f in season_files:
        data = json.loads(f.read_text(encoding="utf-8"))
        season_label = data["season"]  # e.g. "2025-26"
        season_year = data["generated"]  # e.g. 2026
        players = data["players"]

        stats_by_season[season_label] = [pick(p, STATS_FIELDS) for p in players]
        ndl_by_season[season_label] = [pick(p, NDL_FIELDS) for p in players]

        pool = [p for p in players if (p.get("G") or 0) >= MIN_G]
        scatter_by_season[season_label] = [pick(p, SCATTER_FIELDS) for p in pool]
        if data.get("has_shooting_data") and pool:
            game_by_season[season_label] = {
                "baseRates": season_base_rates(pool),
                "players": [game_player(p) for p in pool],
            }

        for p in players:
            bid = p.get("BbrefId")
            if not bid:
                continue
            if bid not in careers:
                careers[bid] = {"name": p.get("Player"), "seasons": []}
            row = [season_year, p.get("Team")] + [p.get(k) for k in CAREER_STAT_KEYS]
            careers[bid]["seasons"].append(row)

    seasons_list = list(stats_by_season.keys())  # 오래된 -> 최신 순

    out1 = OUT_DIR / "all_seasons_stats.json"
    out1.write_text(json.dumps({"seasons": seasons_list, "stats": stats_by_season},
                                ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    out1b = OUT_DIR / "all_seasons_ndl.json"
    out1b.write_text(json.dumps({"seasons": seasons_list, "ndl": ndl_by_season},
                                 ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    careers_payload = {
        "statKeys": CAREER_STAT_KEYS,
        "players": careers,  # bbrefId -> {name, seasons:[[year, team, ...stats], ...]}
    }
    out2 = OUT_DIR / "careers.json"
    out2.write_text(json.dumps(careers_payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    out3 = OUT_DIR / "all_seasons_scatter.json"
    out3.write_text(json.dumps({"seasons": seasons_list, "scatter": scatter_by_season},
                                ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    game_seasons = list(game_by_season.keys())
    out4 = OUT_DIR / "all_seasons_game.json"
    out4.write_text(json.dumps({"seasons": game_seasons, "game": game_by_season},
                                ensure_ascii=False, separators=(",", ":")), encoding="utf-8")

    print(f"{out1.name}: {out1.stat().st_size/1024/1024:.2f} MB, {len(stats_by_season)}개 시즌")
    print(f"{out1b.name}: {out1b.stat().st_size/1024/1024:.2f} MB, {len(ndl_by_season)}개 시즌")
    print(f"{out2.name}: {out2.stat().st_size/1024/1024:.2f} MB, {len(careers)}명 선수")
    print(f"{out3.name}: {out3.stat().st_size/1024/1024:.2f} MB, {len(scatter_by_season)}개 시즌 "
          f"(G>={MIN_G} 필터)")
    print(f"{out4.name}: {out4.stat().st_size/1024/1024:.2f} MB, {len(game_seasons)}개 시즌 "
          f"({game_seasons[0]}~{game_seasons[-1]}, 슛구역 데이터 보유 시즌만)")


if __name__ == "__main__":
    main()
