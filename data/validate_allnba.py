"""NDL Score 상위권이 실제 All-NBA 선정 선수와 얼마나 겹치는지 확인한다.

팀 득실차로 하는 검증은 "좋은 선수가 많은 팀이 실제로 잘했나"를 보는 간접 방식이다.
이 검증은 더 직접적이다 — 그 시즌 최고의 선수 N명을 꼽으라고 했을 때, 지표가 고른 명단이
리그가 실제로 선정한 명단과 얼마나 겹치는지 센다.

All-NBA는 기자단 투표라 박스스코어 지표와 완전히 일치할 수 없고, 그럴 필요도 없다.
비교 기준으로 기존 공인 지표(WS, BPM, PER)도 같은 방식으로 재서 나란히 놓는다.

3rd team은 1988-89 시즌부터 생겨 그 이전에는 10명, 이후에는 15명이 선정된다.
매 시즌 선정 인원 수에 맞춰 각 지표의 상위 N명을 뽑아 비교한다.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from fetch_all_seasons import dedupe_traded_players, fetch_html, parse_table

BASE = Path(__file__).parent
SEASONS_DIR = BASE / "seasons"
OUT = BASE / "validation_allnba.json"

METRICS = [("NDL_Score", "NDL Score"), ("WS", "WS"), ("BPM", "BPM"), ("PER", "PER")]
ALLNBA = re.compile(r"NBA[123]")


def allnba_players(season: int) -> set[str]:
    """해당 시즌 All-NBA 1·2·3팀 선정 선수 이름."""
    try:
        html = fetch_html("", f"per_game_stats_html_{season}")
    except Exception:
        return set()
    df = parse_table(html, "per_game_stats")
    if "Awards" not in df.columns:
        return set()
    df = dedupe_traded_players(df)
    hit = df[df["Awards"].astype(str).str.contains(ALLNBA, na=False)]
    return set(hit["Player"])


def main():
    rows = []
    for f in sorted(SEASONS_DIR.glob("players_*.json"), key=lambda p: int(p.stem.split("_")[1])):
        season = int(f.stem.split("_")[1])
        selected = allnba_players(season)
        if len(selected) < 8:          # 수상 정보가 없는 옛날 시즌은 건너뜀
            continue

        data = json.loads(f.read_text(encoding="utf-8"))
        players = pd.DataFrame(data["players"])
        # 여러 팀을 뛴 선수의 합산 행이 중복으로 잡히지 않도록 이름 기준 최고값만 사용
        row = {"season": data["season"], "year": season, "n": len(selected)}
        for col, label in METRICS:
            if col not in players.columns:
                row[label] = None
                continue
            best = players.dropna(subset=[col]).sort_values(col, ascending=False)
            top = list(dict.fromkeys(best["Player"]))[:len(selected)]
            row[label] = len(set(top) & selected) / len(selected)
        rows.append(row)

    res = pd.DataFrame(rows)
    labels = [l for _, l in METRICS]

    print(f"{'시즌':<9}{'선정':>5}" + "".join(f"{l:>12}" for l in labels))
    for _, r in res.iterrows():
        line = f"{r['season']:<9}{int(r['n']):>5}"
        for l in labels:
            line += f"{r[l]:>11.0%}" if pd.notna(r[l]) else f"{'-':>12}"
        print(line)

    print("\n" + "=" * 58)
    print(f"{'전체 평균':<14}" + "".join(f"{res[l].mean():>11.1%}" for l in labels))
    print(f"{'2000년 이후':<14}" + "".join(
        f"{res[res.year >= 2000][l].mean():>11.1%}" for l in labels))
    print(f"\n대상: {len(res)}개 시즌 ({res['season'].iloc[0]} ~ {res['season'].iloc[-1]})")

    OUT.write_text(json.dumps({
        "seasons": len(res),
        "range": [res["season"].iloc[0], res["season"].iloc[-1]],
        "overall": {l: round(float(res[l].mean()), 4) for l in labels},
        "since_2000": {l: round(float(res[res.year >= 2000][l].mean()), 4) for l in labels},
        "by_season": res.to_dict(orient="records"),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: {OUT.name}")


if __name__ == "__main__":
    main()
