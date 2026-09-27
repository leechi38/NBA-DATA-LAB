"""홈 화면에 얹을 요약 데이터를 만든다.

데이터 프로젝트의 첫 화면이 링크 메뉴뿐이면 무엇을 가진 사이트인지 보이지 않는다.
실제 데이터에서 뽑은 역대 순위와 규모 수치를 함께 내보낸다.
"""
import json
from pathlib import Path

BASE = Path(__file__).parent
OUT = BASE / "home_data.json"
TOP_N = 8


def main():
    rows, seasons, players_seen = [], [], set()
    total_rows = 0

    for f in sorted((BASE / "seasons").glob("players_*.json"),
                    key=lambda p: int(p.stem.split("_")[1])):
        d = json.loads(f.read_text(encoding="utf-8"))
        seasons.append(d["season"])
        for p in d["players"]:
            total_rows += 1
            if p.get("BbrefId"):
                players_seen.add(p["BbrefId"])
            if p.get("NDL_Score") is None:
                continue
            rows.append({
                "player": p["Player"], "season": d["season"], "team": p.get("Team"),
                "ndl": p["NDL_Score"], "pts": p.get("PTS"),
                "trb": p.get("TRB"), "ast": p.get("AST"),
            })

    rows.sort(key=lambda r: (r["ndl"], r["pts"] or 0), reverse=True)

    # NDL Score는 0~100으로 잘리기 때문에 최상위권에 동점이 생긴다. 순위를 매기는 대신
    # "척도 상한에 닿은 시즌"으로 제시해야 동점이 오류가 아니라 의미로 읽힌다.
    at_cap = [r for r in rows if r["ndl"] >= 100]
    latest = seasons[-1]
    current = sorted((r for r in rows if r["season"] == latest),
                     key=lambda r: r["ndl"], reverse=True)[:5]

    payload = {
        "stats": {
            "seasons": len(seasons),
            "span": f"{seasons[0]} – {seasons[-1]}",
            "playerSeasons": total_rows,
            "players": len(players_seen),
        },
        "atCap": at_cap,
        "topSeasons": rows[:TOP_N],
        "currentSeason": {"label": latest, "players": current},
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"시즌 {len(seasons)} · 선수-시즌 {total_rows:,} · 고유 선수 {len(players_seen):,}")
    print(f"\n100점 상한에 닿은 시즌: {len(at_cap)}개")
    for r in at_cap:
        print(f"    {r['player']:<24}{r['season']}  {r['team']}  "
              f"{r['pts']}득점 {r['trb']}리바 {r['ast']}어시")
    print(f"\n{latest} 상위 5:")
    for r in current:
        print(f"    {r['ndl']:>5.1f}  {r['player']:<24}{r['team']}")
    print(f"\n저장: {OUT.name}")


if __name__ == "__main__":
    main()
