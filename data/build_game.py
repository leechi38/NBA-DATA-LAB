"""시뮬레이션 게임 페이지(game.html)용 데이터를 만든다.
players_2025_26.json에서 엔진에 필요한 필드만 추려 트림하고,
구역별 리그 평균 성공률(BASE_RATES)을 계산해 함께 내보낸다."""
import json
from pathlib import Path

BASE = Path(__file__).parent
OUT_DIR = BASE.parent / "pages"
OUT_DIR.mkdir(exist_ok=True)

data = json.loads((BASE / "players_2025_26.json").read_text(encoding="utf-8"))
players = data["players"]

MIN_G = 15  # 로테이션에 들어갈 수 있는 최소 표본(다른 페이지들과 동일 기준)

ATTR_KEYS = [
    "Finishing", "Paint", "Midrange", "Corner3", "Arc3", "FreeThrow",
    "Playmaking", "BallHandling", "ShotCreation",
    "OnBallDefense", "RimProtection", "DefRebound", "OffRebound", "DefensiveIQ",
    "Teamwork", "Composure", "WinningMentality", "Experience", "Athleticism", "Stamina",
]
ZONE_KEYS = ["RA", "Paint", "Midrange", "Corner3", "Arc3"]


def zone_base_rate(field, pool):
    vals = [p[field] for p in pool if p.get(field) is not None]
    return round(sum(vals) / len(vals), 3) if vals else 0.4


pool = [p for p in players if p["G"] >= MIN_G]
paint_vals = []
arc3_vals = []
for p in pool:
    a = p.get("FGpct_3_10")
    b = p.get("FGpct_10_16")
    if a is not None and b is not None:
        paint_vals.append((a + b) / 2)
    c = p.get("Corner3PctOf3PA") or 0
    overall3 = p.get("FGpct_3P")
    corner3 = p.get("Corner3Pct")
    if overall3 is not None and corner3 is not None and (1 - c) > 0.05:
        arc3_vals.append((overall3 - corner3 * c) / (1 - c))

BASE_RATES = {
    "RA": zone_base_rate("FGpct_0_3", pool),
    "Paint": round(sum(paint_vals) / len(paint_vals), 3) if paint_vals else 0.42,
    "Midrange": zone_base_rate("FGpct_16_3P", pool),
    "Corner3": zone_base_rate("Corner3Pct", pool),
    "Arc3": round(sum(arc3_vals) / len(arc3_vals), 3) if arc3_vals else 0.35,
    "FT": zone_base_rate("FT%", pool),
}

game_players = []
for p in pool:
    game_players.append({
        "Player": p["Player"],
        "Team": p["Team"],
        "Pos": p["Pos"],
        "PosGroup": p["PosGroup"],
        "BbrefId": p["BbrefId"],
        "MP": p["MP"],
        "USG": p["USG%"],
        "NDL": p["NDL_Score"],
        "PTS": p["PTS"], "TRB": p["TRB"], "AST": p["AST"], "STL": p["STL"], "BLK": p["BLK"],
        "Attr": {k: p[f"Attr_{k}"] for k in ATTR_KEYS},
        "Zone": {k: p[f"Zone_{k}"] for k in ZONE_KEYS},
    })

template = (BASE / "game_template.html").read_text(encoding="utf-8")
out = template.replace(
    "__PLAYER_DATA__", json.dumps(game_players, ensure_ascii=False, separators=(",", ":"))
).replace(
    "__BASE_RATES__", json.dumps(BASE_RATES, separators=(",", ":"))
)

out_path = OUT_DIR / "game.html"
out_path.write_text(out, encoding="utf-8")
print(f"{out_path} 생성 완료 ({len(game_players)}명, base_rates={BASE_RATES}, {len(out)/1024:.0f} KB)")
