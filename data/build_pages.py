"""템플릿 + players_2025_26.json 을 합쳐 pages/*.html 최종 파일을 생성한다."""
import json
from pathlib import Path

BASE = Path(__file__).parent
OUT_DIR = BASE.parent / "pages"
OUT_DIR.mkdir(exist_ok=True)

data = json.loads((BASE / "players_2025_26.json").read_text(encoding="utf-8"))
players = data["players"]

STATS_FIELDS = ["Player", "Team", "TeamsPlayed", "Pos", "G", "MP", "PTS", "TRB",
                "AST", "STL", "BLK", "FG%", "3P%", "TS%", "PER", "WS", "BPM"]
NDL_FIELDS = ["Player", "Team", "Pos", "G", "MP", "NDL_Score", "NDL_Weight",
              "PTS", "TRB", "AST", "NDL_Production", "NDL_Efficiency",
              "NDL_Playmaking", "NDL_Defense", "NDL_WinImpact"]


def pick(p, fields):
    return {k: p.get(k) for k in fields}


stats_data = [pick(p, STATS_FIELDS) for p in players]
ndl_data = [pick(p, NDL_FIELDS) for p in players]

for name, payload, fields in [
    ("stats_template.html", stats_data, "stats.html"),
    ("ndl_template.html", ndl_data, "ndl-score.html"),
]:
    template = (BASE / name).read_text(encoding="utf-8")
    json_str = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    out = template.replace("__PLAYER_DATA__", json_str)
    out_path = OUT_DIR / fields
    out_path.write_text(out, encoding="utf-8")
    print(f"{out_path} 생성 완료 ({len(payload)}행, {len(out)/1024:.0f} KB)")
