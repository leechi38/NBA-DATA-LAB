"""scatter_template.html + players_2025_26.json + headshots/*.jpg 를 합쳐
pages/scatter.html 최종 파일을 생성한다."""
import base64
import json
from pathlib import Path

BASE = Path(__file__).parent
OUT_DIR = BASE.parent / "pages"
OUT_DIR.mkdir(exist_ok=True)
HEADSHOT_DIR = BASE / "headshots"
MIN_G = 15  # 표본이 불안정한 선수는 산점도에서 제외

data = json.loads((BASE / "players_2025_26.json").read_text(encoding="utf-8"))
players = [p for p in data["players"] if p["G"] >= MIN_G]

SCATTER_FIELDS = ["Player", "Team", "Pos", "PosGroup", "BbrefId", "MP", "G",
                  "FGA", "FG%", "3PA", "3P%", "eFG%", "TS%", "PTS", "TRB", "AST",
                  "TOV", "USG%", "PER", "NDL_Score"]


def pick(p, fields):
    return {k: p.get(k) for k in fields}


scatter_players = [pick(p, SCATTER_FIELDS) for p in players]

images = {}
for p in players:
    bbref_id = p["BbrefId"]
    img_path = HEADSHOT_DIR / f"{bbref_id}.jpg"
    if img_path.exists():
        b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
        images[bbref_id] = f"data:image/jpeg;base64,{b64}"

template = (BASE / "scatter_template.html").read_text(encoding="utf-8")
out = template.replace(
    "__PLAYER_DATA__", json.dumps(scatter_players, ensure_ascii=False, separators=(",", ":"))
).replace(
    "__IMAGE_DATA__", json.dumps(images, separators=(",", ":"))
)

out_path = OUT_DIR / "scatter.html"
out_path.write_text(out, encoding="utf-8")
print(f"{out_path} 생성 완료 ({len(scatter_players)}명, 이미지 {len(images)}개, {len(out)/1024/1024:.2f} MB)")
