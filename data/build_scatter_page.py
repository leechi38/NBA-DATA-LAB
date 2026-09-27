"""scatter_template.html + all_seasons_scatter.json + headshots → pages/scatter.html

헤드샷은 최근 시즌 선수 위주로만 보유하고 있다(BbrefId 기준 매칭). 옛날 시즌
선수는 이미지 없이 기록만 표시되며, 템플릿이 fallback 아바타로 대체한다.
"""
import base64
import json
from pathlib import Path

BASE = Path(__file__).parent
OUT_DIR = BASE.parent / "pages"
OUT_DIR.mkdir(exist_ok=True)
HEADSHOT_DIR = BASE / "headshots"

data = json.loads((BASE / "all_seasons_scatter.json").read_text(encoding="utf-8"))

# 데이터에 실제로 등장하는 선수의 이미지만 임베드 (쓰지 않는 이미지로 용량 낭비 방지)
used_ids = {p["BbrefId"] for season in data["scatter"].values()
            for p in season if p.get("BbrefId")}

images = {}
if HEADSHOT_DIR.exists():
    for img_path in HEADSHOT_DIR.glob("*.jpg"):
        bbref_id = img_path.stem
        if bbref_id not in used_ids:
            continue
        b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
        images[bbref_id] = f"data:image/jpeg;base64,{b64}"

template = (BASE / "scatter_template.html").read_text(encoding="utf-8")
out = template.replace(
    "__SEASON_DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))
).replace(
    "__IMAGE_DATA__", json.dumps(images, separators=(",", ":"))
)

out_path = OUT_DIR / "scatter.html"
out_path.write_text(out, encoding="utf-8")
print(f"{out_path} 생성 완료 ({len(data['seasons'])}개 시즌, "
      f"이미지 {len(images)}개, {len(out)/1024/1024:.2f} MB)")
