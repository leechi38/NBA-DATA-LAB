"""game_template.html + all_seasons_game.json → pages/game.html

슛 구역별 데이터가 있는 시즌(1996-97~)만 포함한다. 엔진이 구역별 슛 선택으로
포제션을 계산하므로, 구역 분포가 균등값으로 채워지는 그 이전 시즌은 시뮬레이션
결과가 의미를 갖지 못한다.
"""
import json
from pathlib import Path

from site_config import apply_base_css

BASE = Path(__file__).parent
OUT_DIR = BASE.parent / "pages"
OUT_DIR.mkdir(exist_ok=True)

data = json.loads((BASE / "all_seasons_game.json").read_text(encoding="utf-8"))

template = (BASE / "game_template.html").read_text(encoding="utf-8")
out = template.replace(
    "__SEASON_DATA__", json.dumps(data, ensure_ascii=False, separators=(",", ":"))
)

out_path = OUT_DIR / "game.html"
out = apply_base_css(out)
out_path.write_text(out, encoding="utf-8")
seasons = data["seasons"]
print(f"{out_path} 생성 완료 ({len(seasons)}개 시즌 {seasons[0]}~{seasons[-1]}, "
      f"{len(out)/1024/1024:.2f} MB)")
