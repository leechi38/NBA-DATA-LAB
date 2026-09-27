"""stats_multiseason_template.html + all_seasons_stats.json 을 합쳐 발행용 stats.html을 만든다."""
from pathlib import Path

from site_config import apply_base_css

BASE = Path(__file__).parent
OUT_DIR = BASE.parent / "pages"
OUT_DIR.mkdir(exist_ok=True)

URLS = {
    "__HOME_URL__": "https://claude.ai/code/artifact/18a37a89-066f-4b51-a5b2-61b174abbe14",
    "__NDL_URL__": "https://claude.ai/code/artifact/a2668865-848a-4670-a365-1a5f2bffce07",
    "__PLAYER_URL__": "https://claude.ai/artifact/2ZnLftGFrGrc9bJJY1rEt6",
    "__SCATTER_URL__": "https://claude.ai/code/artifact/8d314bec-60df-47a7-a230-ec902e645663",
    "__GAME_URL__": "https://claude.ai/code/artifact/7e46c135-6b01-4457-8702-1d11158912d0",
}

template = (BASE / "stats_multiseason_template.html").read_text(encoding="utf-8")
season_json = (BASE / "all_seasons_stats.json").read_text(encoding="utf-8")

out = template.replace("__SEASON_DATA__", season_json)
for placeholder, url in URLS.items():
    out = out.replace(placeholder, url)

out_path = OUT_DIR / "stats.html"
out = apply_base_css(out)
out_path.write_text(out, encoding="utf-8")
print(f"{out_path} 생성 완료 ({len(out)/1024/1024:.2f} MB)")
