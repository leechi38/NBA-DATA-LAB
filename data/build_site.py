"""정적 웹사이트(Vercel 등) 배포용 site/ 폴더를 통째로 생성한다.

아티팩트 배포본과 다른 점
  1. 페이지 간 링크가 상대경로(/stats, /player …)
  2. 선수 얼굴 이미지를 HTML에 base64로 박지 않고 site/img/ 에 파일로 내보낸다.
     아티팩트는 CSP가 외부 이미지를 막아 어쩔 수 없이 인라인했지만, 일반 웹에서는
     파일로 두는 편이 용량도 작고(base64는 약 33% 팽창) 브라우저 캐싱도 된다.
  3. vercel.json으로 확장자 없는 주소(/stats)를 쓰게 한다.
"""
import json
import shutil
from pathlib import Path

from site_config import WEB_URLS

BASE = Path(__file__).parent
SITE = BASE.parent / "site"
HEADSHOT_DIR = BASE / "headshots"


def apply_urls(html: str) -> str:
    for placeholder, url in WEB_URLS.items():
        html = html.replace(placeholder, url)
    return html


def write(name: str, html: str):
    path = SITE / name
    path.write_text(html, encoding="utf-8")
    return path.stat().st_size


def main():
    if SITE.exists():
        shutil.rmtree(SITE)
    SITE.mkdir(parents=True)
    (SITE / "img").mkdir()

    sizes = {}

    # --- 홈 ---
    sizes["index.html"] = write("index.html",
                                 apply_urls((BASE / "home_template.html").read_text(encoding="utf-8")))

    # --- 시즌 선택형 데이터 페이지 3종 ---
    for name, template, data_file, placeholder in [
        ("stats.html", "stats_multiseason_template.html", "all_seasons_stats.json", "__SEASON_DATA__"),
        ("ndl.html", "ndl_multiseason_template.html", "all_seasons_ndl.json", "__SEASON_DATA__"),
        ("player.html", "player_template.html", "careers.json", "__CAREERS_DATA__"),
    ]:
        html = (BASE / template).read_text(encoding="utf-8")
        html = html.replace(placeholder, (BASE / data_file).read_text(encoding="utf-8"))
        sizes[name] = write(name, apply_urls(html))

    # --- 포지셔닝: 이미지를 파일로 분리 ---
    scatter_data = json.loads((BASE / "all_seasons_scatter.json").read_text(encoding="utf-8"))
    used_ids = {p["BbrefId"] for season in scatter_data["scatter"].values()
                for p in season if p.get("BbrefId")}
    image_map = {}
    copied = 0
    if HEADSHOT_DIR.exists():
        for img in HEADSHOT_DIR.glob("*.jpg"):
            if img.stem not in used_ids:
                continue
            shutil.copy2(img, SITE / "img" / img.name)
            image_map[img.stem] = f"/img/{img.name}"
            copied += 1

    html = (BASE / "scatter_template.html").read_text(encoding="utf-8")
    html = html.replace("__SEASON_DATA__",
                        json.dumps(scatter_data, ensure_ascii=False, separators=(",", ":")))
    html = html.replace("__IMAGE_DATA__", json.dumps(image_map, separators=(",", ":")))
    sizes["scatter.html"] = write("scatter.html", apply_urls(html))

    # --- 시뮬레이션 ---
    html = (BASE / "game_template.html").read_text(encoding="utf-8")
    html = html.replace("__SEASON_DATA__",
                        (BASE / "all_seasons_game.json").read_text(encoding="utf-8"))
    sizes["game.html"] = write("game.html", apply_urls(html))

    # --- Vercel 설정: 확장자 없는 주소 + 정적 자산 캐싱 ---
    (SITE / "vercel.json").write_text(json.dumps({
        "cleanUrls": True,
        "trailingSlash": False,
        "headers": [{
            "source": "/img/(.*)",
            "headers": [{"key": "Cache-Control", "value": "public, max-age=31536000, immutable"}],
        }],
    }, indent=2), encoding="utf-8")

    total = sum(sizes.values())
    img_total = sum(f.stat().st_size for f in (SITE / "img").glob("*"))
    print(f"site/ 생성 완료 → {SITE}")
    for name, size in sizes.items():
        print(f"  {size/1024/1024:>7.2f} MB  {name}")
    print(f"  {img_total/1024/1024:>7.2f} MB  img/ ({copied}개 파일)")
    print(f"  {'-'*30}")
    print(f"  {(total + img_total)/1024/1024:>7.2f} MB  합계")

    # 링크 누락 점검
    leftovers = []
    for f in SITE.glob("*.html"):
        text = f.read_text(encoding="utf-8")
        if "__" in text and any(k in text for k in WEB_URLS):
            leftovers.append(f.name)
        if "claude.ai" in text:
            leftovers.append(f"{f.name} (claude.ai 링크 잔존)")
    print(f"\n링크 점검: {'문제 없음' if not leftovers else leftovers}")


if __name__ == "__main__":
    main()
