"""선수별 슛 차트(개별 슛 x/y 좌표) 페이지를 긁어 좌/중앙/우 슛 비중을 계산한다.

basketball-reference 선수 개인 페이지의 '/shooting/{시즌}' 서브페이지에는
그 시즌 모든 슛의 코트 좌표(top/left px, 500px 폭 하프코트 기준)가
tooltip으로 들어있다. left 좌표로 좌/중앙/우를 나눈다(center=250 확인됨,
Jokić 표본 median=242로 검증). 오래 걸리는 작업이라 이미 받은 선수는 건너뛴다.
"""
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = Path(__file__).parent
DATA = json.loads((BASE / "players_2025_26.json").read_text(encoding="utf-8"))
SEASON = 2026
MIN_G = 15  # 시뮬레이션 대상(zone/headshot과 동일 기준)

CACHE_DIR = BASE / ".cache" / "shot_sides"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = BASE / "shot_sides.json"

SHOT_PATTERN = re.compile(
    r'style="top:(-?\d+)px;left:(-?\d+)px;"\s*tip="([^"]*)"\s*class="tooltip (make|miss)"'
)
LEFT_MAX = 200   # left < 200 => 코트 왼쪽
RIGHT_MIN = 300  # left > 300 => 코트 오른쪽 (그 사이는 중앙)


def classify(left: int) -> str:
    if left < LEFT_MAX:
        return "Left"
    if left > RIGHT_MIN:
        return "Right"
    return "Center"


def fetch_shots(bbref_id: str) -> list[str] | None:
    cache_path = CACHE_DIR / f"{bbref_id}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    url = f"https://www.basketball-reference.com/players/{bbref_id[0]}/{bbref_id}/shooting/{SEASON}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        html = urllib.request.urlopen(req, timeout=20).read().decode("utf-8")
    except urllib.error.HTTPError:
        return None
    sides = [classify(int(left)) for _, left, _tip, _res in SHOT_PATTERN.findall(html)]
    cache_path.write_text(json.dumps(sides), encoding="utf-8")
    return sides


def main():
    targets = [p for p in DATA["players"] if p["G"] >= MIN_G]
    print(f"대상 {len(targets)}명")
    results = {}
    for i, p in enumerate(targets, 1):
        sides = fetch_shots(p["BbrefId"])
        if sides:
            n = len(sides)
            results[p["BbrefId"]] = {
                "n_shots": n,
                "Side_Left": round(sides.count("Left") / n, 3),
                "Side_Center": round(sides.count("Center") / n, 3),
                "Side_Right": round(sides.count("Right") / n, 3),
            }
        time.sleep(0.3)
        if i % 25 == 0:
            print(f"  {i}/{len(targets)}")

    OUT_PATH.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    print(f"완료: {len(results)}/{len(targets)} 저장 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
