"""산점도 페이지에 쓸 선수 헤드샷 이미지를 basketball-reference에서 내려받는다.

너무 표본이 적은 선수(15경기 미만)는 산점도 자체에서 제외하므로 다운로드 대상에서도 뺀다.
이미 받은 파일은 다시 받지 않는다(재실행 시 이어받기).
"""
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = Path(__file__).parent
DATA = json.loads((BASE / "players_2025_26.json").read_text(encoding="utf-8"))
VERSION = DATA["headshot_version"]
MIN_G = 15

IMG_DIR = BASE / "headshots"
IMG_DIR.mkdir(exist_ok=True)

targets = [p for p in DATA["players"] if p["G"] >= MIN_G]
print(f"다운로드 대상 {len(targets)}명 (전체 {len(DATA['players'])}명 중 {MIN_G}경기 이상)")

ok, failed = 0, []
for i, p in enumerate(targets, 1):
    bbref_id = p["BbrefId"]
    out_path = IMG_DIR / f"{bbref_id}.jpg"
    if out_path.exists():
        ok += 1
        continue
    url = f"https://www.basketball-reference.com/req/{VERSION}/images/headshots/{bbref_id}.jpg"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        data = urllib.request.urlopen(req, timeout=15).read()
        out_path.write_bytes(data)
        ok += 1
    except urllib.error.HTTPError as e:
        failed.append((p["Player"], bbref_id, str(e)))
    time.sleep(0.15)
    if i % 50 == 0:
        print(f"  {i}/{len(targets)} 처리...")

print(f"완료: 성공 {ok} / 실패 {len(failed)}")
if failed:
    for name, bid, err in failed[:20]:
        print("  실패:", name, bid, err)
