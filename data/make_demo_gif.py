"""README 맨 위의 데모 GIF를 docs/screens/*.png 스크린샷으로 만든다. → docs/demo.gif

    python data/make_demo_gif.py

스크린샷은 크기가 제각각이라 같은 크기의 바탕(사이트 배경색 open-color gray 0)에 맞춰
넣는다. 한 장을 HOLD_MS 동안 보여주고, FADE_STEPS장의 중간 프레임으로 다음 장에 겹쳐
넘긴다. 읽을 시간은 주되 멈춘 것처럼 보이지 않게 3.5초로 잡았다.
"""
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).parent.parent
SCREENS = ROOT / "docs" / "screens"
OUT = ROOT / "docs" / "demo.gif"

WIDTH, HEIGHT = 1100, 790      # GitHub 본문 폭(약 880px)보다 조금 크게
PAD = 24
PAPER = (0xF8, 0xF9, 0xFA)     # open-color gray 0, 사이트 배경과 같다
HOLD_MS = 3500
FADE_STEPS = 4
FADE_MS = 100


def fit(path):
    """비율을 지켜 캔버스 안에 맞추고 가운데 둔다."""
    img = Image.open(path).convert("RGB")
    scale = min((WIDTH - 2 * PAD) / img.width, (HEIGHT - 2 * PAD) / img.height)
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
    canvas = Image.new("RGB", (WIDTH, HEIGHT), PAPER)
    canvas.paste(img, ((WIDTH - img.width) // 2, (HEIGHT - img.height) // 2))
    return canvas


def main():
    slides = [fit(p) for p in sorted(SCREENS.glob("*.png"))]
    frames, durations = [], []
    for i, cur in enumerate(slides):
        frames.append(cur)
        durations.append(HOLD_MS)
        nxt = slides[(i + 1) % len(slides)]
        for k in range(1, FADE_STEPS + 1):
            frames.append(Image.blend(cur, nxt, k / (FADE_STEPS + 1)))
            durations.append(FADE_MS)
    # 프레임마다 팔레트를 따로 뽑으면 화면 간 색이 흔들려, 모든 장을 이어 붙인 한 장에서
    # 256색 팔레트를 하나 만들어 같이 쓴다.
    sheet = Image.new("RGB", (WIDTH, HEIGHT * len(slides)))
    for i, s in enumerate(slides):
        sheet.paste(s, (0, HEIGHT * i))
    palette = sheet.quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    quantized = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
    quantized[0].save(OUT, save_all=True, append_images=quantized[1:], duration=durations,
                      loop=0, optimize=True, disposal=1)
    total = sum(durations) / 1000
    print(f"{len(slides)}장, 프레임 {len(frames)}개, 한 바퀴 {total:.1f}초, "
          f"{OUT.stat().st_size / 1024 / 1024:.2f} MB → {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
