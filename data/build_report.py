"""상세 보고서 docs/REPORT.md를 내려받아 읽을 PDF로 만든다. → docs/NBA-DATA-LAB-report.pdf

    pip install markdown
    python data/build_report.py

마크다운을 HTML로 바꿔 사이트와 같은 글꼴(Pretendard)·색(open-color)을 입히고, 설치된
Chrome(또는 Edge)의 헤드리스 인쇄로 A4 PDF를 만든다. 글꼴은 CDN에서 받으므로 인터넷이 필요하다.
"""
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import markdown

ROOT = Path(__file__).parent.parent
SRC = ROOT / "docs" / "REPORT.md"
OUT = ROOT / "docs" / "NBA-DATA-LAB-report.pdf"

BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "chromium", "chromium-browser",
]

# 색은 모두 open-color 단계 (gray 0·1·3·6·7·9, orange 0·9)
CSS = """
@page { size: A4; margin: 18mm 16mm 18mm 16mm; }
html { font-family: "Pretendard Variable", "Pretendard Std Variable"; color: #212529; }
body { font-size: 10pt; line-height: 1.7; word-break: keep-all; }
h1 { font-size: 20pt; margin: 0 0 6pt; letter-spacing: -0.01em; }
h2 { font-size: 14pt; margin: 22pt 0 6pt; padding-bottom: 4pt; border-bottom: 1.5pt solid #212529;
     break-after: avoid; }
h3 { font-size: 11.5pt; margin: 16pt 0 4pt; break-after: avoid; }
p, li { margin: 0 0 6pt; }
ul, ol { padding-left: 16pt; margin: 0 0 8pt; }
a { color: #d9480f; text-decoration: none; }
strong { font-weight: 700; }
code { font-size: 8.8pt; background: #f1f3f5; padding: 0 3pt; border-radius: 2pt; }
pre { background: #f8f9fa; border: 0.75pt solid #dee2e6; padding: 8pt 10pt; font-size: 8.5pt;
      white-space: pre-wrap; break-inside: avoid; }
pre code { background: none; padding: 0; }
blockquote { margin: 10pt 0; padding: 6pt 12pt; border-left: 3pt solid #d9480f; background: #fff4e6;
             break-inside: avoid; }
blockquote p { margin: 0 0 3pt; }
table { border-collapse: collapse; width: 100%; margin: 6pt 0 12pt; font-size: 9pt;
        font-variant-numeric: tabular-nums; break-inside: avoid; }
th { text-align: left; font-weight: 600; color: #495057; border-bottom: 0.75pt solid #212529;
     padding: 4pt 6pt; }
td { border-bottom: 0.5pt solid #dee2e6; padding: 4pt 6pt; vertical-align: top; }
tr:first-child th { border-top: 1.5pt solid #212529; }
em { color: #495057; }
hr { border: none; border-top: 0.75pt solid #dee2e6; margin: 14pt 0; }
"""


def find_browser():
    for b in BROWSERS:
        if os.path.isfile(b) or shutil.which(b):
            return shutil.which(b) or b
    raise SystemExit("Chrome 또는 Edge를 찾지 못했습니다. BROWSERS 목록에 경로를 추가하세요.")


def main():
    body = markdown.markdown(SRC.read_text(encoding="utf-8"),
                             extensions=["tables", "sane_lists", "fenced_code"])
    html = f"""<!DOCTYPE html><html lang="ko"><head><meta charset="utf-8">
<title>NBA DATA LAB 상세 보고서</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-dynamic-subset.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/variable/pretendardvariable-std-dynamic-subset.min.css">
<style>{CSS}</style></head><body>{body}</body></html>"""
    with tempfile.TemporaryDirectory() as tmp:
        page = Path(tmp) / "report.html"
        page.write_text(html, encoding="utf-8")
        subprocess.run([find_browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                        "--virtual-time-budget=15000", f"--print-to-pdf={OUT}", page.as_uri()],
                       check=True, capture_output=True)
    print(f"저장 {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
