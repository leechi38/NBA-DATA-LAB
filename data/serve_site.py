"""site/ 폴더를 로컬에서 미리보기 (배포 전 확인용).

Vercel의 cleanUrls(확장자 없는 주소)를 흉내 내므로, 배포 후와 같은 주소로 돌아간다.
    python data/serve_site.py      → http://localhost:8000
"""
import functools
import http.server
import socketserver
import sys
from pathlib import Path

SITE = Path(__file__).parent.parent / "site"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000


class CleanUrlHandler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        local = super().translate_path(path)
        p = Path(local)
        # /stats 처럼 확장자 없는 경로면 .html을 붙여 찾아본다 (Vercel cleanUrls 동작)
        if not p.exists() and not p.suffix:
            candidate = p.with_suffix(".html")
            if candidate.exists():
                return str(candidate)
        return local

    def log_message(self, fmt, *args):
        sys.stderr.write(f"  {self.address_string()} {fmt % args}\n")


def main():
    if not SITE.exists():
        print(f"site/ 폴더가 없습니다. 먼저 build_site.py를 실행하세요.")
        return
    handler = functools.partial(CleanUrlHandler, directory=str(SITE))
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", PORT), handler) as httpd:
        print(f"미리보기: http://localhost:{PORT}  (Ctrl+C로 종료)")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
