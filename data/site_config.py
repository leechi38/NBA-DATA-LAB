"""배포 대상별 페이지 간 링크 주소.

같은 템플릿을 두 곳에 배포한다.
  artifact — claude.ai 아티팩트. 페이지마다 독립된 URL이라 절대주소로 연결한다.
  web      — 정적 호스팅(Vercel 등). 같은 도메인 안이라 상대경로를 쓴다.
"""

ARTIFACT_URLS = {
    "__HOME_URL__": "https://claude.ai/code/artifact/18a37a89-066f-4b51-a5b2-61b174abbe14",
    "__STATS_URL__": "https://claude.ai/code/artifact/3d9d87f0-73d5-4715-8b0c-59fa08b33163",
    "__NDL_URL__": "https://claude.ai/code/artifact/a2668865-848a-4670-a365-1a5f2bffce07",
    "__PLAYER_URL__": "https://claude.ai/artifact/2ZnLftGFrGrc9bJJY1rEt6",
    "__SCATTER_URL__": "https://claude.ai/code/artifact/8d314bec-60df-47a7-a230-ec902e645663",
    "__GAME_URL__": "https://claude.ai/code/artifact/7e46c135-6b01-4457-8702-1d11158912d0",
}

# vercel.json의 cleanUrls 설정으로 .html 확장자 없이 접근한다.
WEB_URLS = {
    "__HOME_URL__": "/",
    "__STATS_URL__": "/stats",
    "__NDL_URL__": "/ndl",
    "__PLAYER_URL__": "/player",
    "__SCATTER_URL__": "/scatter",
    "__GAME_URL__": "/game",
}

# 템플릿 → 웹 배포 시 파일명
WEB_FILENAMES = {
    "home": "index.html",
    "stats": "stats.html",
    "ndl": "ndl.html",
    "player": "player.html",
    "scatter": "scatter.html",
    "game": "game.html",
}
