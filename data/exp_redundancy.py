"""왜 '더 좋은 구성요소'가 '더 나쁜 지표'를 만들었는지 확인.

가설: 생산성 축이 이미 AST·STL·BLK·TRB를 절대 수치로 담고 있어, 플레이메이킹·수비기여를
      절대 수치로 바꾸면 세 축이 같은 정보를 중복 측정하게 된다. 개별 설명력이 올라가도
      합성 지표에 새로 더해주는 정보가 줄어 전체 성능은 떨어진다.

확인: 각 축과 생산성 축의 상관계수를 두 변형에서 비교한다.
"""
from pathlib import Path

import pandas as pd

BASE = Path(__file__).parent
players = pd.read_csv(BASE / "_exp_components.csv")

print("축 간 상관계수 (선수 단위, 전 시즌 통합)\n")
print(f"{'':28}{'생산성과의 상관':>16}")
pairs = [
    ("플레이메이킹 — 비율(%)", "playmaking_rate"),
    ("플레이메이킹 — 절대", "playmaking_abs"),
    ("수비기여 — 비율(%)", "defense_rate"),
    ("수비기여 — 절대", "defense_abs"),
]
for label, col in pairs:
    r = players["production"].corr(players[col])
    print(f"  {label:<26}{r:>16.3f}")

print("\n두 변형 사이의 상관 (같은 축의 비율 vs 절대)")
for axis in ["playmaking", "defense"]:
    r = players[f"{axis}_rate"].corr(players[f"{axis}_abs"])
    print(f"  {axis:<26}{r:>16.3f}")

print("\n해석용 — 효율성/승리기여와 생산성의 상관 (참고 기준선)")
for col in ["efficiency", "win_impact"]:
    r = players["production"].corr(players[col])
    print(f"  {col:<26}{r:>16.3f}")
