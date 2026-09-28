"""나이에 따른 NDL Score 변화를 재본다 — 내년 성적 예측이 가능한지 타당성 점검.

두 가지를 확인한다.
  1. 연속 시즌 쌍이 나이대별로 몇 개나 되는가 (표본이 있어야 곡선을 그린다)
  2. 델타법으로 그린 나이 곡선이 생존 편향에 얼마나 오염되는가

생존 편향: 나이 a에서 못한 선수는 a+1에 리그를 떠난다. 남은 선수만 보면
"늙어도 안 떨어진다"는 결론이 나온다. 이게 델타법의 고전적 함정이라,
곡선을 그리기 전에 그 크기부터 재야 한다.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
MIN_MP = 500  # 총 출전시간 하한. 이 아래는 표본이 너무 작아 추이가 잡음이다.


def load():
    """(bbrefId, 시즌끝연도) -> {age, ndl, mp} 로 펼친다."""
    rows = {}
    for f in sorted((BASE / "seasons").glob("players_*.json"),
                    key=lambda p: int(p.stem.split("_")[1])):
        d = json.loads(f.read_text(encoding="utf-8"))
        year = int(d["season"].split("-")[0]) + 1
        for p in d["players"]:
            pid, age, ndl = p.get("BbrefId"), p.get("Age"), p.get("NDL_Score")
            g, mp = p.get("G"), p.get("MP")
            if not pid or age is None or ndl is None or not g or mp is None:
                continue
            total_mp = g * mp
            if total_mp < MIN_MP:
                continue
            rows[(pid, year)] = {"age": int(age), "ndl": ndl, "mp": total_mp}
    return rows


def main():
    rows = load()
    print(f"분석 대상 선수-시즌: {len(rows):,}개  (총 출전 {MIN_MP}분 이상)")

    years = sorted({y for _, y in rows})
    print(f"시즌 범위: {years[0]} ~ {years[-1]}\n")

    # --- 1) 연속 시즌 쌍 ---
    pairs = defaultdict(list)          # 나이 -> [(올해 NDL, 내년 NDL, 올해 출전시간)]
    survived = defaultdict(lambda: [0, 0])  # 나이 -> [다음 시즌 생존, 전체]

    for (pid, y), cur in rows.items():
        nxt = rows.get((pid, y + 1))
        survived[cur["age"]][1] += 1
        if nxt is None:
            continue
        survived[cur["age"]][0] += 1
        pairs[cur["age"]].append((cur["ndl"], nxt["ndl"], cur["mp"]))

    total_pairs = sum(len(v) for v in pairs.values())
    print(f"연속 두 시즌을 모두 뛴 쌍: {total_pairs:,}개\n")

    print("나이  쌍 개수   델타(생존자만)   잔류율   떠난 선수 평균NDL  남은 선수 평균NDL")
    print("-" * 82)

    curve = {}
    for age in range(19, 41):
        pr = pairs.get(age, [])
        kept, tot = survived[age]
        if tot < 30:
            continue
        # 생존자 기준 델타
        delta = np.mean([b - a for a, b, _ in pr]) if len(pr) >= 20 else None
        # 떠난 선수 vs 남은 선수의 올해 성적 — 편향의 크기
        stay = [r["ndl"] for (pid, y), r in rows.items()
                if r["age"] == age and (pid, y + 1) in rows]
        gone = [r["ndl"] for (pid, y), r in rows.items()
                if r["age"] == age and (pid, y + 1) not in rows and y < years[-1]]
        if delta is not None:
            curve[age] = delta
        print(f"{age:>3}  {len(pr):>7,}   "
              f"{('%+.2f' % delta) if delta is not None else '  —  ':>13}   "
              f"{kept/tot*100:>5.1f}%   "
              f"{np.mean(gone) if gone else float('nan'):>15.1f}  "
              f"{np.mean(stay) if stay else float('nan'):>17.1f}")

    print("\n[읽는 법] '떠난 선수'가 '남은 선수'보다 점수가 낮으면, 생존자만 본 델타는")
    print("          실제 노화보다 낙관적이다. 그 격차가 생존 편향의 크기다.")

    # --- 2) 누적 곡선 ---
    print("\n나이별 누적 변화 (25세를 0으로 두고 델타를 이어붙임)")
    print("-" * 50)
    ages = sorted(curve)
    cum, acc = {}, 0.0
    for a in ages:
        cum[a] = acc
        acc += curve[a]
    base = cum.get(25, 0.0)
    for a in ages:
        v = cum[a] - base
        bar = "#" * int(abs(v) * 2)
        print(f"  {a:>2}세  {v:>+6.2f}  {bar}")


if __name__ == "__main__":
    main()
