"""현대 농구(2016~2024) 완전 홀드아웃 검증.

가중치는 1952~2015의 홀수 연도 시즌으로만 학습했다. 2016년 이후 시즌은
학습에도, 그때의 짝수 연도 검증에도 쓰이지 않은 완전한 미지의 구간이다.
3점 혁명 이후 경기 양상이 크게 바뀐 시기이므로, 여기서도 개선이 유지되는지가
'우연히 과거 데이터에 맞춘 것'인지 '실제 신호를 찾은 것'인지를 가른다.
"""
import numpy as np
import pandas as pd

from optimize_weights import (COMPONENTS, LABELS, ORIGINAL_WEIGHTS,
                              build_team_table, season_corr)

OPTIMIZED = np.array([0.090, 0.297, 0.093, 0.120, 0.400])  # optimize_weights2.py 채택안(B)

SPLITS = [
    ("학습 구간 (1952~2015 홀수)", lambda y: (y <= 2015) & (y % 2 == 1)),
    ("검증 구간 (1952~2015 짝수)", lambda y: (y <= 2015) & (y % 2 == 0)),
    ("현대 홀드아웃 (2016~2024)", lambda y: y >= 2016),
]


def main():
    table = build_team_table()
    print(f"전체: {len(table)}개 팀-시즌, {table['year_id'].nunique()}개 시즌\n")
    print(f"{'구간':<26}{'시즌':>5}{'기존':>9}{'최적화':>9}{'차이':>9}{'개선시즌':>10}")

    for name, pred in SPLITS:
        sub = table[pred(table["year_id"])]
        if sub.empty:
            continue
        r_old = season_corr(sub, ORIGINAL_WEIGHTS)
        r_new = season_corr(sub, OPTIMIZED)
        improved = int((r_new > r_old).sum())
        print(f"{name:<26}{sub['year_id'].nunique():>5}"
              f"{r_old.mean():>9.4f}{r_new.mean():>9.4f}"
              f"{r_new.mean() - r_old.mean():>+9.4f}"
              f"{f'{improved}/{len(r_old)}':>10}")

    modern = table[table["year_id"] >= 2016]
    r_old = season_corr(modern, ORIGINAL_WEIGHTS)
    r_new = season_corr(modern, OPTIMIZED)
    print("\n[현대 홀드아웃 시즌별 상세]")
    print(f"{'시즌':>6}{'기존':>9}{'최적화':>9}{'차이':>9}")
    for year in sorted(modern["year_id"].unique()):
        label = f"{year-1}-{str(year)[-2:]}"
        print(f"{label:>6}{r_old[year]:>9.4f}{r_new[year]:>9.4f}{r_new[year] - r_old[year]:>+9.4f}")


if __name__ == "__main__":
    main()
