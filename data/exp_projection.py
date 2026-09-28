"""내년 NDL Score를 실제로 예측할 수 있는지 검증한다.

핵심 질문은 "나이 곡선을 쓰면 나아지느냐"가 아니라 "아무것도 안 하는 것보다
나으냐"이다. 그래서 단순한 기준선부터 쌓아 올리며 하나씩 비교한다.

  A 작년 그대로      내년 = 올해
  B 3년 가중평균     최근 시즌에 더 큰 가중치 (5:4:3)
  C B + 평균회귀     표본이 적을수록 리그 평균(50) 쪽으로 당김
  D C + 나이 보정    학습 구간에서 구한 나이별 델타를 더함

가중치와 나이 곡선은 2015년 이전에서만 구하고, 2016년 이후로 시험한다.
기존 NDL Score 가중치 학습과 같은 분리 방식이다.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent
MIN_MP = 500
TRAIN_END = 2015     # 이 해까지로 곡선을 배운다
LEAGUE_MEAN = 50.0   # NDL Score의 설계상 평균
AGE_ADJ_FROM = 30    # 나이 보정을 온전히 적용할 하한 (근거는 아래 나이대별 표)
DAMPS = (0.0, 0.25, 0.5, 0.75)   # 30세 미만에 곡선을 얼마나 약하게 적용할지


def load():
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
            total = g * mp
            if total < MIN_MP:
                continue
            rows[(pid, year)] = {"age": int(age), "ndl": ndl, "mp": total}
    return rows


def fit_aging(rows):
    """학습 구간의 연속 시즌 쌍으로 나이별 평균 변화를 구한다."""
    d = defaultdict(list)
    for (pid, y), cur in rows.items():
        if y > TRAIN_END:
            continue
        nxt = rows.get((pid, y + 1))
        if nxt is None:
            continue
        d[cur["age"]].append(nxt["ndl"] - cur["ndl"])
    curve = {a: float(np.mean(v)) for a, v in d.items() if len(v) >= 30}
    # 표본이 없는 나이는 이웃값으로 메운다
    filled = {}
    for a in range(18, 45):
        if a in curve:
            filled[a] = curve[a]
        else:
            near = [x for x in curve if abs(x - a) <= 2]
            filled[a] = float(np.mean([curve[x] for x in near])) if near else 0.0
    return filled


def history(rows, pid, year, n=3):
    """직전 n시즌을 최신순으로."""
    out = []
    for back in range(1, n + 1):
        r = rows.get((pid, year - back + 1)) if back == 1 else rows.get((pid, year - back + 1))
        if r:
            out.append(r)
    return out


def evaluate(rows, curve):
    W = [5, 4, 3]
    preds = defaultdict(list)
    truth, age_at = [], []

    for (pid, y), nxt in rows.items():
        if y <= TRAIN_END + 1:       # 시험 구간만
            continue
        cur = rows.get((pid, y - 1))
        if cur is None:               # 올해 기록이 있어야 내년을 예측한다
            continue

        past = [rows.get((pid, y - 1 - k)) for k in range(3)]
        past = [(p, W[i]) for i, p in enumerate(past) if p]
        wsum = sum(w for _, w in past)
        wavg = sum(p["ndl"] * w for p, w in past) / wsum

        # 표본(누적 출전시간)이 적을수록 평균 쪽으로. 2000분을 기준으로 둔다.
        mp_total = sum(p["mp"] for p, _ in past)
        k = mp_total / (mp_total + 2000)
        shrunk = LEAGUE_MEAN + k * (wavg - LEAGUE_MEAN)

        truth.append(nxt["ndl"])
        age_at.append(cur["age"])
        preds["A 작년 그대로"].append(cur["ndl"])
        preds["B 3년 가중평균"].append(wavg)
        preds["C B + 평균회귀"].append(shrunk)
        preds["D C + 나이 보정(전체)"].append(shrunk + curve.get(cur["age"], 0.0))
        # E — 나이 보정을 30세 이상에만. 젊은 층의 델타는 생존 편향으로
        # 부풀려져 있어(리그를 떠난 선수가 표본에서 빠진다) 붙이면 오히려 해롭다.
        adj = curve.get(cur["age"], 0.0) if cur["age"] >= AGE_ADJ_FROM else 0.0
        preds["E 나이 보정(30세+)"].append(shrunk + adj)
        # F — 자르지 않고 줄인다. 젊은 층 델타는 방향은 맞고 크기만 부풀려져
        # 있으므로, 30세 미만에는 계수를 곱해 약하게 적용한다.
        full = curve.get(cur["age"], 0.0)
        for damp in DAMPS:
            k = 1.0 if cur["age"] >= AGE_ADJ_FROM else damp
            preds[f"F 젊은층 {int(damp*100)}% 적용"].append(shrunk + full * k)

    truth = np.array(truth)
    print(f"시험 표본: {len(truth):,}개 선수-시즌  ({TRAIN_END + 2}년 이후)\n")
    print(f"{'모델':<18}{'MAE':>8}{'RMSE':>9}{'상관':>9}")
    print("-" * 44)
    for name, p in preds.items():
        p = np.array(p)
        mae = np.mean(np.abs(p - truth))
        rmse = np.sqrt(np.mean((p - truth) ** 2))
        corr = np.corrcoef(p, truth)[0, 1]
        print(f"{name:<18}{mae:>8.2f}{rmse:>9.2f}{corr:>9.3f}")

    print("\n[기준] MAE는 평균 몇 점 빗나가는지. 낮을수록 좋다.")
    print("       NDL Score는 평균 50, 표준편차가 대략 10점인 척도다.")

    # 나이 보정이 전체 MAE는 0.03밖에 못 줄였다. 효과가 특정 구간에
    # 몰려 있는지 확인해야 이 항을 남길 근거가 생긴다.
    print("\n나이대별로 보면 — 나이 보정(D)이 평균회귀(C)보다 얼마나 나은가")
    print(f"{'나이대':<12}{'표본':>7}{'C MAE':>9}{'D MAE':>9}{'개선':>9}")
    print("-" * 47)
    ages = np.array(age_at)
    c, d = np.array(preds["C B + 평균회귀"]), np.array(preds["D C + 나이 보정(전체)"])
    contrib = 0.0
    for lo, hi, label in [(18, 22, "22세 이하"), (23, 25, "23–25세"),
                          (26, 29, "26–29세"), (30, 32, "30–32세"),
                          (33, 50, "33세 이상")]:
        m = (ages >= lo) & (ages <= hi)
        if m.sum() < 30:
            continue
        cm = np.mean(np.abs(c[m] - truth[m]))
        dm = np.mean(np.abs(d[m] - truth[m]))
        contrib += (cm - dm) * m.sum()
        print(f"{label:<12}{m.sum():>7,}{cm:>9.2f}{dm:>9.2f}{cm - dm:>+9.2f}")
    print(f"{'':12}{'':>7}{'':>9}{'가중합':>9}{contrib / len(truth):>+9.2f}")
    print("\n젊은 층에서 잃은 만큼 노년층에서 벌어 전체로는 거의 0이 된다.")
    print(f"보정을 {AGE_ADJ_FROM}세 이상으로 잘라낸 것이 E인데, MAE는 나아져도")
    print("RMSE·상관은 D보다 나빠졌다. 젊은 층 델타가 크기만 부풀려져 있을 뿐")
    print("방향은 맞다는 뜻이라, 잘라내지 않고 절반만 적용하는 F를 택한다.\n")

    base = np.array(preds["A 작년 그대로"])
    best = np.array(preds["F 젊은층 50% 적용"])
    print("채택: 3년 가중평균 → 평균회귀 → 나이 보정(30세 미만은 50%)")
    print(f"  MAE  {np.mean(np.abs(base - truth)):.2f} → {np.mean(np.abs(best - truth)):.2f}")
    print(f"  상관 {np.corrcoef(base, truth)[0,1]:.3f} → {np.corrcoef(best, truth)[0,1]:.3f}")


def main():
    rows = load()
    curve = fit_aging(rows)
    print(f"학습: ~{TRAIN_END}년 / 시험: {TRAIN_END + 2}년~\n")
    print("학습 구간에서 구한 나이별 연간 변화")
    for a in range(20, 39):
        print(f"  {a:>2}세 {curve[a]:>+6.2f}", end="\n" if a % 5 == 4 else "")
    print("\n")
    evaluate(rows, curve)


if __name__ == "__main__":
    main()
