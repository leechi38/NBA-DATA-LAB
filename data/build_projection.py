"""다음 시즌 NDL Score 예측값을 만들어 projection.json으로 내보낸다.

    예측 = b0 + (기량 구간별 기울기)×(최근 3시즌 5:4:3 가중평균)
              + b×(나이 델타, 30세 미만) + b×(나이 델타, 30세 이상)

평균회귀를 하나의 기울기로 두면 모든 선수에게 같은 비율을 적용하게 된다. 실제로는
등급마다 크게 다르다 — 직전 3시즌 평균이 80점 이상인 선수는 다음 시즌에 평균 쪽으로
6%만 내려오는 반면, 50점 미만은 54% 내려온다(후자는 상당 부분 생존 편향이다).
단일 기울기는 상위권을 두 배로 끌어내린다.

그래서 50/60/70/80점을 마디로 구간별 기울기를 따로 추정한다. 다만 자유롭게 두면
70-80 구간이 1.058, 즉 "평균에서 멀어진다"는 값을 내놓는다. 상위 구간의 표본이
얇아서(80점 이상 쌍이 역대 157개) 생기는 과적합이다. 그래서 두 가지를 제약한다.

    1) 구간이 올라갈수록 기울기가 줄지 않는다  (잘하는 선수일수록 덜 회귀)
    2) 어떤 구간도 기울기가 1을 넘지 않는다    (평균에서 멀어지지는 않는다)

가설을 제약으로 넣고, 그 안에서 데이터가 값을 고르게 하는 방식이다.

계수는 모두 학습 구간(~2015년)에서 최소제곱으로 추정한다. 처음에는 수축 계수를
mp/(mp+2000) 꼴로 손수 정했는데, 점수대별 편향을 재보니 80점대 선수를 5.7점,
90점대를 6.5점씩 낮게 예측하고 있었다. 상위권일수록 과도하게 끌어내린 것이다.
회귀로 추정하면 정의상 편향이 사라져, 같은 검증에서 ±0.6 안으로 들어온다.

나이 델타를 두 항으로 나눈 것도 같은 이유다. 젊은 구간의 곡선은 생존 편향으로
부풀려져 있는데(다음 시즌에 리그를 떠난 선수가 표본에서 빠진다), 얼마나 덜
믿을지를 손으로 정하는 대신 계수가 정하게 했다. 결과는 1.14 대 1.65로,
젊은 구간을 실제로 덜 신뢰한다.

홀드아웃(2016-17년~, 2,576개) 성능
    MAE  4.61 → 4.13     상관 0.730 → 0.761     (기준: 작년 성적 그대로)
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

BASE = Path(__file__).parent
OUT = BASE / "projection.json"

MIN_MP = 500          # 총 출전시간 하한. 이 아래는 시즌 간 추이가 잡음이다.
WEIGHTS = [5, 4, 3]   # 최근 시즌부터의 가중치
AGE_SPLIT = 30        # 나이 델타를 두 항으로 가르는 경계
TRAIN_END = 2015      # 계수 검증용 학습 구간 끝
MIN_AGE_PAIRS = 30    # 나이 델타를 신뢰할 최소 표본
KNOTS = [50, 60, 70, 80]   # 기량 구간의 마디


def load():
    rows, meta = {}, {}
    for f in sorted((BASE / "seasons").glob("players_*.json"),
                    key=lambda p: int(p.stem.split("_")[1])):
        d = json.loads(f.read_text(encoding="utf-8"))
        year = int(d["season"].split("-")[0]) + 1
        for p in d["players"]:
            pid, age, ndl = p.get("BbrefId"), p.get("Age"), p.get("NDL_Score")
            g, mp = p.get("G"), p.get("MP")
            if not pid or age is None or ndl is None or not g or mp is None:
                continue
            if g * mp < MIN_MP:
                continue
            rows[(pid, year)] = {"age": int(age), "ndl": ndl, "mp": g * mp}
            meta[pid] = {"name": p["Player"], "team": p.get("Team"),
                         "pos": p.get("Pos"), "season": d["season"]}
    return rows, meta


def aging_curve(rows, upto=None):
    """연속 두 시즌을 모두 뛴 선수들로 나이별 평균 변화를 구한다."""
    d = defaultdict(list)
    for (pid, y), cur in rows.items():
        if upto is not None and y > upto:
            continue
        nxt = rows.get((pid, y + 1))
        if nxt is not None:
            d[cur["age"]].append(nxt["ndl"] - cur["ndl"])
    raw = {a: float(np.mean(v)) for a, v in d.items() if len(v) >= MIN_AGE_PAIRS}
    lo, hi = min(raw), max(raw)
    # 관측 범위 밖(예: 41세)을 0으로 두면 "그 나이엔 노화가 없다"고 말하는 셈이라
    # 실제보다 후하게 예측된다. 양 끝은 가장 바깥 3개 나이의 평균으로 외삽한다.
    tail_hi = float(np.mean([raw[a] for a in sorted(raw)[-3:]]))
    tail_lo = float(np.mean([raw[a] for a in sorted(raw)[:3]]))
    filled = {}
    for a in range(18, 46):
        if a in raw:
            filled[a] = raw[a]
        elif a > hi:
            filled[a] = tail_hi
        elif a < lo:
            filled[a] = tail_lo
        else:  # 범위 안의 구멍은 이웃 ±2세로 메운다
            near = [x for x in raw if abs(x - a) <= 2]
            filled[a] = float(np.mean([raw[x] for x in near])) if near else tail_hi
    return filled, raw, {a: len(v) for a, v in d.items()}


def features(rows, curve, pid, year):
    """year 시즌까지의 기록으로 설명변수를 만든다."""
    past = [(rows.get((pid, year - k)), WEIGHTS[k]) for k in range(3)]
    past = [(r, w) for r, w in past if r]
    if not past:
        return None
    ws = sum(w for _, w in past)
    wavg = sum(r["ndl"] * w for r, w in past) / ws
    cur = past[0][0]
    raw_d = curve.get(cur["age"], 0.0)
    old = cur["age"] >= AGE_SPLIT
    # [절편, wavg, 마디별 추가 기울기…, 나이델타(젊음), 나이델타(30세+)]
    x = ([1.0, wavg] + [max(0.0, wavg - k) for k in KNOTS]
         + [0.0 if old else raw_d, raw_d if old else 0.0])
    return {
        "x": x, "wavg": wavg, "delta": raw_d, "cur": cur,
        "seasons": len(past), "mp": sum(r["mp"] for r, _ in past),
    }


def slopes_of(coef):
    """구간별 실효 기울기. 마디 계수가 누적된다."""
    out, s = [coef[1]], coef[1]
    for i in range(len(KNOTS)):
        s += coef[2 + i]
        out.append(s)
    return out


def fit(rows, curve, upto):
    """제약 최소제곱. 마디 계수 >= 0 (기울기 비감소), 기울기 <= 1."""
    X, y = [], []
    for (pid, yr), nxt in rows.items():
        f = features(rows, curve, pid, yr - 1)
        if f is None or (pid, yr - 1) not in rows or yr - 1 > upto:
            continue
        X.append(f["x"])
        y.append(nxt["ndl"])
    X, y = np.array(X), np.array(y)

    n = X.shape[1]
    lo = np.full(n, -np.inf)
    hi = np.full(n, np.inf)
    lo[2:2 + len(KNOTS)] = 0.0   # 마디 계수는 음수가 될 수 없다
    hi[1] = 1.0                  # 기본 기울기부터 1을 넘지 않는다
    coef = lsq_linear(X, y, bounds=(lo, hi)).x

    # 누적 기울기가 1을 넘으면 위쪽 마디부터 깎아 맞춘다
    over = max(slopes_of(coef)) - 1.0
    for i in range(len(KNOTS) - 1, -1, -1):
        if over <= 1e-9:
            break
        cut = min(coef[2 + i], over)
        coef[2 + i] -= cut
        over -= cut
    return coef, len(y)


def holdout_report(rows):
    """계수를 학습 구간에서만 뽑아 이후 구간으로 시험한다."""
    curve, _, _ = aging_curve(rows, upto=TRAIN_END)
    coef, n_train = fit(rows, curve, TRAIN_END)
    pred, base, truth = [], [], []
    for (pid, yr), nxt in rows.items():
        if yr - 1 <= TRAIN_END + 1:
            continue
        f = features(rows, curve, pid, yr - 1)
        if f is None or (pid, yr - 1) not in rows:
            continue
        pred.append(float(np.dot(coef, f["x"])))
        base.append(f["cur"]["ndl"])
        truth.append(nxt["ndl"])
    pred, base, truth = map(np.array, (pred, base, truth))
    return {
        "coef": [round(float(c), 3) for c in coef],
        "trainSize": n_train, "testSize": len(truth),
        "mae": round(float(np.mean(np.abs(pred - truth))), 2),
        "baselineMae": round(float(np.mean(np.abs(base - truth))), 2),
        "corr": round(float(np.corrcoef(pred, truth)[0, 1]), 3),
        "baselineCorr": round(float(np.corrcoef(base, truth)[0, 1]), 3),
    }


def main():
    rows, meta = load()

    report = holdout_report(rows)
    print("홀드아웃 검증 (계수는 ~2015년에서만 학습)")
    print(f"  학습 {report['trainSize']:,}개 / 시험 {report['testSize']:,}개")
    print(f"  MAE  {report['baselineMae']} → {report['mae']}")
    print(f"  상관 {report['baselineCorr']} → {report['corr']}\n")

    # 실제 예측은 전 구간으로 다시 적합한다. 성능 수치는 위 홀드아웃 것을 쓴다.
    curve, raw_curve, counts = aging_curve(rows)
    coef, _ = fit(rows, curve, upto=10**9)
    b = [round(float(c), 3) for c in coef]
    sl = slopes_of(coef)
    labels = ["50 미만"] + [f"{k}–{KNOTS[i+1]}" if i + 1 < len(KNOTS) else f"{k}+"
                          for i, k in enumerate(KNOTS)]
    print("기량 구간별 평균회귀율")
    for lab, s in zip(labels, sl):
        print(f"  {lab:<10}{(1 - s) * 100:>6.1f}%  (기울기 {s:.3f})")
    print(f"\n나이 델타 계수: 30세 미만 {b[-2]:.2f} / 30세 이상 {b[-1]:.2f}\n")

    last_year = max(y for _, y in rows)
    from_label = next(meta[pid]["season"] for (pid, y) in rows if y == last_year)
    to_label = f"{last_year}-{str(last_year + 1)[-2:]}"

    out = []
    for (pid, y) in rows:
        if y != last_year:
            continue
        f = features(rows, curve, pid, last_year)
        if f is None:
            continue
        cur = f["cur"]
        proj = float(np.dot(coef, f["x"]))
        m = meta[pid]
        out.append({
            "id": pid, "player": m["name"], "team": m["team"], "pos": m["pos"],
            "age": cur["age"], "nextAge": cur["age"] + 1,
            "current": round(cur["ndl"], 1),
            "projected": round(proj, 1),
            "change": round(proj - cur["ndl"], 1),
            "weightedAvg": round(f["wavg"], 1),
            "ageDelta": round(f["delta"], 2),
            "seasonsUsed": f["seasons"],
        })
    out.sort(key=lambda r: r["projected"], reverse=True)

    payload = {
        "fromSeason": from_label, "toSeason": to_label,
        "model": {
            **report, "finalCoef": b, "ageSplit": AGE_SPLIT, "knots": KNOTS,
            "tiers": [{"label": lab, "slope": round(float(s), 3),
                       "shrink": round((1 - float(s)) * 100, 1)}
                      for lab, s in zip(labels, sl)],
        },
        "agingCurve": [{"age": a, "delta": round(raw_curve[a], 2), "n": counts[a]}
                       for a in sorted(raw_curve)],
        "players": out,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
                   encoding="utf-8")

    print(f"{from_label} → {to_label} 예측: {len(out):,}명   "
          f"저장 {OUT.name} ({OUT.stat().st_size/1024:.0f} KB)\n")
    print(f"{'':3}{'선수':<26}{'팀':<5}{'나이':>4}{'올해':>7}{'예측':>7}{'변화':>7}")
    print("-" * 62)
    for i, r in enumerate(out[:10], 1):
        print(f"{i:>2} {r['player']:<26}{r['team']:<5}{r['nextAge']:>4}"
              f"{r['current']:>7.1f}{r['projected']:>7.1f}{r['change']:>+7.1f}")


if __name__ == "__main__":
    main()
