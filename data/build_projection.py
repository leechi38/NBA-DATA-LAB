"""다음 시즌 NDL Score 예측값을 만들어 projection.json으로 내보낸다.

    예측 = b0 + (기량 구간별 기울기)×(최근 3시즌 5:4:3 가중평균)
              + b×(직전 시즌 − 가중평균)
              + b×(나이 델타, 30세 미만)×(기량: 55점 위 10점 단위, 2시즌 이상만)
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
믿을지를 손으로 정하는 대신 계수가 정하게 했다. 결과는 0.87 대 1.42로,
젊은 구간을 실제로 덜 신뢰한다.

5:4:3도 손으로 정한 상수였다. '직전 시즌 − 가중평균' 항을 하나 더해 시즌별 비중을
데이터가 고르게 했더니 계수가 0.31로, 직전 시즌 비중이 42%에서 약 60%로 올라갔다.
MAE 4.13 → 4.08(부트스트랩 95% 신뢰구간 [+0.015, +0.082])이고, 나이대(≤23·24–29·30+)와
예측 점수대(<50·50–70·70+)로 나눠도 모든 구간에서 줄었다. 평균이 엇갈린 효과를
가린 것이 아니다. 출전시간·경기수·선발 비율·경력 연차도 같은 방식으로 시험했으나
0.01 안팎이라 넣지 않았다.

나이 곡선은 모든 젊은 선수의 평균이라 잘하는 유망주의 성장을 덜 준다. 22세 이하·기량 60점
이상은 실제가 예측보다 1.6점 높았다. 젊은 구간 나이델타에 기량을 곱한 항을 더하자(2시즌
이상만) MAE 4.083 → 4.065, 신뢰구간 [+0.010, +0.028], 그 그룹 편향 +1.62 → +0.65.
같은 항이 26~29세 스타는 조금 더 내리는데, 최근 시험 구간에서 이 그룹은 실제가 예측보다
2.6점 낮았던 터라 방향이 맞는다. 급등한 해를 기량별로 더 믿게 하는 항은 계수가 0이었다.

홀드아웃(2016-17년~, 2,576개) 성능
    MAE  4.61 → 4.07     상관 0.730 → 0.766     (기준: 작년 성적 그대로)
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
GROWTH_FROM = 55      # 이 점수 위부터 나이 성장분을 기량에 비례해 더 준다 (아래 features 참고)
FULL_GAMES = 0.85     # 시즌 경기 수의 이 비율 이상 뛰면 결장 없음으로 본다. 학습 구간 안에서만
                      # 골랐다(~2005 적합 → 2006~2014 검증, 60~100% 중 85%가 최소 오차)
REF_MINUTES = 2000    # NDL이 출전시간 감점을 다 푸는 총 출전시간 (fetch_all_seasons.py와 같은 값)


def load():
    """선수-시즌 기록. 'adj'는 결장 경기만 되살린 점수다.

    NDL은 50 + (경기력 × 출전 비중)이라 부상으로 빠진 시즌은 점수가 50 쪽으로 눌린다.
    아데토쿤보 2025-26은 36경기(비중 0.52)라 62.1인데, 같은 경기력으로 결장 없이 뛰었다면
    약 73이다. 다음 시즌을 맞히는 데는 이 쪽이 낫다. 다만 경기당 출전시간은 건드리지 않는다.
    처음에 총 출전시간 전체를 되살렸더니 원래 적게 뛰는 식스맨까지 부풀려 오히려 나빠졌다.
    """
    rows, meta = {}, {}
    for f in sorted((BASE / "seasons").glob("players_*.json"),
                    key=lambda p: int(p.stem.split("_")[1])):
        d = json.loads(f.read_text(encoding="utf-8"))
        year = int(d["season"].split("-")[0]) + 1
        season_games = max(int(p.get("G") or 0) for p in d["players"])   # 단축 시즌(50·66경기) 대응
        full = FULL_GAMES * season_games
        for p in d["players"]:
            pid, age, ndl = p.get("BbrefId"), p.get("Age"), p.get("NDL_Score")
            g, mp = p.get("G"), p.get("MP")
            if not pid or age is None or ndl is None or not g or mp is None:
                continue
            if g * mp < MIN_MP:
                continue
            w = max(p.get("NDL_Weight") or 0.0, 0.05)
            w_full = min(1.0, mp * max(g, full) / REF_MINUTES)   # 결장 없이 뛰었다면의 비중
            rows[(pid, year)] = {
                "age": int(age), "ndl": ndl, "mp": g * mp, "games": int(g),
                "share": min(1.0, g / full),
                "adj": min(100.0, max(0.0, 50 + (ndl - 50) * w_full / w)),
            }
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
    # 결장 보정 점수의 가중평균. 적게 뛴 시즌은 덜 믿도록 경기 비율을 가중치에 곱한다.
    ws = [w * r["share"] for r, w in past]
    wavg = sum(r["adj"] * w for (r, _), w in zip(past, ws)) / sum(ws)
    cur = past[0][0]
    raw_d = curve.get(cur["age"], 0.0)
    old = cur["age"] >= AGE_SPLIT
    # 나이 곡선은 모든 선수의 평균이라, 결국 자리 잡지 못하는 선수까지 섞여 있다.
    # 잘하는 젊은 선수는 그보다 가파르게 큰다. 젊은 구간 나이델타에 기량(55점 위로
    # 10점마다 1)을 곱한 항을 더해, 얼마나 더 줄지는 계수가 정하게 한다. 한 시즌만
    # 뛴 신인은 뺀다. 첫해 고득점에는 운이 섞여 최근에는 2년차에 오히려 내려왔다.
    talent = max(0.0, (wavg - GROWTH_FROM) / 10) if len(past) >= 2 else 0.0
    young_d = 0.0 if old else raw_d
    # [절편, wavg, 마디별 추가 기울기…, 직전 시즌 편차, 유망주 성장, 나이델타(젊음), 나이델타(30세+)]
    # 나이델타 두 항은 끝에 둔다. 페이지가 계수 배열의 끝 두 개를 나이델타로 읽는다.
    x = ([1.0, wavg] + [max(0.0, wavg - k) for k in KNOTS]
         + [cur["adj"] - wavg, young_d * talent]
         + [young_d, raw_d if old else 0.0])
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


RECENCY_IDX = 2 + len(KNOTS)   # 계수 배열에서 직전 시즌 편차 항의 위치
GROWTH_IDX = 3 + len(KNOTS)    # 유망주 성장 항의 위치


def recency_weights(coef, tier=2):
    """세 시즌을 다 뛴 선수 기준, 예측에서 각 시즌이 실제로 차지하는 비중.
    tier는 slopes_of의 구간 번호(2 = 60–70)."""
    s, r = slopes_of(coef)[tier], coef[RECENCY_IDX]
    w = np.array(WEIGHTS, dtype=float) / sum(WEIGHTS)
    eff = s * w + r * (np.eye(len(w))[0] - w)   # s×wavg + r×(직전 − wavg)
    return eff / eff.sum()


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


# ── 범위와 상한 ──────────────────────────────────────────────────────────
# 예측값 하나는 비슷한 선수들의 가운데값이라, 크니펠 같은 신인을 평균 경로(66 → 69)로만
# 보여준다. 실제로는 비슷한 출발을 한 선수 셋 중 하나가 80점을 넘었다. 그 가능성을 숫자로
# 보이려고 두 가지를 덧붙인다.
#   1) 다음 시즌 80% 범위 — 나이·기량·경력이 비슷한 그룹에서 실제가 예측을 벗어난 폭
#   2) 5년 안에 70·80·90점 도달 확률 — 비슷한 과거 선수(비교 선수)가 실제로 오른 폭

RANGE_YEARS = 15      # 범위는 예측 시점 직전 15시즌의 잔차로 잡는다. 2016년 이후 기량 70+ 선수가
                      # 다음 해 결장하는 비율이 26~28%에서 59%로 뛰어(로드 매니지먼트), 오래된
                      # 시즌까지 섞으면 스타의 아래쪽 폭을 못 따라간다
RANGE_MIN_N = 100     # 그룹 표본이 이보다 적으면 이웃 그룹과 합친다
COMP_K = 30           # 비교 선수 수
COMP_WINDOW = (25, 5) # 비교 풀: 예측 시점 25~5년 전 시즌 (5년 뒤 결과까지 관측된 시즌만)
REACH = (70, 80, 90)
HORIZON = 5


def range_key(f):
    a, w = f["cur"]["age"], f["wavg"]
    return (0 if a <= 22 else 1 if a <= 25 else 2 if a <= 29 else 3,
            0 if w < 50 else 1 if w < 60 else 2 if w < 70 else 3,
            1 if f["seasons"] == 1 else 2)


def range_bands(rows, curve, coef, since, upto):
    """그룹별로 (실제 − 예측)의 10·90 백분위. 신인 여부 → 이웃 나이대 순으로 합친다."""
    g = defaultdict(list)
    for (pid, yr), nxt in rows.items():
        if not since <= yr - 1 <= upto:
            continue
        f = features(rows, curve, pid, yr - 1)
        if f is None or (pid, yr - 1) not in rows:
            continue
        g[range_key(f)].append(nxt["ndl"] - float(np.dot(coef, f["x"])))
    bands = {}
    for k in [(a, t, s) for a in range(4) for t in range(4) for s in (1, 2)]:
        src = g.get(k, [])
        if len(src) < RANGE_MIN_N:
            src = [r for kk, v in g.items() if kk[:2] == k[:2] for r in v]
        if len(src) < RANGE_MIN_N:
            src = [r for kk, v in g.items() if kk[1] == k[1] and abs(kk[0] - k[0]) <= 1 for r in v]
        bands[k] = (float(np.percentile(src, 10)), float(np.percentile(src, 90)))
    return bands


def holdout_range(rows):
    """~2015년으로 만든 범위가 이후 시즌의 실제를 몇 %나 담는지."""
    curve, _, _ = aging_curve(rows, upto=TRAIN_END)
    coef, _ = fit(rows, curve, TRAIN_END)
    bands = range_bands(rows, curve, coef, TRAIN_END - RANGE_YEARS + 1, TRAIN_END)
    hit = defaultdict(list)
    for (pid, yr), nxt in rows.items():
        if yr - 1 <= TRAIN_END + 1:
            continue
        f = features(rows, curve, pid, yr - 1)
        if f is None or (pid, yr - 1) not in rows:
            continue
        p = float(np.dot(coef, f["x"]))
        lo, hi = bands[range_key(f)]
        a, w, inside = f["cur"]["age"], f["wavg"], p + lo <= nxt["ndl"] <= p + hi
        for grp, ok in [("전체", True), ("22세 이하", a <= 22), ("23~29세", 23 <= a <= 29),
                        ("30세 이상", a >= 30), ("기량 70 이상", w >= 70), ("신인", f["seasons"] == 1)]:
            if ok:
                hit[grp].append(inside)
        hit["위로 벗어남"].append(nxt["ndl"] > p + hi)
        hit["아래로 벗어남"].append(nxt["ndl"] < p + lo)
    return [{"group": k, "rate": round(float(np.mean(v)), 3), "n": len(v)} for k, v in hit.items()]


def career_profiles(rows, level="adj"):
    """선수-시즌마다 비교에 쓸 프로필과, 이후 5시즌 최고점(리그를 떠났으면 0).
    비교 기준 점수(level)는 결장 보정 점수다. 부상 시즌의 눌린 점수로 비교하면
    아데토쿤보(36경기, 62.1)가 '32세 62점 선수'들과 묶여 80점 도달 0%가 나왔다."""
    by_player = defaultdict(list)
    for pid, y in rows:
        by_player[pid].append(y)
    prof = {}
    for pid, ys in by_player.items():
        ys.sort()
        for i, y in enumerate(ys):
            past = [(rows.get((pid, y - k)), WEIGHTS[k]) for k in range(3)]
            past = [(r, w) for r, w in past if r]
            later = [rows[(pid, y + k)]["ndl"] for k in range(1, HORIZON + 1) if (pid, y + k) in rows]
            prof[(pid, y)] = {
                "age": rows[(pid, y)]["age"], "ndl": rows[(pid, y)][level],
                "wavg": (sum(r[level] * w * r["share"] for r, w in past)
                         / sum(w * r["share"] for r, w in past)),
                "exp": min(i + 1, 3), "peak": max(later) if later else 0.0,
            }
    return prof


def comp_pool(prof, lo, hi):
    pool = defaultdict(list)
    for key, p in prof.items():
        if lo <= key[1] <= hi:
            pool[p["age"]].append((key, p))
    return pool


def find_comps(pool, p):
    """같은 나이(표본이 얇으면 ±1세)에서 올해 점수·3시즌 평균·경력이 가장 가까운 선수들."""
    cand = pool.get(p["age"], [])
    if len(cand) < COMP_K * 3:
        cand = [c for a in (p["age"] - 1, p["age"], p["age"] + 1) for c in pool.get(a, [])]
    d = [np.hypot(c["ndl"] - p["ndl"], c["wavg"] - p["wavg"]) + 4 * abs(c["exp"] - p["exp"])
         for _, c in cand]
    return [cand[i] for i in np.argsort(d)[:COMP_K]]


def reach_probs(p, comps):
    """비교 선수의 최고점을 그대로 쓰지 않고, 그 시점에서 오른 폭을 옮겨 붙인다.
    최상위 선수는 자기보다 위에 비교 대상이 거의 없어, 최고점을 그대로 쓰면 확률이
    체계적으로 낮아진다(80점 도달 30~50% 예측 → 실제 57%). 폭을 옮기면 34%로 맞는다."""
    return [float(np.mean([c["peak"] > 0 and p["ndl"] + c["peak"] - c["ndl"] >= t
                           for _, c in comps])) for t in REACH]


def holdout_reach(rows, prof):
    """각 시즌을 그 시점에 결과가 나와 있던 비교 풀로만 예측해, 확률이 맞는지 본다."""
    last = max(y for _, y in rows)
    P, O, young = [], [], []
    for ty in range(TRAIN_END - 4, last - HORIZON + 1):
        pool = comp_pool(prof, ty - COMP_WINDOW[0], ty - COMP_WINDOW[1])
        for key, p in prof.items():
            if key[1] != ty:
                continue
            P.append(reach_probs(p, find_comps(pool, p)))
            O.append([p["peak"] >= t for t in REACH])
            young.append(p["age"] <= 23)
    P, O, young = np.array(P), np.array(O, dtype=float), np.array(young)
    out = {"seasons": [TRAIN_END - 4, last - HORIZON], "n": len(P), "byThreshold": []}
    for j, t in enumerate(REACH):
        bins = []
        for lo, hi in [(0, .05), (.05, .15), (.15, .3), (.3, .5), (.5, .8), (.8, 1.01)]:
            m = (P[:, j] >= lo) & (P[:, j] < hi)
            if m.sum() >= 10:
                bins.append({"lo": lo, "hi": min(hi, 1), "n": int(m.sum()),
                             "pred": round(float(P[m, j].mean()), 3), "obs": round(float(O[m, j].mean()), 3)})
        out["byThreshold"].append({
            "threshold": t, "pred": round(float(P[:, j].mean()), 4), "obs": round(float(O[:, j].mean()), 4),
            "youngPred": round(float(P[young, j].mean()), 4), "youngObs": round(float(O[young, j].mean()), 4),
            "brier": round(float(np.mean((P[:, j] - O[:, j]) ** 2)), 4), "bins": bins})
    return out


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
    rw = recency_weights(coef)
    print(f"\n직전 시즌 편차 계수 {b[RECENCY_IDX]:.3f} → 실효 시즌 비중(60점대 선수) "
          + ":".join(f"{w:.0%}" for w in rw) + f"  (5:4:3은 {5/12:.0%}:{4/12:.0%}:{3/12:.0%})")
    print(f"유망주 성장 계수 {b[GROWTH_IDX]:.3f} → 기량 {GROWTH_FROM}점 위 10점마다 젊은 구간 "
          f"성장분이 +{b[GROWTH_IDX] / b[-2]:.0%}씩")
    print(f"나이 델타 계수: 30세 미만 {b[-2]:.2f} / 30세 이상 {b[-1]:.2f}\n")

    last_year = max(y for _, y in rows)
    from_label = next(meta[pid]["season"] for (pid, y) in rows if y == last_year)
    to_label = f"{last_year}-{str(last_year + 1)[-2:]}"

    range_check = holdout_range(rows)
    print("다음 시즌 80% 범위 — 시험 구간에서 실제가 범위 안에 든 비율 (목표 80%)")
    print("  " + "  ".join(f"{r['group']} {r['rate']:.1%}(n{r['n']})" for r in range_check))
    prof = career_profiles(rows)
    reach_check = holdout_reach(rows, prof)
    print(f"\n{HORIZON}년 내 도달 확률 — {reach_check['seasons'][0] - 1}~{reach_check['seasons'][1]} 시즌을 "
          "그 시점의 비교 풀로만 예측")
    for t in reach_check["byThreshold"]:
        print(f"  {t['threshold']}점  평균 예측 {t['pred']:.1%} / 실제 {t['obs']:.1%}  Brier {t['brier']:.4f}  | "
              + "  ".join(f"{b['lo']:.0%}~{b['hi']:.0%}: {b['pred']:.0%}→{b['obs']:.0%}(n{b['n']})" for b in t["bins"]))

    bands = range_bands(rows, curve, coef, last_year - RANGE_YEARS, last_year - 1)
    pool = comp_pool(prof, last_year - COMP_WINDOW[0], last_year - COMP_WINDOW[1])
    print()

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
        lo, hi = bands[range_key(f)]
        p = prof[(pid, last_year)]
        comps = find_comps(pool, p)
        out.append({
            "low": round(max(0.0, proj + lo), 1), "high": round(min(100.0, proj + hi), 1),  # 척도는 0~100
            "reach": [round(v, 2) for v in reach_probs(p, comps)],
            "comps": [{"player": meta[k[0]]["name"], "season": f"{k[1] - 1}-{str(k[1])[-2:]}",
                       "age": c["age"], "ndl": round(c["ndl"], 1), "peak": round(c["peak"], 1)}
                      for k, c in comps[:3]],
            "id": pid, "player": m["name"], "team": m["team"], "pos": m["pos"],
            "age": cur["age"], "nextAge": cur["age"] + 1,
            "current": round(cur["ndl"], 1),
            "games": cur["games"], "gamesShare": round(cur["share"], 2), "adjusted": round(cur["adj"], 1),
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
            "recencyCoef": b[RECENCY_IDX],
            "recencyWeights": [round(float(w), 3) for w in rw],
            "growthCoef": b[GROWTH_IDX], "growthFrom": GROWTH_FROM,
            "rangeCheck": range_check, "reachCheck": reach_check,
            "reach": {"thresholds": list(REACH), "horizon": HORIZON, "k": COMP_K,
                      "pool": [f"{y - 1}-{str(y)[-2:]}" for y in
                               (last_year - COMP_WINDOW[0], last_year - COMP_WINDOW[1])]},
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
