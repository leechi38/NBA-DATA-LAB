"""팀 예측에 넣어봤지만 채택하지 않은 세 가지를 재현한다. 결과 → exp_team_features.json

build_team_projection.py의 채택 모델(comp+prev)을 기준으로, 하나씩 켜고 홀드아웃
(2016~2026, 330팀-시즌)에서 비교한다. 대체값·계수는 모두 학습 구간(~2015)에서만 뽑는다.

  1) 드래프트 순번 — 신인을 전원 같은 값(NDL 49.8)으로 두는 대신 지명 순번 구간별
     학습 구간 평균을 준다. 상위 지명 신인이 있는 팀이 낮게 잡힌다는 가설의 검증.
  2) 500분 미만 기록 — 선수 모델은 총 500분 이상 시즌만 보므로, 짧게라도 뛴 기록이
     있는 선수가 신인과 같은 대체값을 받는다. 이 기록으로 따로 회귀를 적합한다.
  3) 로스터 연속성 — 작년에도 같은 팀에서 뛴 선수의 출전 비중. 로스터가 많이 바뀐
     팀일수록 작년 팀 레이팅을 덜 믿도록 '작년 레이팅 × 연속성' 항을 더한다.

결론: 셋 다 순레이팅 오차가 2.74 → 2.73~2.74로 차이가 없다. 지명 순번은 1~3순위만
평균보다 높고(55.6) 나머지는 48.5~50.2라 대체값이 거의 같아진다.

드래프트 페이지(draft/NBA_{year}.html)는 fetch_team_data.py와 같은 캐시·간격 규칙으로
받는다. 처음 실행하면 51개 페이지를 받느라 3분쯤 걸린다.
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import build_projection as bp
import build_team_projection as tp
from fetch_team_data import fetch_html

BASE = Path(__file__).parent
OUT = BASE / "exp_team_features.json"
TRAIN_END = tp.TRAIN_END
DRAFT_YEARS = range(1976, 2027)     # 1979 시즌 신인부터 덮으려면 1976 드래프트부터
PICK_BINS = [(1, 3), (4, 10), (11, 20), (21, 30), (31, 60)]


def load_draft():
    """선수 ID → 전체 지명 순번. 같은 선수가 두 번 지명되면 처음 것."""
    picks = {}
    for y in DRAFT_YEARS:
        html = fetch_html(f"https://www.basketball-reference.com/draft/NBA_{y}.html",
                          f"draft_html_{y}")
        table = re.search(r'<table[^>]*id="stats".*?</table>', html, re.S).group(0)
        for row in re.findall(r"<tr.*?</tr>", table, re.S):
            pk = re.search(r'data-stat="pick_overall"[^>]*>(?:<a[^>]*>)?(\d+)', row)
            pid = re.search(r"/players/\w/(\w+)\.html", row)
            if pk and pid:
                picks.setdefault(pid.group(1), int(pk.group(1)))
    return picks


def pick_bin(pick):
    for lo, hi in PICK_BINS:
        if pick is not None and lo <= pick <= hi:
            return f"{lo}–{hi}"
    return "미지명"   # 1989년 이전 3라운드 이하 지명도 여기 들어간다


class Players:
    """시즌 year의 선수 예측. opts로 실험 항목을 켠다. year-1까지의 기록만 본다."""

    def __init__(self, rows_q, rows_all, stints, picks, opts):
        self.rows_q, self.rows_all, self.stints, self.picks = rows_q, rows_all, stints, picks
        self.opts = opts
        self.first = {}
        for pid, y in rows_all:
            self.first[pid] = min(self.first.get(pid, 9999), y)
        self.curve, _, _ = bp.aging_curve(rows_q, upto=TRAIN_END)
        self.coef, _ = bp.fit(rows_q, self.curve, TRAIN_END)
        self._fit_fallbacks()

    def history(self, pid, year):
        """출전시간 하한 없이 직전 세 시즌의 가중평균 (구성요소, NDL, 총 출전시간)."""
        past = [(self.rows_all.get((pid, year - 1 - k)), bp.WEIGHTS[k]) for k in range(3)]
        past = [(r, w) for r, w in past if r]
        if not past:
            return None
        ws = sum(w for _, w in past)
        return (np.sum([np.array(r["comps"]) * w for r, w in past], axis=0) / ws,
                sum(r["ndl"] * w for r, w in past) / ws,
                sum(r["mpg"] * r["g"] for r, _ in past))

    def category(self, pid, year):
        if bp.features(self.rows_q, self.curve, pid, year - 1) is not None:
            return "known"
        if self.history(pid, year) is not None:
            return "low"
        if self.first.get(pid) == year:
            return "rookie:" + pick_bin(self.picks.get(pid)) if "draft" in self.opts else "rookie"
        return "other"

    def _fit_fallbacks(self):
        acc = defaultdict(lambda: ([], [], [], []))   # ndl, comps, 총 출전, 경기당 출전
        low_x, low_y, low_w = [], [], []
        for y in range(tp.FIRST_TARGET, TRAIN_END + 1):
            for pid, st in self.stints[y].items():
                cur = self.rows_all.get((pid, y))
                cat = self.category(pid, y)
                if cur is None or cat == "known":
                    continue
                minutes = sum(m for _, _, m in st)
                if cat == "low":
                    _, ndl, mins = self.history(pid, y)
                    low_x.append([1.0, ndl, np.log1p(mins)])
                    low_y.append(cur["ndl"])
                    low_w.append(minutes)
                for key in {cat, "unknown"}:
                    a = acc[key]
                    a[0].append(cur["ndl"]); a[1].append(cur["comps"])
                    a[2].append(minutes); a[3].append(cur["mpg"])
        self.fallback = {k: {"ndl": float(np.average(v[0], weights=v[2])),
                             "comps": np.average(np.array(v[1]), axis=0, weights=v[2]),
                             "mpg": float(np.mean(v[3])), "n": len(v[0])}
                         for k, v in acc.items()}
        w = np.sqrt(np.array(low_w))
        self.low_coef = np.linalg.lstsq(np.array(low_x) * w[:, None], np.array(low_y) * w,
                                        rcond=None)[0]

    def player(self, pid, year):
        cat = self.category(pid, year)
        prev = (self.rows_all.get((pid, year - 1)) or self.rows_all.get((pid, year - 2))
                or self.rows_all.get((pid, year - 3)))
        mpg = prev["mpg"] if prev else self.fallback["unknown"]["mpg"]
        if cat == "known":
            f = bp.features(self.rows_q, self.curve, pid, year - 1)
            return {"proj": float(np.dot(self.coef, f["x"])),
                    "comps": self.history(pid, year)[0], "mpg": mpg, "cat": cat}
        if cat == "low" and "lowfit" in self.opts:
            comps, ndl, mins = self.history(pid, year)
            return {"proj": float(np.dot(self.low_coef, [1.0, ndl, np.log1p(mins)])),
                    "comps": comps, "mpg": mpg, "cat": cat}
        fb = self.fallback[cat if cat.startswith("rookie:") else "unknown"]
        return {"proj": fb["ndl"], "comps": fb["comps"],
                "mpg": mpg if prev else fb["mpg"], "cat": cat}


def dataset(pl, stints, rt):
    recs = []
    for y in range(tp.FIRST_TARGET, tp.TARGET):
        teams = defaultdict(list)
        for pid, st in stints[y].items():
            p = dict(pl.player(pid, y))
            last = stints.get(y - 1, {}).get(pid)
            p["same"] = 1.0 if last and last[-1][0] == st[0][0] else 0.0
            teams[st[0][0]].append(p)
        for team, ps in teams.items():
            w = np.array([p["mpg"] for p in ps])
            recs.append({"year_id": y, "team_abbr": team,
                         "x_real": tp.team_feature(ps, w),
                         "cont": float(np.dot(w / w.sum(), [p["same"] for p in ps]))})
    ds = pd.DataFrame(recs).merge(rt, on=["year_id", "team_abbr"])
    return tp.prev_ratings(ds, rt)


def design(df, continuity):
    X = tp.design(df, "x_real", "comp+prev")
    if continuity:
        X = np.column_stack([X, df["prev_rel_ortg"] * df["cont"], df["prev_rel_drtg"] * df["cont"]])
    return X


def score(ds, continuity=False):
    tr, te = ds[ds.year_id <= TRAIN_END].copy(), ds[ds.year_id > TRAIN_END].copy()
    k_win = float(np.polyfit(tr.rel_ortg - tr.rel_drtg, tr.wpct - 0.5, 1)[0])
    for tgt in ("ortg", "drtg"):
        b = tp.ols(design(tr, continuity), tr[f"rel_{tgt}"].values)
        te[f"p_m_{tgt}"] = design(te, continuity) @ b
        te[f"p_m_{tgt}"] = tp.center(te, f"p_m_{tgt}")
    r = tp.evaluate(te, k_win)["m"]
    nrtg = te.p_m_ortg - te.p_m_drtg - (te.rel_ortg - te.rel_drtg)
    cho = nrtg[(te.year_id == 2026) & (te.team_abbr == "CHO")]
    return {"maeN": r["maeN"], "maeW": r["maeW"], "corrD": r["corrD"], "corrN": r["corrN"],
            "charlotte2026": round(float(cho.iloc[0]), 1) if len(cho) else None}


def main():
    rows_q, _ = bp.load()
    rows_all = tp.load_all_rows()
    rt = tp.load_ratings()
    picks = load_draft()
    print(f"드래프트 기록 {len(picks):,}명")
    stints = {y: dict(tp.team_stints(y)) for y in range(tp.FIRST_TARGET - 1, tp.TARGET)}

    variants = [("기존 (신인 전원 같은 값)", (), False),
                ("+ 드래프트 순번", ("draft",), False),
                ("+ 500분 미만 기록", ("lowfit",), False),
                ("+ 로스터 연속성", (), True)]
    results, pl_draft = [], None
    for label, opts, cont in variants:
        pl = Players(rows_q, rows_all, stints, picks, opts)
        if "draft" in opts:
            pl_draft = pl
        r = score(dataset(pl, stints, rt), cont)
        results.append({"label": label, **r})
        print(f"  {label:<22} NRtg {r['maeN']:.2f}  승수 {r['maeW']:.2f}  "
              f"수비 상관 {r['corrD']:.3f}  샬럿 2025-26 {r['charlotte2026']:+.1f}")

    bins = []
    for lo, hi in [*PICK_BINS, (None, None)]:
        key = f"rookie:{lo}–{hi}" if lo else "rookie:미지명"
        fb = pl_draft.fallback[key]
        bins.append({"label": f"{lo}~{hi}순위" if lo else "미지명·3라운드 이하",
                     "n": fb["n"], "ndl": round(fb["ndl"], 1), "mpg": round(fb["mpg"], 1)})
        print(f"  {bins[-1]['label']:<14} n={fb['n']:>4}  NDL {fb['ndl']:.1f}  {fb['mpg']:.1f}분")
    overall = pl_draft.fallback["unknown"]

    OUT.write_text(json.dumps({
        "variants": results, "pickBins": bins,
        "unknownAvg": {"ndl": round(overall["ndl"], 1), "mpg": round(overall["mpg"], 1)},
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"저장 {OUT.name}")


if __name__ == "__main__":
    main()
