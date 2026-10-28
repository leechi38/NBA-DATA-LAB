"""선수별 다음 시즌 NDL 예측을 팀 단위로 모아 다음 시즌 팀 ORtg/DRtg를 예측한다.

방식 — 예: 2017-18 클리블랜드
    1) 그 시즌 클리블랜드 로스터의 선수 전원에 대해, 2016-17까지의 기록만으로
       다음 시즌 NDL Score를 예측한다 (build_projection.py와 같은 모델)
    2) 예상 출전시간으로 가중 평균해 팀 피처를 만든다
    3) 피처 → 그 시즌 실제 팀 ORtg/DRtg(리그 평균 대비)로 회귀를 학습한다
    4) 학습에 안 쓴 시즌에서 에러를 잰다

시즌 시작 시점에 알 수 있는 것만 쓰도록 두 가지를 정한다.
    - 로스터: 그 시즌에 처음 뛴 팀을 소속으로 본다. 시즌 중 트레이드는 모르는 채
      개막 로스터로 예측하는 셈이다(2026-27 예측도 같은 조건이다).
    - 출전시간: 그 시즌 실제 출전시간 대신 직전 시즌 경기당 출전시간을 쓴다.
      직전 시즌 기록이 없으면(신인 등) 학습 구간의 같은 처지 선수 평균을 쓴다.
비교용으로 실제 로스터·실제 출전시간을 쓰는 '오라클' 버전도 함께 잰다. 둘의
차이가 로스터·출전시간을 모르는 데서 오는 손실이다.

목표값은 일정 보정된 ORtg/DRtg(ORtg/A, DRtg/A)에서 그 시즌 리그 평균을 뺀 값이다.
리그 전체 득점 수준은 시대마다 10점 넘게 움직여, 빼지 않으면 시대를 맞히는
문제가 된다. DRtg는 낮을수록 좋은 수비다.

학습: 1979~2015 시즌 / 홀드아웃: 2016~2026 시즌 (build_projection.py와 같은 경계)
"""
import io
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import build_projection as bp

BASE = Path(__file__).parent
CACHE_DIR = BASE / ".cache"
RATINGS_CSV = BASE / "team_ratings.csv"
ROSTER_CSV = BASE / "rosters_2026_27.csv"
OUT = BASE / "team_projection.json"

FIRST_TARGET = 1979   # 직전 시즌 팀 레이팅(기준선 비교용)이 있는 첫 시즌
TRAIN_END = 2015
TARGET = 2027         # 2026-27
COMPS = ["NDL_Production", "NDL_Efficiency", "NDL_Playmaking", "NDL_Defense", "NDL_WinImpact"]
COMP_LABEL = ["생산성", "효율성", "플레이메이킹", "수비기여", "승리기여"]

# 연고지 이전·개명으로 약자가 바뀐 프랜차이즈. 직전 시즌 팀 레이팅을 찾을 때만 쓴다.
PREDECESSOR = {"SAC": ["KCK"], "LAC": ["SDC"], "BRK": ["NJN"], "OKC": ["SEA"],
               "MEM": ["VAN"], "NOH": ["CHH", "NOK"], "NOK": ["NOH"], "NOP": ["NOH"],
               "CHO": ["CHA"], "WAS": ["WSB"]}


# ── 데이터 ──────────────────────────────────────────────────────────────

def load_all_rows():
    """출전시간 하한 없이 모든 선수-시즌. 구성요소·출전시간 계산용."""
    rows = {}
    for f in (BASE / "seasons").glob("players_*.json"):
        d = json.loads(f.read_text(encoding="utf-8"))
        year = int(d["season"].split("-")[0]) + 1
        for p in d["players"]:
            pid, g, mp = p.get("BbrefId"), p.get("G"), p.get("MP")
            if not pid or not g or mp is None or p.get("NDL_Score") is None:
                continue
            w = p.get("NDL_Weight") or 0.0
            # 구성요소는 수축 전 z-점수라 저출전 선수가 ±10까지 튄다. NDL Score와
            # 같은 출전시간 가중치를 곱해 평균 쪽으로 당긴다.
            comps = [(p.get(c) or 0.0) * w for c in COMPS]
            rows[(pid, year)] = {"ndl": p["NDL_Score"], "comps": comps, "mpg": mp, "g": g}
    return rows


def team_stints(year):
    """선수별 팀 구간(팀, 경기수, 총 출전시간)을 시즌 내 순서대로. 트레이드 합산 행은 뺀다."""
    html = (CACHE_DIR / f"per_game_stats_html_{year}.html").read_text(encoding="utf-8")
    t = pd.read_html(io.StringIO(html), attrs={"id": "per_game_stats"}, extract_links="body")[0]
    out = defaultdict(list)
    for _, r in t.iterrows():
        name, href = r["Player"]
        team, team_href = r["Team"]
        if not href or not team_href:   # 헤더 반복·리그 평균·2TM/TOT 합산 행
            continue
        pid = href.rsplit("/", 1)[1].removesuffix(".html")
        g = pd.to_numeric(r["G"][0], errors="coerce")
        mpg = pd.to_numeric(r["MP"][0], errors="coerce")
        if pd.isna(g) or pd.isna(mpg):
            continue
        out[pid].append((team, float(g), float(g * mpg)))
    return out


def load_ratings():
    rt = pd.read_csv(RATINGS_CSV)
    rt = rt.dropna(subset=["ortg_adj", "drtg_adj"]).copy()
    for c in ["ortg_adj", "drtg_adj"]:
        rt["rel_" + c[:4]] = rt[c] - rt.groupby("year_id")[c].transform("mean")
    rt["games"] = rt["w"] + rt["l"]
    rt["wpct"] = rt["w"] / rt["games"]
    return rt


# ── 선수 → 팀 피처 ───────────────────────────────────────────────────────

class PlayerModel:
    """시즌 year의 선수 예측. year-1까지의 기록만 본다."""

    def __init__(self, rows_q, rows_all, curve, coef):
        self.rows_q, self.rows_all, self.curve, self.coef = rows_q, rows_all, curve, coef
        self.fallback = None   # 기록 없는 선수용 (ndl, comps, mpg)

    def history(self, pid, year):
        f = bp.features(self.rows_q, self.curve, pid, year - 1)
        proj = float(np.dot(self.coef, f["x"])) if f else None
        past = [(self.rows_all.get((pid, year - 1 - k)), bp.WEIGHTS[k]) for k in range(3)]
        past = [(r, w) for r, w in past if r]
        comps = (np.sum([np.array(r["comps"]) * w for r, w in past], axis=0)
                 / sum(w for _, w in past)) if past else None
        prev = self.rows_all.get((pid, year - 1)) or self.rows_all.get((pid, year - 2))
        mpg = prev["mpg"] if prev else None
        return proj, comps, mpg

    def fit_fallback(self, entries):
        """기록 없는 선수가 실제로 낸 성적·출전시간의 평균 (학습 구간에서만)."""
        ndl, comps, mpg, wts = [], [], [], []
        for pid, year, minutes in entries:
            proj, c, m = self.history(pid, year)
            cur = self.rows_all.get((pid, year))
            if cur is None:
                continue
            if proj is None:
                ndl.append(cur["ndl"]); wts.append(minutes)
                comps.append(cur["comps"])
            if m is None:
                mpg.append(cur["mpg"])
        wts = np.array(wts)
        self.fallback = (float(np.average(ndl, weights=wts)),
                         np.average(np.array(comps), axis=0, weights=wts),
                         float(np.mean(mpg)))

    def player(self, pid, year):
        proj, comps, mpg = self.history(pid, year)
        fb_ndl, fb_comps, fb_mpg = self.fallback
        return {"proj": fb_ndl if proj is None else proj,
                "comps": fb_comps if comps is None else comps,
                "mpg": fb_mpg if mpg is None else mpg,
                "known": proj is not None}


def team_feature(players, weights):
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    proj = float(np.dot(w, [p["proj"] for p in players]))
    comps = np.dot(w, np.array([p["comps"] for p in players]))
    return [proj, *comps.tolist()]


def build_dataset(pm, years):
    """(시즌, 팀)마다 현실 버전·오라클 버전·상한 버전 피처를 만든다.

    상한(x_true)은 그 시즌 실제 NDL과 실제 출전시간을 쓴다. 선수 예측이 완벽하다면
    팀을 얼마나 맞힐 수 있는지, 즉 남은 오차 중 선수 예측 몫이 얼마인지 재는 용도다.
    """
    recs = []
    for y in years:
        stints = team_stints(y)
        realistic, oracle, truth = defaultdict(list), defaultdict(list), defaultdict(list)
        for pid, st in stints.items():
            pl = pm.player(pid, y)
            realistic[st[0][0]].append((pl, pl["mpg"]))     # 처음 뛴 팀 = 개막 로스터
            cur = pm.rows_all.get((pid, y))
            for team, g, minutes in st:
                if minutes > 0:
                    oracle[team].append((pl, minutes))
                    if cur is not None:
                        truth[team].append(({"proj": cur["ndl"], "comps": cur["comps"]}, minutes))
        for team in realistic:
            r, o, t = realistic[team], oracle.get(team, []), truth.get(team, [])
            if not o or not t:
                continue
            recs.append({"year_id": y, "team_abbr": team,
                         "x_real": team_feature([p for p, _ in r], [w for _, w in r]),
                         "x_orac": team_feature([p for p, _ in o], [w for _, w in o]),
                         "x_true": team_feature([p for p, _ in t], [w for _, w in t]),
                         "known_share": float(np.average([p["known"] for p, _ in r],
                                                         weights=[w for _, w in r]))})
    return pd.DataFrame(recs)


def roster_entries(years):
    out = []
    for y in years:
        for pid, st in team_stints(y).items():
            out.append((pid, y, sum(m for _, _, m in st)))
    return out


# ── 팀 모델 ─────────────────────────────────────────────────────────────

def design(df, key, spec):
    X = np.array(df[key].tolist())
    cols = [np.ones(len(df))]
    if spec in ("ndl", "comp", "comp+prev"):
        cols.append(X[:, 0])
    if spec in ("comp", "comp+prev"):
        cols.extend(X[:, 1:].T)
    if spec == "comp+prev":
        cols.extend([df["prev_rel_ortg"].values, df["prev_rel_drtg"].values])
    return np.column_stack(cols)


def ols(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def center(df, col):
    """한 시즌 30팀의 상대 레이팅 합은 0이어야 한다. 예측값도 시즌별로 맞춘다."""
    return df[col] - df.groupby("year_id")[col].transform("mean")


def loso_mae(df, key, spec):
    """학습 구간 안에서 시즌 하나씩 빼고 맞혀보는 교차검증. 모델 선택은 이것으로만 한다."""
    errs = []
    for y in sorted(df["year_id"].unique()):
        tr, te = df[df["year_id"] != y], df[df["year_id"] == y]
        for tgt in ("rel_ortg", "rel_drtg"):
            b = ols(design(tr, key, spec), tr[tgt].values)
            p = design(te, key, spec) @ b
            errs.extend(np.abs((p - p.mean()) - te[tgt].values))
    return float(np.mean(errs))


def prev_ratings(df, rt):
    idx = {(r.year_id, r.team_abbr): r for r in rt.itertuples()}

    def find(y, t):
        for code in [t, *PREDECESSOR.get(t, [])]:
            if (y - 1, code) in idx:
                return idx[(y - 1, code)]
        return None
    prev = [find(r.year_id, r.team_abbr) for r in df.itertuples()]
    df["prev_rel_ortg"] = [p.rel_ortg if p is not None else 0.0 for p in prev]
    df["prev_rel_drtg"] = [p.rel_drtg if p is not None else 0.0 for p in prev]
    return df


def evaluate(te, k_win):
    """홀드아웃 에러. 예측 컬럼 접두어별로 ORtg·DRtg·NRtg·승수 MAE와 순위 상관."""
    out = {}
    for name in sorted({c[:-5] for c in te.columns if c.endswith("_ortg") and c.startswith("p_")}):
        o, d = te[f"{name}_ortg"], te[f"{name}_drtg"]
        n = o - d
        wins = te["games"] * (0.5 + k_win * n)
        true_n = te["rel_ortg"] - te["rel_drtg"]
        rho = [spearmanr(n[te["year_id"] == y], true_n[te["year_id"] == y])[0]
               for y in sorted(te["year_id"].unique())] if n.std() else [np.nan]
        out[name[2:]] = {
            "maeO": round(float(np.mean(np.abs(o - te["rel_ortg"]))), 2),
            "maeD": round(float(np.mean(np.abs(d - te["rel_drtg"]))), 2),
            "maeN": round(float(np.mean(np.abs(n - true_n))), 2),
            "maeW": round(float(np.mean(np.abs(wins - te["w"]))), 2),
            "corrO": round(float(np.corrcoef(o, te["rel_ortg"])[0, 1]), 3) if o.std() else 0.0,
            "corrD": round(float(np.corrcoef(d, te["rel_drtg"])[0, 1]), 3) if d.std() else 0.0,
            "corrN": round(float(np.corrcoef(n, true_n)[0, 1]), 3) if n.std() else 0.0,
            "rankRho": None if np.isnan(rho).all() else round(float(np.nanmean(rho)), 3),
            "playoff": playoff_hits(te, n),
        }
    return out


def playoff_hits(te, n):
    """컨퍼런스별 상위 6팀(직행)을 몇 팀 맞혔나. 시즌 평균, 12팀 만점."""
    hits = []
    for y in sorted(te["year_id"].unique()):
        s = te[te["year_id"] == y].assign(pn=n[te["year_id"] == y])
        h = 0
        for _, c in s.groupby("conf"):
            h += len(set(c.nlargest(6, "pn")["team_abbr"]) & set(c.nlargest(6, "wpct")["team_abbr"]))
        hits.append(h)
    return round(float(np.mean(hits)), 2)


# ── 실행 ─────────────────────────────────────────────────────────────────

def main():
    rows_q, meta = bp.load()
    rows_all = load_all_rows()
    rt = load_ratings()

    # 1) 홀드아웃: 선수 모델·팀 모델 모두 ~2015년으로만 학습
    curve, _, _ = bp.aging_curve(rows_q, upto=TRAIN_END)
    coef, _ = bp.fit(rows_q, curve, TRAIN_END)
    pm = PlayerModel(rows_q, rows_all, curve, coef)
    years = list(range(FIRST_TARGET, TARGET))
    pm.fit_fallback(roster_entries(range(FIRST_TARGET, TRAIN_END + 1)))
    print(f"기록 없는 선수 대체값: NDL {pm.fallback[0]:.1f}, 경기당 {pm.fallback[2]:.1f}분")

    ds = build_dataset(pm, years).merge(rt, on=["year_id", "team_abbr"], how="inner")
    ds = prev_ratings(ds, rt)
    tr, te = ds[ds["year_id"] <= TRAIN_END].copy(), ds[ds["year_id"] > TRAIN_END].copy()
    print(f"팀-시즌  학습 {len(tr)} / 홀드아웃 {len(te)}   "
          f"(현실 버전에서 기록 있는 선수 출전시간 비중 {ds['known_share'].mean():.0%})\n")

    # 승률 ≈ 0.5 + k × 순레이팅. k도 학습 구간에서만.
    k_win = float(np.polyfit(tr["rel_ortg"] - tr["rel_drtg"], tr["wpct"] - 0.5, 1)[0])

    specs = ["ndl", "comp", "comp+prev"]
    print("학습 구간 교차검증 (시즌 하나씩 빼기, ORtg·DRtg 평균 MAE)")
    cv = {s: loso_mae(tr, "x_real", s) for s in specs}
    for s in specs:
        print(f"  {s:<10}{cv[s]:.3f}")
    best = min(cv, key=cv.get)
    print(f"  → 채택: {best}\n")

    for tgt in ("ortg", "drtg"):
        te[f"p_zero_{tgt}"] = 0.0
        te[f"p_prev_{tgt}"] = center(te, f"prev_rel_{tgt}")
        for key, tag in (("x_real", "real"), ("x_orac", "orac"), ("x_true", "true")):
            for s in specs:
                b = ols(design(tr, key, s), tr[f"rel_{tgt}"].values)
                te[f"p_{tag}_{s}_{tgt}"] = design(te, key, s) @ b
                te[f"p_{tag}_{s}_{tgt}"] = center(te, f"p_{tag}_{s}_{tgt}")
    report = evaluate(te, k_win)

    print(f"홀드아웃 {te['year_id'].min()}~{te['year_id'].max()} ({len(te)}팀-시즌)")
    print(f"{'모델':<20}{'ORtg':>6}{'DRtg':>6}{'NRtg':>6}{'승수':>6}"
          f"{'상관O':>7}{'상관D':>7}{'상관N':>7}{'순위ρ':>7}{'상위6':>6}")
    for name, r in report.items():
        print(f"{name:<20}{r['maeO']:>6.2f}{r['maeD']:>6.2f}{r['maeN']:>6.2f}{r['maeW']:>6.2f}"
              f"{r['corrO']:>7.3f}{r['corrD']:>7.3f}{r['corrN']:>7.3f}"
              f"{r['rankRho'] if r['rankRho'] is not None else '-':>7}{r['playoff']:>6.2f}")

    # 시즌별로 쪼개 본다. 평균 하나가 특정 시즌의 실패를 가리지 않는지 확인용.
    true_n = te["rel_ortg"] - te["rel_drtg"]
    print(f"\n시즌별 NRtg MAE   {'작년 그대로':>10}{'로스터 NDL':>10}")
    by_season = []
    for y, s in te.groupby("year_id"):
        tn = true_n[s.index]
        e_prev = np.abs(s["p_prev_ortg"] - s["p_prev_drtg"] - tn).mean()
        e_mod = np.abs(s[f"p_real_{best}_ortg"] - s[f"p_real_{best}_drtg"] - tn).mean()
        by_season.append({"season": f"{y - 1}-{str(y)[-2:]}",
                          "prev": round(float(e_prev), 2), "model": round(float(e_mod), 2)})
        print(f"  {y - 1}-{str(y)[-2:]}       {e_prev:>10.2f}{e_mod:>10.2f}"
              f"{'' if e_mod < e_prev else '   ← 기준선이 나음'}")
    te.to_pickle(CACHE_DIR / "team_holdout.pkl")   # 사례 분석용 (재생성 가능)

    # 피처 해석용: 학습 구간 계수 (현실 버전, 구성요소 모델)
    coefs = {}
    for tgt in ("ortg", "drtg"):
        b = ols(design(tr, "x_real", "comp"), tr[f"rel_{tgt}"].values)
        coefs[tgt] = [round(float(v), 3) for v in b]
    print("\n구성요소 모델 계수 (현실 버전, 학습 구간)")
    print(f"  {'':<8}{'절편':>8}{'예측NDL':>8}" + "".join(f"{c:>8}" for c in COMP_LABEL))
    for tgt in ("ortg", "drtg"):
        print(f"  {tgt:<8}" + "".join(f"{v:>8.2f}" for v in coefs[tgt]))

    # 2) 2026-27 예측: 선수 모델·팀 모델을 전 구간으로 다시 학습
    curve_f, _, _ = bp.aging_curve(rows_q)
    coef_f, _ = bp.fit(rows_q, curve_f, upto=10**9)
    pm_f = PlayerModel(rows_q, rows_all, curve_f, coef_f)
    pm_f.fit_fallback(roster_entries(range(FIRST_TARGET, TARGET)))
    k_win_f = float(np.polyfit(ds["rel_ortg"] - ds["rel_drtg"], ds["wpct"] - 0.5, 1)[0])

    roster = pd.read_csv(ROSTER_CSV, dtype={"exp": str}, keep_default_na=False)
    # 백테스트 로스터는 '그 시즌에 실제로 뛴 선수'라 시즌을 통째로 날린 선수가 없다.
    # 조건을 맞추려 시즌아웃이 확정된 선수만 뺀다. 단기 부상자는 그대로 둔다.
    out_season = roster["injury"].str.startswith("Out For Season")
    print(f"\n시즌아웃 제외 {out_season.sum()}명: " + ", ".join(roster.loc[out_season, "player"]))
    roster = roster[~out_season]
    cur_rt = pd.read_csv(RATINGS_CSV).query("year_id == @TARGET")
    conf = dict(zip(cur_rt["team_abbr"], cur_rt["conf"]))
    teams, players_out = [], []
    for team, g in roster.groupby("team_abbr"):
        pls = [pm_f.player(pid, TARGET) for pid in g["id"]]
        feat = team_feature(pls, [p["mpg"] for p in pls])
        teams.append({"team_abbr": team, "x_real": feat, "year_id": TARGET})
        tot = sum(p["mpg"] for p in pls)
        for (_, r), p in zip(g.iterrows(), pls):
            players_out.append({"team": team, "id": r["id"], "player": r["player"],
                                "projected": round(p["proj"], 1), "known": p["known"],
                                "share": round(p["mpg"] / tot, 3), "injury": r["injury"],
                                "twoWay": bool(r["two_way"] == "True" or r["two_way"] is True)})
    cur = prev_ratings(pd.DataFrame(teams), rt)
    for tgt in ("ortg", "drtg"):
        b = ols(design(ds, "x_real", best), ds[f"rel_{tgt}"].values)
        cur[f"p_{tgt}"] = design(cur, "x_real", best) @ b
        cur[f"p_{tgt}"] -= cur[f"p_{tgt}"].mean()
    cur["p_nrtg"] = cur["p_ortg"] - cur["p_drtg"]
    cur["p_wins"] = 82 * (0.5 + k_win_f * cur["p_nrtg"])
    cur["conf"] = cur["team_abbr"].map(conf)
    cur["proj_ndl"] = [x[0] for x in cur["x_real"]]
    cur["actual_w"] = cur["team_abbr"].map(dict(zip(cur_rt["team_abbr"], cur_rt["w"])))
    cur["actual_l"] = cur["team_abbr"].map(dict(zip(cur_rt["team_abbr"], cur_rt["l"])))

    print(f"\n2026-27 예측 (모델 {best}, 승률 계수 {k_win_f:.4f}/점)")
    for c in ("E", "W"):
        s = cur[cur["conf"] == c].sort_values("p_nrtg", ascending=False)
        print(f"\n  {'동부' if c == 'E' else '서부'}  {'팀':<5}{'예상승':>6}{'ORtg':>7}{'DRtg':>7}"
              f"{'NRtg':>7}{'팀NDL':>7}{'현재':>7}")
        for i, r in enumerate(s.itertuples(), 1):
            print(f"  {i:>3}   {r.team_abbr:<5}{r.p_wins:>6.1f}{r.p_ortg:>+7.1f}{r.p_drtg:>+7.1f}"
                  f"{r.p_nrtg:>+7.1f}{r.proj_ndl:>7.1f}   {int(r.actual_w)}-{int(r.actual_l)}")

    payload = {
        "season": "2026-27",
        "model": {"chosen": best, "cv": {k: round(v, 3) for k, v in cv.items()},
                  "holdout": report, "trainSeasons": [FIRST_TARGET, TRAIN_END],
                  "holdoutSeasons": [TRAIN_END + 1, TARGET - 1], "winCoef": round(k_win_f, 4),
                  "compCoef": coefs, "bySeason": by_season},
        "teams": [{"team": r.team_abbr, "conf": r.conf, "ortg": round(r.p_ortg, 2),
                   "drtg": round(r.p_drtg, 2), "nrtg": round(r.p_nrtg, 2),
                   "wins": round(r.p_wins, 1), "teamNdl": round(r.proj_ndl, 2)}
                  for r in cur.sort_values("p_nrtg", ascending=False).itertuples()],
        "players": players_out,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"\n저장 {OUT.name}")


if __name__ == "__main__":
    main()
