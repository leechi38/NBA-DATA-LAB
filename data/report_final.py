"""README에 싣는 최종 수치를 한 기준으로 통일해 산출한다.

주의점: 가중치 학습은 '구성요소 가중합'에서 이뤄지지만, 실제 배포되는 NDL Score는 여기에
출전시간 보정까지 적용한 값이다. 두 기준의 상관계수가 약 0.02 차이나므로, 보고하는 숫자는
전부 '실제 배포되는 값' 기준으로 맞춘다.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
AXES = ["NDL_Production", "NDL_Efficiency", "NDL_Playmaking", "NDL_Defense", "NDL_WinImpact"]
OLD = np.array([0.25, 0.20, 0.15, 0.20, 0.20])
NEW = np.array([0.090, 0.297, 0.093, 0.120, 0.400])

SPLITS = [
    ("학습용 (1952~2015 홀수)", lambda y: (y <= 2015) & (y % 2 == 1)),
    ("검증용 (1952~2015 짝수)", lambda y: (y <= 2015) & (y % 2 == 0)),
    ("최종 시험용 (2016~2024)", lambda y: y >= 2016),
    ("전체", lambda y: y > 0),
]


def load() -> pd.DataFrame:
    rows = []
    for f in sorted((BASE / "seasons").glob("players_*.json"),
                    key=lambda p: int(p.stem.split("_")[1])):
        d = json.loads(f.read_text(encoding="utf-8"))
        for p in d["players"]:
            t = p.get("Team")
            if not t or "TM" in str(t) or p.get("MP") is None or p.get("G") is None:
                continue
            if any(p.get(a) is None for a in AXES) or p.get("NDL_Weight") is None:
                continue
            rows.append({
                "year_id": d["generated"], "Team": t, "w": p["MP"] * p["G"],
                "pt": p["NDL_Weight"], "WS": p.get("WS"), "BPM": p.get("BPM"),
                **{a: p[a] for a in AXES},
            })
    return pd.DataFrame(rows)


def team_corr(df, values) -> pd.Series:
    """선수 점수 → 출전시간 가중 팀 평균 → 팀 실제 득실차와의 상관계수(시즌별)."""
    d = df.assign(v=values).dropna(subset=["v"])
    d["wv"] = d["v"] * d["w"]
    agg = d.groupby(["year_id", "Team"]).agg(wv=("wv", "sum"), w=("w", "sum")).reset_index()
    agg["v"] = agg["wv"] / agg["w"]
    margins = pd.read_csv(BASE / "team_margins.csv").rename(columns={"team_abbr": "Team"})
    m = agg.merge(margins[["year_id", "Team", "margin"]], on=["year_id", "Team"])
    m = m[m.groupby("year_id")["Team"].transform("size") >= 8]
    return m.groupby("year_id").apply(lambda g: g["v"].corr(g["margin"]), include_groups=False)


def main():
    df = load()
    comp = df[AXES].values
    # 실제 배포값 기준: 가중합에 출전시간 보정을 곱한 점수
    old_corr = team_corr(df, (comp @ OLD) * df["pt"])
    new_corr = team_corr(df, (comp @ NEW) * df["pt"])
    ws_corr = team_corr(df, df["WS"])
    bpm_corr = team_corr(df, df["BPM"])

    print("■ 가중치 학습 전후 (실제 배포되는 NDL Score 기준)\n")
    print(f"{'구간':<26}{'시즌':>5}{'기존':>9}{'학습후':>9}{'개선':>9}{'개선시즌':>10}")
    for name, pred in SPLITS:
        o, n = old_corr[pred(old_corr.index)], new_corr[pred(new_corr.index)]
        print(f"{name:<26}{len(o):>5}{o.mean():>9.3f}{n.mean():>9.3f}"
              f"{n.mean() - o.mean():>+9.3f}{f'{int((n > o).sum())}/{len(o)}':>10}")

    print("\n■ 공인 지표와 비교 (전 시즌 평균)\n")
    print(f"  NDL Score {new_corr.mean():.3f}   WS {ws_corr.mean():.3f}   BPM {bpm_corr.mean():.3f}")
    print(f"  (WS는 {len(ws_corr.dropna())}개, BPM은 {len(bpm_corr.dropna())}개 시즌에서 산출 가능)")

    payload = {
        "basis": "실제 배포되는 NDL Score(출전시간 보정 포함) 기준",
        "splits": {name: {"seasons": int(len(old_corr[pred(old_corr.index)])),
                           "old": round(float(old_corr[pred(old_corr.index)].mean()), 4),
                           "new": round(float(new_corr[pred(new_corr.index)].mean()), 4),
                           "improved": int((new_corr[pred(new_corr.index)] >
                                            old_corr[pred(old_corr.index)]).sum())}
                    for name, pred in SPLITS},
        "vs_public": {"NDL": round(float(new_corr.mean()), 4),
                       "WS": round(float(ws_corr.mean()), 4),
                       "BPM": round(float(bpm_corr.mean()), 4)},
    }
    (BASE / "report_final.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n저장: report_final.json")


if __name__ == "__main__":
    main()
