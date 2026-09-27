"""NBA 전 시즌(1950~) 선수 스탯 수집 + NDL Score 계산.

fetch_stats.py(최신 시즌 전용, build_pages.py가 참조)의 로직을 시즌별로
반복 실행할 수 있게 재구성한 버전. 시즌마다 data/seasons/players_{season}.json
으로 개별 저장하며, 이미 저장된 시즌은 건너뛰어 중단 후 재실행이 가능하다.

옛날 시즌 대응:
- shooting(슛 구역별) 테이블은 1996-97 시즌부터만 존재 → 없으면 관련 컬럼을
  전부 NaN으로 채우고 계속 진행(has_shooting_data=False로 표시).
- 스틸/블록(1973-74 이전), 턴오버(1977-78 이전), 3점슛(1979-80 이전)이
  아예 없는 시즌은 해당 컬럼 자체가 테이블에 없을 수 있어 ensure_cols로
  NaN 컬럼을 미리 만들어 KeyError를 방지한다.
- zscore_ref는 표본 전체가 NaN(해당 스탯이 그 시대에 존재하지 않음)이면
  0(중립)을 반환하도록 수정 — 없는 스탯이 점수를 깎거나 NaN으로 오염시키지 않게.
"""
import io
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

REF_MIN_G = 15
REF_MINUTES = 2000
REF_ZONE_ATTEMPTS = 3.0
POS_GROUPS = {"PG": "G", "SG": "G", "SF": "F", "PF": "F", "C": "C"}

SEASON_START = 1950  # BAA/NBL 통합 이후 "NBA" 첫 시즌
SEASON_END = 2026    # 가장 최근 시즌
SHOOTING_FIRST_SEASON = 1997  # 슛 구역별(shooting) 테이블이 존재하는 첫 시즌

# NDL Score 구성요소 가중치.
#
# 초기값은 설계자가 임의로 정한 25/20/15/20/20이었다. 이후 팀별 실제 경기당
# 득실 마진(538 ELO 게임 로그 1952~2015)을 정답값으로 두고, 팀 소속 선수들의
# 구성요소 출전시간 가중 평균이 그 마진을 얼마나 설명하는지로 가중치를 학습했다.
#   - 학습: 홀수 연도 시즌 / 검증: 짝수 연도 시즌 (시즌 단위 분리로 정보 누수 차단)
#   - 무제약 최적화는 승리기여(WS·VORP 기반)에 83%를 몰아줬는데, 두 지표 자체가
#     팀 성적에서 역산되는 값이라 순환논리가 된다. 그래서 각 요소를 5~40%로
#     제한해 5개 축이 모두 살아있도록 했다.
#   - 결과: 검증 시즌 평균 상관계수 0.8661 → 0.9099
# 산출 스크립트: optimize_weights2.py (결과는 optimized_weights.json)
WEIGHTS = {"production": 0.090, "efficiency": 0.297, "playmaking": 0.093,
           "defense": 0.120, "win_impact": 0.400}

SHOOTING_COLS = [
    "Rk", "Player", "Age", "Team", "Pos", "G", "GS", "MP", "FG%",
    "Dist_Avg", "PctFGA_2P", "PctFGA_0_3", "PctFGA_3_10", "PctFGA_10_16",
    "PctFGA_16_3P", "PctFGA_3P",
    "FGpct_2P", "FGpct_0_3", "FGpct_3_10", "FGpct_10_16", "FGpct_16_3P", "FGpct_3P",
    "AstPct_2P", "AstPct_3P", "DunkPctFGA", "DunkCount",
    "Corner3PctOf3PA", "Corner3Pct", "HalfCourt_Att", "HalfCourt_Md", "Awards",
]

NUM_COLS_PG = ["Age", "G", "GS", "MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%",
               "2P", "2PA", "2P%", "eFG%", "FT", "FTA", "FT%", "ORB", "DRB",
               "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]

NUM_COLS_ADV = ["PER", "TS%", "3PAr", "FTr", "ORB%", "DRB%", "TRB%", "AST%",
                "STL%", "BLK%", "TOV%", "USG%", "OWS", "DWS", "WS", "WS/48",
                "OBPM", "DBPM", "BPM", "VORP"]

NUM_COLS_SHOOTING = ["FGpct_0_3", "FGpct_3_10", "FGpct_10_16", "FGpct_16_3P",
                      "FGpct_3P", "PctFGA_0_3", "PctFGA_3_10", "PctFGA_10_16",
                      "PctFGA_16_3P", "PctFGA_3P", "Corner3PctOf3PA", "Corner3Pct",
                      "AstPct_2P", "AstPct_3P", "DunkPctFGA"]

BASE = Path(__file__).parent
CACHE_DIR = BASE / ".cache"
SEASONS_DIR = BASE / "seasons"
ID_PATTERN = re.compile(r'data-append-csv="([^"]+)"[^>]*>\s*<a[^>]*>([^<]+)</a>')
HEADSHOT_VERSION_PATTERN = re.compile(r"/req/(\d+)/images/headshots/")


def fetch_html(url: str, cache_name: str) -> str:
    """캐시 우선. 실제 다운로드가 일어난 경우에만 예의상 간격을 둔다.

    캐시가 채워진 뒤의 재계산은 네트워크를 전혀 타지 않으므로, sleep을 호출부가
    아니라 여기에 두어야 재실행이 수 분씩 헛돌지 않는다.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"{cache_name}.html"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=20).read().decode("utf-8")
    cache_path.write_text(html, encoding="utf-8")
    time.sleep(3)
    return html


def parse_table(html: str, table_id: str) -> pd.DataFrame:
    df = pd.read_html(io.StringIO(html), attrs={"id": table_id})[0]
    df = df[df["Player"] != "Player"]
    df = df[df["Player"] != "League Average"]
    df = df.dropna(subset=["Player", "Team"])
    return df


def extract_player_ids(html: str) -> dict:
    mapping = {}
    for pid, name in ID_PATTERN.findall(html):
        mapping.setdefault(name, pid)
    return mapping


def discover_headshot_version(sample_bbref_id: str) -> str:
    try:
        html = fetch_html(
            f"https://www.basketball-reference.com/players/{sample_bbref_id[0]}/{sample_bbref_id}.html",
            "sample_player_page",
        )
        m = HEADSHOT_VERSION_PATTERN.search(html)
        return m.group(1) if m else "202605210"
    except Exception:
        return "202605210"


def parse_shooting_table(html: str) -> pd.DataFrame:
    df = pd.read_html(io.StringIO(html), attrs={"id": "shooting"})[0]
    df.columns = SHOOTING_COLS
    df = df[df["Player"] != "Player"]
    df = df[df["Player"] != "League Average"]
    df = df.dropna(subset=["Player", "Team"])
    return df


def dedupe_traded_players(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for player, g in df.groupby("Player", sort=False):
        if len(g) == 1:
            rows.append(g.iloc[0].to_dict())
            continue
        combined = g[g["Team"].str.contains("TM", na=False)]
        individual = g[~g["Team"].str.contains("TM", na=False)]
        base = (combined.iloc[0] if len(combined) else g.iloc[0]).to_dict()
        if len(individual):
            base["Team"] = individual.iloc[-1]["Team"]
            base["TeamsPlayed"] = "/".join(individual["Team"].tolist())
        else:
            base["TeamsPlayed"] = base["Team"]
        rows.append(base)
    return pd.DataFrame(rows)


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def ensure_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """옛날 시즌 테이블엔 아예 없는 컬럼(3P, STL, BLK, TOV 등)을 NaN으로 채워 넣어
    이후 단계에서 KeyError가 나지 않게 한다."""
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df


def zscore_ref(series: pd.Series, ref_mask: pd.Series) -> pd.Series:
    mean = series[ref_mask].mean()
    std = series[ref_mask].std(ddof=0)
    if std == 0 or np.isnan(std):
        # 해당 스탯이 이 시대에 아예 존재하지 않으면(전부 NaN) 중립(0)으로 처리.
        # series * 0 은 series가 전부 NaN일 때 NaN을 그대로 반환하므로 쓰면 안 됨.
        return pd.Series(0.0, index=series.index)
    return (series - mean) / std


def pos_zscore_ref(df: pd.DataFrame, col: str, ref_mask: pd.Series) -> pd.Series:
    result = pd.Series(np.nan, index=df.index, dtype=float)
    for grp, sub in df.groupby("PosGroup"):
        mask = ref_mask & (df["PosGroup"] == grp)
        mean = df.loc[mask, col].mean()
        std = df.loc[mask, col].std(ddof=0)
        if std == 0 or np.isnan(std):
            result.loc[sub.index] = 0
        else:
            result.loc[sub.index] = (sub[col] - mean) / std
    return result.fillna(0)


def pct_scale(series: pd.Series, ref_mask: pd.Series, invert: bool = False,
              fillna_value=None) -> pd.Series:
    s = series.astype(float).copy()
    if fillna_value is not None:
        s = s.fillna(fillna_value)
    ref_vals = np.sort(s[ref_mask].dropna().values)
    if len(ref_vals) == 0:
        return pd.Series(50.0, index=series.index)
    pct = np.searchsorted(ref_vals, s.values, side="right") / len(ref_vals) * 100
    pct = np.clip(pct, 1, 99)
    if invert:
        pct = 100 - pct
    return pd.Series(pct, index=series.index)


def primary_pos_group(pos) -> str | None:
    if pos is None or (isinstance(pos, float) and np.isnan(pos)):
        return None
    primary = str(pos).split("-")[0].strip()
    return POS_GROUPS.get(primary)


def fetch_season_tables(season: int):
    per_game_url = f"https://www.basketball-reference.com/leagues/NBA_{season}_per_game.html"
    advanced_url = f"https://www.basketball-reference.com/leagues/NBA_{season}_advanced.html"
    shooting_url = f"https://www.basketball-reference.com/leagues/NBA_{season}_shooting.html"

    pg_html = fetch_html(per_game_url, f"per_game_stats_html_{season}")
    pg = parse_table(pg_html, "per_game_stats")
    id_map = extract_player_ids(pg_html)

    adv_html = fetch_html(advanced_url, f"advanced_html_{season}")
    adv = parse_table(adv_html, "advanced")

    has_shooting = season >= SHOOTING_FIRST_SEASON
    shooting = pd.DataFrame(columns=["Player", "Team"])
    if has_shooting:
        try:
            shooting_html = fetch_html(shooting_url, f"shooting_html_{season}")
            shooting = parse_shooting_table(shooting_html)
        except Exception as exc:
            print(f"  [{season}] shooting 테이블 수집 실패, 없이 진행: {exc}")
            has_shooting = False

    return pg, adv, shooting, id_map, has_shooting


def process_season(season: int) -> dict:
    pg, adv, shooting, id_map, has_shooting = fetch_season_tables(season)

    pg = dedupe_traded_players(pg)
    adv = dedupe_traded_players(adv)

    pg = ensure_cols(pg, NUM_COLS_PG)
    for c in NUM_COLS_PG:
        pg[c] = to_num(pg[c])

    adv = ensure_cols(adv, NUM_COLS_ADV)
    for c in NUM_COLS_ADV:
        adv[c] = to_num(adv[c])

    if has_shooting and len(shooting):
        shooting = dedupe_traded_players(shooting)
        shooting = ensure_cols(shooting, NUM_COLS_SHOOTING)
        for c in NUM_COLS_SHOOTING:
            shooting[c] = to_num(shooting[c])
        merged = pg.merge(
            adv[["Player", "Team"] + NUM_COLS_ADV],
            on=["Player", "Team"], how="left", suffixes=("", "_adv"),
        ).merge(
            shooting[["Player", "Team"] + NUM_COLS_SHOOTING],
            on=["Player", "Team"], how="left", suffixes=("", "_shoot"),
        )
    else:
        merged = pg.merge(
            adv[["Player", "Team"] + NUM_COLS_ADV],
            on=["Player", "Team"], how="left", suffixes=("", "_adv"),
        )
        merged = ensure_cols(merged, NUM_COLS_SHOOTING)

    merged["PosGroup"] = merged["Pos"].map(primary_pos_group)
    merged["BbrefId"] = merged["Player"].map(id_map)
    merged["TotalMP"] = merged["MP"] * merged["G"]
    ref_mask = merged["G"] >= REF_MIN_G

    def z(col: str) -> pd.Series:
        """개별 스탯 z-score. 그 선수만 기록이 없는 경우(부분 결측)는 중립(0)으로.

        zscore_ref는 '컬럼 전체가 없는 시대'만 0으로 처리한다. 하지만 1970-71의
        STL/BLK/VORP, 1973-74의 TOV처럼 일부 선수만 값이 없는 시즌이 있고, 이때
        NaN이 합계로 전파되면 해당 선수의 NDL_Score 자체가 통째로 사라진다.
        (실제로 1970-71~1974-75 시즌 선수의 90% 이상이 이 경로로 null이었다.)
        """
        return zscore_ref(merged[col], ref_mask).fillna(0)

    production = (z("PTS") + z("TRB") + z("AST") + z("STL") + z("BLK") - z("TOV"))
    win_impact = z("WS") + z("VORP")

    efficiency = (
        pos_zscore_ref(merged, "TS%", ref_mask) + pos_zscore_ref(merged, "eFG%", ref_mask)
    )
    playmaking = (
        pos_zscore_ref(merged, "AST%", ref_mask) - pos_zscore_ref(merged, "TOV%", ref_mask)
    )
    defense = (
        pos_zscore_ref(merged, "STL%", ref_mask) + pos_zscore_ref(merged, "BLK%", ref_mask)
        + pos_zscore_ref(merged, "DRB%", ref_mask)
    )

    raw_score = (
        WEIGHTS["production"] * zscore_ref(production, ref_mask).fillna(0)
        + WEIGHTS["efficiency"] * zscore_ref(efficiency, ref_mask).fillna(0)
        + WEIGHTS["playmaking"] * zscore_ref(playmaking, ref_mask).fillna(0)
        + WEIGHTS["defense"] * zscore_ref(defense, ref_mask).fillna(0)
        + WEIGHTS["win_impact"] * zscore_ref(win_impact, ref_mask).fillna(0)
    )

    # MP/G가 누락된 초기 시즌 일부 행 → 가중치 0(=점수를 평균으로 수축)으로 처리
    pt_weight = (merged["TotalMP"] / REF_MINUTES).clip(lower=0, upper=1).fillna(0)
    adjusted_score = raw_score * pt_weight

    sigma_raw = raw_score[ref_mask].std(ddof=0)
    if sigma_raw == 0 or np.isnan(sigma_raw):
        ndl = pd.Series(50.0, index=merged.index)
    else:
        ndl = 50 + 10 * (adjusted_score / sigma_raw)
    ndl = ndl.clip(lower=0, upper=100)

    merged["NDL_Production"] = zscore_ref(production, ref_mask).round(2)
    merged["NDL_Efficiency"] = zscore_ref(efficiency, ref_mask).round(2)
    merged["NDL_Playmaking"] = zscore_ref(playmaking, ref_mask).round(2)
    merged["NDL_Defense"] = zscore_ref(defense, ref_mask).round(2)
    merged["NDL_WinImpact"] = zscore_ref(win_impact, ref_mask).round(2)
    merged["NDL_Weight"] = pt_weight.round(2)
    merged["NDL_Score"] = ndl.round(1)

    def pool_mean(col):
        return merged.loc[ref_mask, col].mean()

    def rate_and_volume(rate_col, volume_col, invert=False):
        rate_pct = pct_scale(merged[rate_col], ref_mask, invert=invert, fillna_value=pool_mean(rate_col))
        vol_pct = pct_scale(merged[volume_col], ref_mask, invert=invert, fillna_value=pool_mean(volume_col))
        return (rate_pct + vol_pct) / 2

    teamwork_ratio = merged["AST%"] / merged["USG%"].replace(0, np.nan)
    starter_ratio = merged["GS"] / merged["G"].replace(0, np.nan)

    share_ra = merged["PctFGA_0_3"].fillna(0)
    share_paint = merged["PctFGA_3_10"].fillna(0) + merged["PctFGA_10_16"].fillna(0)
    share_mid = merged["PctFGA_16_3P"].fillna(0)
    corner_of_3pa = merged["Corner3PctOf3PA"].fillna(0)
    pct_fga_3p = merged["PctFGA_3P"].fillna(0)
    share_corner3 = pct_fga_3p * corner_of_3pa
    share_arc3 = pct_fga_3p * (1 - corner_of_3pa)

    zone_shares_raw = {"RA": share_ra, "Paint": share_paint, "Midrange": share_mid,
                        "Corner3": share_corner3, "Arc3": share_arc3}
    total_share = sum(zone_shares_raw.values())
    zone_shares = {
        name: pd.Series(np.where(total_share > 0, s / total_share.replace(0, np.nan), 0.2), index=merged.index)
        for name, s in zone_shares_raw.items()
    }

    zone_conf = {
        name: (zone_shares[name] * merged["FGA"] / REF_ZONE_ATTEMPTS).clip(0, 1)
        for name in zone_shares
    }

    paint_fg_input = merged[["FGpct_3_10", "FGpct_10_16"]].mean(axis=1, skipna=True)
    denom_arc = (1 - corner_of_3pa)
    arc3_fg = pd.Series(
        np.where(denom_arc > 0.05,
                 (merged["FGpct_3P"] - merged["Corner3Pct"] * corner_of_3pa) / denom_arc,
                 merged["FGpct_3P"]),
        index=merged.index,
    )

    zone_skill_raw = {
        "Finishing": pct_scale(merged["FGpct_0_3"], ref_mask, fillna_value=pool_mean("FGpct_0_3")),
        "Paint": pct_scale(paint_fg_input, ref_mask, fillna_value=paint_fg_input[ref_mask].mean()),
        "Midrange": pct_scale(merged["FGpct_16_3P"], ref_mask, fillna_value=pool_mean("FGpct_16_3P")),
        "Corner3": pct_scale(merged["Corner3Pct"], ref_mask, fillna_value=pool_mean("FGpct_3P")),
        "Arc3": pct_scale(arc3_fg, ref_mask, fillna_value=pool_mean("FGpct_3P")),
    }
    zone_conf_map = {"Finishing": "RA", "Paint": "Paint", "Midrange": "Midrange",
                      "Corner3": "Corner3", "Arc3": "Arc3"}

    raw = {
        "Finishing": zone_skill_raw["Finishing"],
        "Paint": zone_skill_raw["Paint"],
        "Midrange": zone_skill_raw["Midrange"],
        "Corner3": zone_skill_raw["Corner3"],
        "Arc3": zone_skill_raw["Arc3"],
        "FreeThrow": pct_scale(merged["FT%"], ref_mask, fillna_value=pool_mean("FT%")),
        "Playmaking": rate_and_volume("AST%", "AST"),
        "BallHandling": rate_and_volume("TOV%", "TOV", invert=True),
        "ShotCreation": pct_scale(merged["AstPct_2P"], ref_mask, invert=True, fillna_value=pool_mean("AstPct_2P")),
        "OnBallDefense": rate_and_volume("STL%", "STL"),
        "RimProtection": rate_and_volume("BLK%", "BLK"),
        "DefRebound": rate_and_volume("DRB%", "DRB"),
        "OffRebound": rate_and_volume("ORB%", "ORB"),
        "DefensiveIQ": pct_scale(merged["DBPM"], ref_mask, fillna_value=pool_mean("DBPM")),
        "Teamwork": pct_scale(teamwork_ratio, ref_mask, fillna_value=teamwork_ratio[ref_mask].mean()),
        "Composure": (pct_scale(merged["FT%"], ref_mask, fillna_value=pool_mean("FT%"))
                      + pct_scale(merged["TOV%"], ref_mask, invert=True, fillna_value=pool_mean("TOV%"))) / 2,
        "WinningMentality": pct_scale(merged["BPM"], ref_mask, fillna_value=pool_mean("BPM")),
        "Experience": (pct_scale(merged["Age"], ref_mask, fillna_value=pool_mean("Age"))
                       + pct_scale(starter_ratio, ref_mask, fillna_value=0)) / 2,
        "Athleticism": (0.5 * pct_scale(merged["DunkPctFGA"], ref_mask, fillna_value=0)
                        + 0.3 * pct_scale(merged["BLK%"], ref_mask, fillna_value=pool_mean("BLK%"))
                        + 0.2 * pct_scale(merged["ORB%"], ref_mask, fillna_value=pool_mean("ORB%"))),
    }

    for name, raw_pct in raw.items():
        conf = pt_weight * zone_conf[zone_conf_map[name]] if name in zone_conf_map else pt_weight
        shrunk = 50 + (raw_pct - 50) * conf
        # 아주 옛날 시즌엔 일부 선수 행의 MP/G 자체가 누락되어 pt_weight가 NaN이
        # 될 수 있음 → conf가 NaN → astype(int) 직전에 fillna로 중립값(50) 처리.
        merged[f"Attr_{name}"] = shrunk.fillna(50).round(0).clip(1, 99).astype(int)

    stamina_raw = (0.6 * pct_scale(merged["MP"], ref_mask, fillna_value=pool_mean("MP"))
                   + 0.4 * pct_scale(merged["G"], ref_mask, fillna_value=pool_mean("G")))
    merged["Attr_Stamina"] = stamina_raw.fillna(50).round(0).clip(1, 99).astype(int)

    ATTR_COLS = [f"Attr_{k}" for k in raw] + ["Attr_Stamina"]

    ZONE_COLS = []
    for zname in ["RA", "Paint", "Midrange", "Corner3", "Arc3"]:
        col = f"Zone_{zname}"
        merged[col] = zone_shares[zname].round(3)
        ZONE_COLS.append(col)

    merged = merged.replace({np.nan: None})
    merged = merged.sort_values("PTS", ascending=False)

    keep_cols = ["Player", "Team", "TeamsPlayed", "Pos", "PosGroup", "BbrefId",
                 "Age", "G", "GS", "MP",
                 "FG", "FGA", "FG%", "3P", "3PA", "3P%", "eFG%", "FT", "FTA", "FT%",
                 "ORB", "DRB", "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS",
                 "PER", "TS%", "USG%", "WS", "WS/48", "BPM", "OBPM", "DBPM", "VORP",
                 "FGpct_0_3", "FGpct_3_10", "FGpct_10_16", "FGpct_16_3P", "FGpct_3P",
                 "Corner3Pct", "Corner3PctOf3PA",
                 "PctFGA_0_3", "AstPct_2P", "AstPct_3P", "DunkPctFGA",
                 "NDL_Score", "NDL_Weight", "NDL_Production", "NDL_Efficiency",
                 "NDL_Playmaking", "NDL_Defense", "NDL_WinImpact"] + ATTR_COLS + ZONE_COLS
    for c in keep_cols:
        if c not in merged.columns:
            merged[c] = None
    out = merged[keep_cols].to_dict(orient="records")

    return {
        "season": f"{season - 1}-{str(season)[-2:]}",
        "generated": season,
        "has_shooting_data": has_shooting,
        "ndl_weights": WEIGHTS,
        "players": out,
    }


def main(force: bool = False):
    """force=True면 이미 저장된 시즌도 다시 계산한다(가중치·공식 변경 시 사용).

    HTML은 .cache에 남아 있으므로 재계산은 네트워크를 타지 않는다.
    """
    SEASONS_DIR.mkdir(exist_ok=True)
    ok, skipped, failed = 0, 0, 0
    for season in range(SEASON_START, SEASON_END + 1):
        out_path = SEASONS_DIR / f"players_{season}.json"
        if out_path.exists() and not force:
            skipped += 1
            continue
        print(f"=== {season} 시즌({season - 1}-{str(season)[-2:]}) 수집 시작 ===")
        try:
            result = process_season(season)
        except urllib.error.HTTPError as exc:
            print(f"  [{season}] HTTP 오류로 건너뜀: {exc}")
            failed += 1
            continue
        except Exception as exc:
            print(f"  [{season}] 처리 중 오류로 건너뜀: {type(exc).__name__}: {exc}")
            failed += 1
            continue
        out_path.write_text(
            json.dumps(result, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        print(f"  [{season}] 저장 완료: {len(result['players'])}명, "
              f"has_shooting_data={result['has_shooting_data']}")
        ok += 1

    print(f"\n완료: 성공 {ok}, 이미 존재해서 건너뜀 {skipped}, 실패 {failed}")


if __name__ == "__main__":
    import sys
    main(force="--force" in sys.argv)
