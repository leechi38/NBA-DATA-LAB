"""2025-26 NBA 정규시즌 선수 스탯 수집 + NDL Score 계산.

basketball-reference.com의 per_game / advanced 테이블을 가져와
트레이드로 여러 팀을 거친 선수는 시즌 합산(2TM/3TM) 행으로 통합하고,
NDL Score(자체 종합 평가 지표)를 계산해 JSON으로 저장한다.
"""
import io
import json
import re
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

SEASON = 2026  # 2025-26 시즌 (basketball-reference는 종료 연도로 표기)
REF_MIN_G = 15  # z-score 기준(평균/표준편차) 산출에 포함할 최소 출전 경기수 — 평가 대상 제외 기준이 아님
REF_MINUTES = 2000  # 이 총 출전시간(분) 이상이면 출전시간 가중치 1.0(풀타임 신뢰)
REF_ZONE_ATTEMPTS = 3.0  # 구역별 경기당 시도 횟수가 이 값 이상이면 해당 구역 슛 능력치를 100% 신뢰
POS_GROUPS = {"PG": "G", "SG": "G", "SF": "F", "PF": "F", "C": "C"}

PER_GAME_URL = f"https://www.basketball-reference.com/leagues/NBA_{SEASON}_per_game.html"
ADVANCED_URL = f"https://www.basketball-reference.com/leagues/NBA_{SEASON}_advanced.html"
SHOOTING_URL = f"https://www.basketball-reference.com/leagues/NBA_{SEASON}_shooting.html"

SHOOTING_COLS = [
    "Rk", "Player", "Age", "Team", "Pos", "G", "GS", "MP", "FG%",
    "Dist_Avg", "PctFGA_2P", "PctFGA_0_3", "PctFGA_3_10", "PctFGA_10_16",
    "PctFGA_16_3P", "PctFGA_3P",
    "FGpct_2P", "FGpct_0_3", "FGpct_3_10", "FGpct_10_16", "FGpct_16_3P", "FGpct_3P",
    "AstPct_2P", "AstPct_3P", "DunkPctFGA", "DunkCount",
    "Corner3PctOf3PA", "Corner3Pct", "HalfCourt_Att", "HalfCourt_Md", "Awards",
]

OUT_PATH = Path(__file__).parent / "players_2025_26.json"


CACHE_DIR = Path(__file__).parent / ".cache"
ID_PATTERN = re.compile(r'data-append-csv="([^"]+)"[^>]*>\s*<a[^>]*>([^<]+)</a>')
HEADSHOT_VERSION_PATTERN = re.compile(r"/req/(\d+)/images/headshots/")


def fetch_html(url: str, cache_name: str) -> str:
    CACHE_DIR.mkdir(exist_ok=True)
    cache_path = CACHE_DIR / f"{cache_name}.html"
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    html = urllib.request.urlopen(req, timeout=20).read().decode("utf-8")
    cache_path.write_text(html, encoding="utf-8")
    return html


def parse_table(html: str, table_id: str) -> pd.DataFrame:
    df = pd.read_html(io.StringIO(html), attrs={"id": table_id})[0]
    df = df[df["Player"] != "Player"]  # 중간 헤더 반복 행 제거(안전장치)
    df = df[df["Player"] != "League Average"]  # 리그 평균 요약 행 제거
    df = df.dropna(subset=["Player", "Team"])
    return df


def extract_player_ids(html: str) -> dict:
    """선수 이름 -> basketball-reference 선수 ID(예: jokicni01) 매핑."""
    mapping = {}
    for pid, name in ID_PATTERN.findall(html):
        mapping.setdefault(name, pid)
    return mapping


def discover_headshot_version(sample_bbref_id: str) -> str:
    html = fetch_html(
        f"https://www.basketball-reference.com/players/{sample_bbref_id[0]}/{sample_bbref_id}.html",
        "sample_player_page",
    )
    m = HEADSHOT_VERSION_PATTERN.search(html)
    return m.group(1) if m else "202605210"


def parse_shooting_table(html: str) -> pd.DataFrame:
    """슛 거리별 성공률(골밑/미드레인지/3점 실측치) + 덩크 비율 테이블.
    다단 헤더(멀티인덱스)라 열 이름을 위치 기준으로 새로 붙인다."""
    df = pd.read_html(io.StringIO(html), attrs={"id": "shooting"})[0]
    df.columns = SHOOTING_COLS
    df = df[df["Player"] != "Player"]
    df = df[df["Player"] != "League Average"]
    df = df.dropna(subset=["Player", "Team"])
    return df


def dedupe_traded_players(df: pd.DataFrame) -> pd.DataFrame:
    """트레이드된 선수는 2TM/3TM(시즌 합산) 행만 남기고,
    현재 소속팀은 개별 팀 행 중 마지막(가장 최근) 팀으로 표시한다."""
    rows = []
    for player, g in df.groupby("Player", sort=False):
        if len(g) == 1:
            rows.append(g.iloc[0].to_dict())
            continue
        combined = g[g["Team"].str.contains("TM", na=False)]
        individual = g[~g["Team"].str.contains("TM", na=False)]
        base = (combined.iloc[0] if len(combined) else g.iloc[0]).to_dict()
        if len(individual):
            base["Team"] = individual.iloc[-1]["Team"]  # 가장 최근 소속팀
            base["TeamsPlayed"] = "/".join(individual["Team"].tolist())
        else:
            base["TeamsPlayed"] = base["Team"]
        rows.append(base)
    return pd.DataFrame(rows)


def to_num(s):
    return pd.to_numeric(s, errors="coerce")


def zscore_ref(series: pd.Series, ref_mask: pd.Series) -> pd.Series:
    """ref_mask로 표시된 표본 부분집합의 평균/표준편차 기준으로 전체 series를 표준화."""
    mean = series[ref_mask].mean()
    std = series[ref_mask].std(ddof=0)
    if std == 0 or np.isnan(std):
        return series * 0
    return (series - mean) / std


def pos_zscore_ref(df: pd.DataFrame, col: str, ref_mask: pd.Series) -> pd.Series:
    """포지션 그룹(가드/포워드/센터)별로 표준화 — 빅맨의 골밑 효율 우위 등
    포지션에 따른 구조적 차이를 제거하고 '포지션 내 상대적 우수함'만 남긴다."""
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
    """series 값을 ref_mask 표본 내 백분위(1~99)로 변환. invert면 낮을수록 고득점."""
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


def main():
    print("per_game 테이블 수집 중...")
    pg_html = fetch_html(PER_GAME_URL, "per_game_stats_html")
    pg = parse_table(pg_html, "per_game_stats")
    id_map = extract_player_ids(pg_html)
    time.sleep(2)  # basketball-reference 서버 부담 최소화
    print("advanced 테이블 수집 중...")
    adv_html = fetch_html(ADVANCED_URL, "advanced_html")
    adv = parse_table(adv_html, "advanced")
    time.sleep(2)
    print("shooting(슛 거리별) 테이블 수집 중...")
    shooting_html = fetch_html(SHOOTING_URL, "shooting_html")
    shooting = parse_shooting_table(shooting_html)

    pg = dedupe_traded_players(pg)
    adv = dedupe_traded_players(adv)
    shooting = dedupe_traded_players(shooting)

    num_cols_pg = ["Age", "G", "GS", "MP", "FG", "FGA", "FG%", "3P", "3PA", "3P%",
                   "2P", "2PA", "2P%", "eFG%", "FT", "FTA", "FT%", "ORB", "DRB",
                   "TRB", "AST", "STL", "BLK", "TOV", "PF", "PTS"]
    for c in num_cols_pg:
        pg[c] = to_num(pg[c])

    num_cols_adv = ["PER", "TS%", "3PAr", "FTr", "ORB%", "DRB%", "TRB%", "AST%",
                     "STL%", "BLK%", "TOV%", "USG%", "OWS", "DWS", "WS", "WS/48",
                     "OBPM", "DBPM", "BPM", "VORP"]
    for c in num_cols_adv:
        adv[c] = to_num(adv[c])

    num_cols_shooting = ["FGpct_0_3", "FGpct_3_10", "FGpct_10_16", "FGpct_16_3P",
                          "FGpct_3P", "PctFGA_0_3", "PctFGA_3_10", "PctFGA_10_16",
                          "PctFGA_16_3P", "PctFGA_3P", "Corner3PctOf3PA", "Corner3Pct",
                          "AstPct_2P", "AstPct_3P", "DunkPctFGA"]
    for c in num_cols_shooting:
        shooting[c] = to_num(shooting[c])

    merged = pg.merge(
        adv[["Player", "Team"] + num_cols_adv],
        on=["Player", "Team"], how="left", suffixes=("", "_adv"),
    ).merge(
        shooting[["Player", "Team"] + num_cols_shooting],
        on=["Player", "Team"], how="left", suffixes=("", "_shoot"),
    )

    merged["PosGroup"] = merged["Pos"].map(primary_pos_group)
    merged["BbrefId"] = merged["Player"].map(id_map)
    merged["TotalMP"] = merged["MP"] * merged["G"]
    # z-score의 평균/표준편차는 표본이 안정적인 선수들(15경기 이상)만으로 산출.
    # 단, 평가·랭킹 대상에서 아무도 배제하지 않음 — 출전시간이 적은 선수는
    # 아래 pt_weight로 점수를 평균(50)에 가깝게 선형으로 끌어당길 뿐이다.
    ref_mask = merged["G"] >= REF_MIN_G

    # --- 하위 지표 ---
    # 생산성 · 승리기여: 전체 리그 기준(포지션과 무관하게 비교 가능한 총량/가치 지표)
    production = (
        zscore_ref(merged["PTS"], ref_mask) + zscore_ref(merged["TRB"], ref_mask)
        + zscore_ref(merged["AST"], ref_mask) + zscore_ref(merged["STL"], ref_mask)
        + zscore_ref(merged["BLK"], ref_mask) - zscore_ref(merged["TOV"], ref_mask)
    )
    win_impact = zscore_ref(merged["WS"], ref_mask) + zscore_ref(merged["VORP"], ref_mask)

    # 효율성 · 플레이메이킹 · 수비기여: 포지션 그룹(가드/포워드/센터) 내 상대평가.
    # 빅맨은 골밑 슛 비중이 높아 TS%/eFG%가, 가드는 볼 소유 특성상 AST%가
    # 구조적으로 유·불리해지므로, "포지션 평균 대비 얼마나 잘하는가"로 표준화한다.
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

    weights = {"production": 0.25, "efficiency": 0.20, "playmaking": 0.15,
               "defense": 0.20, "win_impact": 0.20}

    raw_score = (
        weights["production"] * zscore_ref(production, ref_mask)
        + weights["efficiency"] * zscore_ref(efficiency, ref_mask)
        + weights["playmaking"] * zscore_ref(playmaking, ref_mask)
        + weights["defense"] * zscore_ref(defense, ref_mask)
        + weights["win_impact"] * zscore_ref(win_impact, ref_mask)
    )

    # --- 출전시간 신뢰도 가중치(선형) ---
    # 하드 컷오프 대신, 총 출전시간이 REF_MINUTES(풀타임 기준)에 못 미치는 만큼
    # 점수를 평균(0, 즉 NDL 50점)쪽으로 선형 축소(shrinkage)한다.
    pt_weight = (merged["TotalMP"] / REF_MINUTES).clip(lower=0, upper=1)
    adjusted_score = raw_score * pt_weight

    sigma_raw = raw_score[ref_mask].std(ddof=0)
    ndl = 50 + 10 * (adjusted_score / sigma_raw)
    ndl = ndl.clip(lower=0, upper=100)

    merged["NDL_Production"] = zscore_ref(production, ref_mask).round(2)
    merged["NDL_Efficiency"] = zscore_ref(efficiency, ref_mask).round(2)
    merged["NDL_Playmaking"] = zscore_ref(playmaking, ref_mask).round(2)
    merged["NDL_Defense"] = zscore_ref(defense, ref_mask).round(2)
    merged["NDL_WinImpact"] = zscore_ref(win_impact, ref_mask).round(2)
    merged["NDL_Weight"] = pt_weight.round(2)
    merged["NDL_Score"] = ndl.round(1)

    # =====================================================================
    # 게임 능력치(1~99, FM 스타일) — 시뮬레이션 게임에서 쓸 선수 속성.
    # 모두 실제 2025-26 시즌 스탯의 리그 내 백분위로 산출한다.
    # 표본이 적은 선수는 NDL_Score와 동일한 pt_weight로 평균(50)에 선형 수렴시킨다.
    # 단, 스태미나는 '관측된 출전시간' 자체가 값이므로 수렴시키지 않는다(아래 참고).
    # =====================================================================
    def pool_mean(col):
        return merged.loc[ref_mask, col].mean()

    def rate_and_volume(rate_col, volume_col, invert=False):
        """%(비중) 지표와 raw count(1차 스탯) 백분위를 절반씩 섞는다.
        AST%가 높아도 실제 어시스트 개수가 적으면(짧은 출전시간 등) 낮게 나오도록."""
        rate_pct = pct_scale(merged[rate_col], ref_mask, invert=invert, fillna_value=pool_mean(rate_col))
        vol_pct = pct_scale(merged[volume_col], ref_mask, invert=invert, fillna_value=pool_mean(volume_col))
        return (rate_pct + vol_pct) / 2

    teamwork_ratio = merged["AST%"] / merged["USG%"].replace(0, np.nan)
    starter_ratio = merged["GS"] / merged["G"].replace(0, np.nan)

    # --- 구역별(zone) 슛 능력치: 골밑(RA) / 페인트(3-16ft) / 미드레인지(16ft-3점) /
    # 코너3 / 아크3(정면·윙 3점) — 전부 실제 거리별 성공률 기반. 시도 횟수가 적은
    # 구역은 zone_conf로 신뢰도를 낮춰 평균(50) 쪽으로 당긴다(성공률만 보지 않음).
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
        # 기술 — 구역별 슛 능력치(신뢰도 = 구역 시도량 × 전체 표본 신뢰도)
        "Finishing": zone_skill_raw["Finishing"],
        "Paint": zone_skill_raw["Paint"],
        "Midrange": zone_skill_raw["Midrange"],
        "Corner3": zone_skill_raw["Corner3"],
        "Arc3": zone_skill_raw["Arc3"],
        "FreeThrow": pct_scale(merged["FT%"], ref_mask, fillna_value=pool_mean("FT%")),
        "Playmaking": rate_and_volume("AST%", "AST"),
        "BallHandling": rate_and_volume("TOV%", "TOV", invert=True),
        "ShotCreation": pct_scale(merged["AstPct_2P"], ref_mask, invert=True, fillna_value=pool_mean("AstPct_2P")),
        # 수비 — %비중 + 실제 1차 스탯(raw count) 절반씩 반영
        "OnBallDefense": rate_and_volume("STL%", "STL"),
        "RimProtection": rate_and_volume("BLK%", "BLK"),
        "DefRebound": rate_and_volume("DRB%", "DRB"),
        "OffRebound": rate_and_volume("ORB%", "ORB"),
        "DefensiveIQ": pct_scale(merged["DBPM"], ref_mask, fillna_value=pool_mean("DBPM")),
        # 정신
        "Teamwork": pct_scale(teamwork_ratio, ref_mask, fillna_value=teamwork_ratio[ref_mask].mean()),
        "Composure": (pct_scale(merged["FT%"], ref_mask, fillna_value=pool_mean("FT%"))
                      + pct_scale(merged["TOV%"], ref_mask, invert=True, fillna_value=pool_mean("TOV%"))) / 2,
        "WinningMentality": pct_scale(merged["BPM"], ref_mask, fillna_value=pool_mean("BPM")),
        "Experience": (pct_scale(merged["Age"], ref_mask, fillna_value=pool_mean("Age"))
                       + pct_scale(starter_ratio, ref_mask, fillna_value=0)) / 2,
        # 피지컬 (실측 신체능력 데이터가 없어 박스스탯 기반 대체 지표임을 명시)
        "Athleticism": (0.5 * pct_scale(merged["DunkPctFGA"], ref_mask, fillna_value=0)
                        + 0.3 * pct_scale(merged["BLK%"], ref_mask, fillna_value=pool_mean("BLK%"))
                        + 0.2 * pct_scale(merged["ORB%"], ref_mask, fillna_value=pool_mean("ORB%"))),
    }

    for name, raw_pct in raw.items():
        conf = pt_weight * zone_conf[zone_conf_map[name]] if name in zone_conf_map else pt_weight
        shrunk = 50 + (raw_pct - 50) * conf
        merged[f"Attr_{name}"] = shrunk.round(0).clip(1, 99).astype(int)

    # 스태미나/내구성: MP·G는 추정치가 아니라 실측 관측값이므로 신뢰도 가중치를 적용하지 않는다.
    stamina_raw = (0.6 * pct_scale(merged["MP"], ref_mask, fillna_value=pool_mean("MP"))
                   + 0.4 * pct_scale(merged["G"], ref_mask, fillna_value=pool_mean("G")))
    merged["Attr_Stamina"] = stamina_raw.round(0).clip(1, 99).astype(int)

    ATTR_COLS = [f"Attr_{k}" for k in raw] + ["Attr_Stamina"]

    # 구역 포지셔닝 확률(0~1, 합=1) — 시뮬레이션 엔진이 "이 선수가 지금 코트 어디쯤
    # 있을 확률이 높은가"를 정할 때 쓸 실측 기반 데이터. 능력치가 아니라 위치 성향.
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
    out = merged[keep_cols].to_dict(orient="records")

    headshot_version = discover_headshot_version("jokicni01")

    OUT_PATH.write_text(
        json.dumps({"season": "2025-26", "generated": SEASON,
                    "headshot_version": headshot_version, "players": out},
                    ensure_ascii=False, indent=None),
        encoding="utf-8",
    )
    print(f"저장 완료: {OUT_PATH} ({len(out)}명, headshot_version={headshot_version})")

    top10 = merged.sort_values("NDL_Score", ascending=False).head(10)
    print("\n[NDL Score TOP 10]")
    print(top10[["Player", "Team", "MP", "G", "NDL_Weight", "NDL_Score"]].to_string(index=False))


if __name__ == "__main__":
    main()
