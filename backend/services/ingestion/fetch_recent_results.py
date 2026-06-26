"""
This script pulls recent La Liga results from football-data.co.uk so the
pre-match model can learn from more relevant seasons.

It uses the last few seasons instead of the older StatsBomb demo data, which
is more useful for predicting the next season and does not need event-level
information.

Source: football-data.co.uk, free and simple to use. The file includes the
main columns we need: date, home team, away team, full-time goals, and the
final result.

Output: data/processed/prematch_matches_recent.csv
"""

import logging
import pandas as pd

from config.paths import PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

LEAGUE_CODE = "SP1"  # La Liga Primera Division
SEASON_CODES = ["2122", "2223", "2324", "2425", "2526"]  # From 2021/22 up to 2025/26

OUTPUT_PATH = PROCESSED_DATA_DIR / "prematch_matches_recent.csv"

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season_code}/{league_code}.csv"


def season_code_to_label(code: str) -> str:
    """'2425' -> '2024/2025'"""
    start = f"20{code[:2]}"
    end = f"20{code[2:]}"
    return f"{start}/{end}"


def fetch_season(season_code: str) -> pd.DataFrame:
    url = BASE_URL.format(season_code=season_code, league_code=LEAGUE_CODE)
    log.info(f"Fetching {season_code_to_label(season_code)} from {url}")

    df = pd.read_csv(url, encoding="latin1")  # The source files use a Windows-1252 style encoding.
    df = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]].dropna(
        subset=["HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    )
    df["match_date"] = pd.to_datetime(df["Date"], dayfirst=True)
    df["season"] = season_code_to_label(season_code)

    df = df.rename(columns={
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_score",
        "FTAG": "away_score",
    })
    return df[["match_date", "season", "home_team", "away_team", "home_score", "away_score"]]


def validate_season(df: pd.DataFrame, season_label: str) -> None:
    # A full round-robin season should have one match for each pair of teams.
    n_teams = pd.concat([df["home_team"], df["away_team"]]).nunique()
    expected = n_teams * (n_teams - 1)
    if len(df) != expected:
        log.warning(
            f"{season_label}: {len(df)} matches with {n_teams} teams "
            f"(expected {expected} for a full round-robin) — season may be "
            f"in progress or have postponed fixtures not yet played."
        )
    else:
        log.info(f"{season_label}: {len(df)} matches, {n_teams} teams — complete round-robin.")


def label_outcome(row) -> str:
    if row["home_score"] > row["away_score"]:
        return "home_win"
    elif row["home_score"] < row["away_score"]:
        return "away_win"
    return "draw"


def main():
    all_seasons = []
    for code in SEASON_CODES:
        season_df = fetch_season(code)
        validate_season(season_df, season_code_to_label(code))
        all_seasons.append(season_df)

    combined = pd.concat(all_seasons, ignore_index=True)
    combined["outcome"] = combined.apply(label_outcome, axis=1)
    combined = combined.sort_values("match_date").reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    log.info(f"Saved {len(combined)} matches across {len(SEASON_CODES)} seasons "
              f"to {OUTPUT_PATH}")
    log.info(f"Season breakdown:\n{combined['season'].value_counts()}")


if __name__ == "__main__":
    main()