"""
This script is used to build the pre-match feature table for the Dixon-Coles model.
 
Dixon-Coles doesn't need rolling/engineered features the way the live model
does. It fits attack/defense strength parameters per team directly from
full-season match results via maximum likelihood. So this script's job is
simply: pull clean match results, sanity-check them, and save a tidy table.
 
"""

import logging
import pandas as pd
from statsbombpy import sb
from config.paths import PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

COMPETITION_ID = 11 # La Liga
SEASON_ID = 27 # 2015/16

OUTPUT_PATH = PROCESSED_DATA_DIR / "prematch_matches.csv"


def fetch_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    log.info(f"Fetching matches for competition_id={competition_id}, season_id={season_id}")
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    log.info(f"Fetched {len(matches)} matches")
    return matches


def build_prematch_table(matches: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["match_id", "match_date", "match_week",
                     "home_team", "away_team", "home_score", "away_score"]
    missing = [c for c in required_cols if c not in matches.columns]
    if missing:
        raise ValueError(f"Expected columns missing from matches data: {missing}")
    
    table = matches[required_cols].copy()
    table["match_date"] = pd.to_datetime(table["match_date"])
    table = table.sort_values("match_date").reset_index(drop=True)

    # Determine the output label, which is useful for a quick win-rate sanity check
    # and acts as a secondary target if we ever want a simple classifier alongside 
    # the Dixon-Coles
    def outcome(row):
        if row["home_score"] > row["away_score"]:
            return "home_win"
        elif row["home_score"] < row["away_score"]:
            return "away_win"
        return "draw"
    
    table["outcome"] = table.apply(outcome, axis=1)
    return table


def sanity_check(table: pd.DataFrame) -> None:
    n_teams = pd.concat([table["home_team"], table["away_team"]]).nunique()
    log.info(f"Teams: {n_teams}, Matches: {len(table)}")

    expected_matches = n_teams * (n_teams - 1) # collects the full round robin (home and away)
    if len(table) != expected_matches:
        log.warning(
                f"Expected {expected_matches} matches for a full round-robin "
                f"with {n_teams} teams, but found {len(table)}. Check for "
                f"missing or duplicate matches."
            )
    else:
        log.info("Match count matches expected full round-robin schedule.")
    
    if table[["home_score", "away_score"]].isna().any().any():
        log.warning("Found missing scores - check match_status before training.")

    
    outcome_counts = table["outcome"].value_counts(normalize=True).round(3)
    log.info(f"Outcome distribution:\n{outcome_counts}")


def main():
    matches = fetch_matches(COMPETITION_ID, SEASON_ID)
    table = build_prematch_table(matches)
    sanity_check(table)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT_PATH, index=False)
    log.info(f"Saved pre-match feature table to {OUTPUT_PATH} ({len(table)} rows)")


if __name__ == "__main__":
    main()