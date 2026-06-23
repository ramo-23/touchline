"""
Basic coverage check: this script will query all the 80 existing comps
that are ran by statsbomb to check how many available matches are covered
for each competition. This will be used to train the ML model rather than just do
guesswork.
"""

from statsbombpy import sb
import pandas as pd

def main():
    competitions = sb.competitions()

    rows = [] # declare an empty array to store competition data
    print(f"Checking {len(competitions)} competition/season entries...\n")

    for _, comp in competitions.iterrows():
        comp_id = comp["competition_id"]
        season_id = comp["season_id"]
        comp_name = comp["competition_name"]
        season_name = comp["season_name"]

        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
            match_count = len(matches)
        except Exception:
            match_count = 0 # this is done because some entries might not have accessible data

        rows.append({
            "competition": comp_name,
            "season": season_name,
            "competition_id": comp_id,
            "season_id": season_id,
            "match_count": match_count,
        })

        coverage = pd.DataFrame(rows).sort_values("match_count", ascending=False)

        # Here I will aggregate by competitions across all of the seasons as
        # it is useful when I want to combine multiple seasons of the same league
        # for more training data
        by_competition = (
            coverage.groupby("competition")["match_count"]
            .sum()
            .sort_values(ascending=False)
        )

        print("=== Top single competition/season entries by match count ===")
        print(coverage.head(15).to_string(index=False))

        print("\n=== Total matches per competition (summed across seasons) ===")
        print(by_competition.head(15).to_string())

        coverage.to_csv("data/raw/statsbomb/competition_coverage.csv", index=False)
        print("\nSaved full coverage to data/raw/statscomb/competition_coverage.csv")

if __name__ == "__main__":
    main()