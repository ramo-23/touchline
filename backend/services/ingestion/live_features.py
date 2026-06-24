"""
In-play (live) feature table builder for win-probability modeling.

Vectorized version with:
- home/away separation
- corrected score logic (now includes own goals)
- stable rolling + cumulative features
- minimal high-value interaction features
- calibration-level refinements: neutral xg_share at kickoff, activity flag,
  time-normalized score pressure, smoothed goal momentum
- per-match validation against the official scoreline, so silent
  score-tracking bugs (like the own-goal gap) get caught, not assumed away
"""

import logging
import pandas as pd
from statsbombpy import sb
from config.paths import PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

COMPETITION_ID = 11  # La Liga
SEASON_ID = 27       # 2015/16

MATCH_LENGTH_MINUTES = 90
ROLLING_WINDOW_MINUTES = 10
GOAL_MOMENTUM_EWM_SPAN = 5

OUTPUT_PATH = PROCESSED_DATA_DIR / "live_snapshots.csv"


# -----------------------------
# Fetch
# -----------------------------
def fetch_matches(competition_id: int, season_id: int) -> pd.DataFrame:
    log.info("Fetching matches")
    return sb.matches(competition_id=competition_id, season_id=season_id)


# -----------------------------
# Encoding fix (defensive — root cause was the CSV write encoding, see
# main()'s utf-8-sig write below; this stays as a no-op safety net in case
# mojibake ever enters from a different source)
# -----------------------------
def _repair_mojibake(name):
    """Undo a UTF-8-decoded-as-Latin-1 mistake, but only if the round-trip
    succeeds cleanly. If the string is already correct, encoding it to
    Latin-1 and back to UTF-8 will raise on accented characters
    (e.g. 'Gijon') and we fall back to the original instead of silently
    dropping the unrecognized bytes via errors='ignore'."""
    if not isinstance(name, str):
        return name
    try:
        return name.encode("latin1").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return name


def fix_encoding(df: pd.DataFrame) -> pd.DataFrame:
    for col in ["home_team", "away_team"]:
        if col in df.columns:
            df[col] = df[col].astype(str).apply(_repair_mojibake)
    return df


# -----------------------------
# Event preprocessing
# -----------------------------
def preprocess_events(events: pd.DataFrame) -> pd.DataFrame:
    events = events.sort_values("minute").copy()

    shot_goal = (
        (events["type"] == "Shot") &
        (events.get("shot_outcome") == "Goal").fillna(False)
    )

    # Own goals are a separate event type ("Own Goal For" on the team that
    # benefits) and never have type == "Shot" — without this, any match
    # with an own goal silently under-counts from that point on.
    own_goal_for = (events["type"] == "Own Goal For")

    events["is_goal"] = (shot_goal | own_goal_for).astype(int)

    events["xg"] = events.get("shot_statsbomb_xg", 0).fillna(0)

    if "foul_committed_card" in events.columns:
        events["is_red"] = events["foul_committed_card"].eq("Red Card").astype(int)
    else:
        events["is_red"] = 0

    return events


# -----------------------------
# Minute grid
# -----------------------------
def minute_index(events: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"minute": range(int(events["minute"].max()) + 1)})


# -----------------------------
# Team features
# -----------------------------
def build_team_features(events: pd.DataFrame, home: str, away: str) -> pd.DataFrame:
    events = events.copy()

    home_events = events[events["team"] == home]
    away_events = events[events["team"] == away]

    home_agg = home_events.groupby("minute").agg(
        home_goals=("is_goal", "sum"),
        home_xg=("xg", "sum"),
        home_red=("is_red", "sum"),
    )

    away_agg = away_events.groupby("minute").agg(
        away_goals=("is_goal", "sum"),
        away_xg=("xg", "sum"),
        away_red=("is_red", "sum"),
    )

    return home_agg.join(away_agg, how="outer").fillna(0).reset_index()


# -----------------------------
# Features
# -----------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("minute").copy()

    # cumulative
    df["home_goals_cum"] = df["home_goals"].cumsum()
    df["away_goals_cum"] = df["away_goals"].cumsum()

    df["home_xg_cum"] = df["home_xg"].cumsum()
    df["away_xg_cum"] = df["away_xg"].cumsum()

    df["home_red_cum"] = df["home_red"].cumsum()
    df["away_red_cum"] = df["away_red"].cumsum()

    # rolling
    w = ROLLING_WINDOW_MINUTES

    df["home_xg_roll"] = df["home_xg"].rolling(w + 1, min_periods=1).sum()
    df["away_xg_roll"] = df["away_xg"].rolling(w + 1, min_periods=1).sum()

    df["home_goal_roll"] = df["home_goals"].rolling(w + 1, min_periods=1).sum()
    df["away_goal_roll"] = df["away_goals"].rolling(w + 1, min_periods=1).sum()

    # -----------------------------
    # DERIVED SIGNALS
    # -----------------------------

    df["score_diff"] = df["home_goals_cum"] - df["away_goals_cum"]
    df["xg_diff_roll"] = df["home_xg_roll"] - df["away_xg_roll"]
    df["goal_diff_roll"] = df["home_goal_roll"] - df["away_goal_roll"]
    df["manpower_diff"] = df["home_red_cum"] - df["away_red_cum"]

    # control signal — neutral 0.5 when no xG has been generated yet,
    # instead of defaulting to 0 (which would misleadingly read as "away
    # dominance")
    denom = df["home_xg_roll"] + df["away_xg_roll"]
    df["xg_share"] = (df["home_xg_roll"] / (denom + 1e-9)).where(denom > 0, 0.5)

    # activity signal — flags minutes with no shot-quality information yet,
    # so the model doesn't conflate "no data" with "balanced game"
    df["activity"] = df["home_xg_roll"] + df["away_xg_roll"]
    df["is_active_game"] = (df["activity"] > 0).astype(int)

    # interaction signal (important for late-game behavior)
    df["score_manpower_interaction"] = df["score_diff"] * df["manpower_diff"]

    df["minutes_remaining"] = MATCH_LENGTH_MINUTES - df["minute"]
    df["minutes_remaining"] = df["minutes_remaining"].clip(lower=0)

    # time-normalized score signal — a 1-goal lead means more with less
    # time left
    df["score_pressure"] = df["score_diff"] / (df["minutes_remaining"] + 1)

    # smoothed goal momentum — reduces sparse, jumpy 0/1 swings in
    # goal_diff_roll
    df["goal_momentum"] = (
        df["home_goal_roll"] - df["away_goal_roll"]
    ).ewm(span=GOAL_MOMENTUM_EWM_SPAN, adjust=False).mean()

    return df


# -----------------------------
# Outcome
# -----------------------------
def label_outcome(home_score, away_score):
    if home_score > away_score:
        return "home_win"
    elif home_score < away_score:
        return "away_win"
    return "draw"


# -----------------------------
# Validation
# -----------------------------
def validate_final_score(full: pd.DataFrame, match_row: pd.Series):
    """
    Compares the derived running score at the final snapshot row against
    the official scoreline. Returns a dict describing the mismatch if one
    exists, or None if the match checks out. This is what would have
    caught the own-goal gap automatically, rather than relying on someone
    noticing a suspicious row by eye.
    """
    derived_home = full["home_goals_cum"].iloc[-1]
    derived_away = full["away_goals_cum"].iloc[-1]
    official_home = match_row["home_score"]
    official_away = match_row["away_score"]

    if derived_home != official_home or derived_away != official_away:
        mismatch = {
            "match_id": match_row["match_id"],
            "home_team": match_row["home_team"],
            "away_team": match_row["away_team"],
            "derived_score": f"{derived_home}-{derived_away}",
            "official_score": f"{official_home}-{official_away}",
        }
        log.warning(
            f"match {mismatch['match_id']} ({mismatch['home_team']} vs "
            f"{mismatch['away_team']}): derived {mismatch['derived_score']} "
            f"!= official {mismatch['official_score']}"
        )
        return mismatch
    return None


# -----------------------------
# Pipeline
# -----------------------------
def main():
    matches = fetch_matches(COMPETITION_ID, SEASON_ID)
    matches = fix_encoding(matches)

    all_data = []
    failed = []
    score_mismatches = []

    for i, m in matches.iterrows():
        try:
            events = sb.events(match_id=m["match_id"])
            if events.empty:
                continue

            events = preprocess_events(events)

            team_df = build_team_features(events, m["home_team"], m["away_team"])
            full = minute_index(events).merge(team_df, on="minute", how="left").fillna(0)

            full = add_features(full)

            mismatch = validate_final_score(full, m)
            if mismatch:
                score_mismatches.append(mismatch)

            full["match_id"] = m["match_id"]
            full["home_team"] = m["home_team"]
            full["away_team"] = m["away_team"]
            full["final_outcome"] = label_outcome(m["home_score"], m["away_score"])

            all_data.append(full)

        except Exception as e:
            log.warning(f"match {m['match_id']} failed: {e}")
            failed.append(m["match_id"])

        if (i + 1) % 25 == 0:
            log.info(f"{i+1}/{len(matches)} processed")

    df = pd.concat(all_data, ignore_index=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig adds a BOM so Excel and most Windows tools read accented
    # team names (e.g. "Gijón") correctly instead of as mojibake
    df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    log.info(
        f"Saved {len(df)} rows | "
        f"matches={df['match_id'].nunique()} | "
        f"failed={len(failed)} | "
        f"score_mismatches={len(score_mismatches)}"
    )

    if score_mismatches:
        log.warning(
            f"{len(score_mismatches)} match(es) had score mismatches — "
            f"likely own goals or another uncounted event type. "
            f"Review before training:"
        )
        for mm in score_mismatches:
            log.warning(f"  {mm}")


if __name__ == "__main__":
    main()