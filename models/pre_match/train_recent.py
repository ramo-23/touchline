"""
This script trains a Dixon-Coles model for pre-match predictions, but with a
slightly more realistic setup for future seasons.

It is different from the older demo in two main ways:

1. Recent matches matter more. Older results get down-weighted, which makes
   the model respond more to current form instead of treating every season as
   equally important.

2. It uses a full season as a holdout. Instead of testing on the last part of
   the same season, it trains on earlier seasons and checks how well it would
   have predicted the most recent completed season.

This writes out:
    models/pre_match/validation_report_recent.json
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from config.paths import PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

INPUT_PATH = PROCESSED_DATA_DIR / "prematch_matches_recent.csv"
MODEL_DIR = Path("models/pre_match")
MAX_GOALS = 10

HOLDOUT_SEASON = "2025/2026"  # The most recent completed season, kept aside for testing.

# This is the decay rate for the time weights. I started with a fairly common
# value for Dixon-Coles style models, and it can be changed later if the
# holdout results look off.
XI = 0.0018


# -----------------------------
# Load and split the data
# -----------------------------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)
    df["match_date"] = pd.to_datetime(df["match_date"])
    return df.sort_values("match_date").reset_index(drop=True)


def train_holdout_split(df: pd.DataFrame, holdout_season: str):
    train = df[df["season"] != holdout_season].reset_index(drop=True)
    holdout = df[df["season"] == holdout_season].reset_index(drop=True)
    log.info(f"Train: {len(train)} matches (seasons: {sorted(train['season'].unique())}) | "
              f"Holdout: {len(holdout)} matches (season: {holdout_season})")
    return train, holdout


def compute_time_weights(train: pd.DataFrame, xi: float) -> np.ndarray:
    """Give newer matches more influence than older ones.

    The most recent match in the training set gets a weight of 1.0, and older
    matches get smaller weights as time goes on.
    """
    most_recent = train["match_date"].max()
    days_old = (most_recent - train["match_date"]).dt.days
    return np.exp(-xi * days_old).values


# -----------------------------
# Weighted Dixon-Coles likelihood
# -----------------------------
def tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    if x == 0 and y == 0:
        return 1 - lam * mu * rho
    elif x == 0 and y == 1:
        return 1 + lam * rho
    elif x == 1 and y == 0:
        return 1 + mu * rho
    elif x == 1 and y == 1:
        return 1 - rho
    return 1.0


class DixonColesModel:
    def __init__(self, teams):
        self.teams = sorted(teams)
        self.reference_team = self.teams[0]
        self.fit_teams = self.teams[1:]
        self.n_fit_teams = len(self.fit_teams)
        self.params = None

    def _unpack(self, params):
        n = self.n_fit_teams
        attack = dict(zip(self.fit_teams, params[:n]))
        defense = dict(zip(self.fit_teams, params[n:2 * n]))
        attack[self.reference_team] = 0.0
        defense[self.reference_team] = 0.0
        home_adv = params[2 * n]
        rho = params[2 * n + 1]
        return attack, defense, home_adv, rho

    def _neg_log_likelihood(self, params, matches: pd.DataFrame, weights: np.ndarray) -> float:
        attack, defense, home_adv, rho = self._unpack(params)
        ll = 0.0
        for (_, m), w in zip(matches.iterrows(), weights):
            lam = np.exp(attack[m["home_team"]] - defense[m["away_team"]] + home_adv)
            mu = np.exp(attack[m["away_team"]] - defense[m["home_team"]])
            x, y = int(m["home_score"]), int(m["away_score"])

            prob = poisson.pmf(x, lam) * poisson.pmf(y, mu) * tau(x, y, lam, mu, rho)
            prob = max(prob, 1e-10)
            ll += w * np.log(prob)
        return -ll

    def fit(self, matches: pd.DataFrame, weights: np.ndarray):
        n = self.n_fit_teams
        x0 = np.concatenate([np.zeros(n), np.zeros(n), [0.2], [0.0]])

        log.info("Fitting time-weighted Dixon-Coles via MLE (this can take a few minutes "
                 "across multiple seasons)...")
        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(matches, weights),
            method="L-BFGS-B",
            options={"maxiter": 500},
        )

        if not result.success:
            log.warning(f"Optimizer did not fully converge: {result.message}")
        else:
            log.info(f"Converged. Final weighted negative log-likelihood: {result.fun:.2f}")

        self.params = result.x
        return result

    def team_strengths(self):
        return self._unpack(self.params)

    def predict_match_probs(self, home_team: str, away_team: str):
        attack, defense, home_adv, rho = self._unpack(self.params)

        # If a team was not part of the training seasons, it will not have a
        # fitted strength value. It is better to raise an error here than to
        # pretend it is an average team and give a misleading prediction.
        for team in (home_team, away_team):
            if team not in attack and team != self.reference_team:
                raise KeyError(
                    f"'{team}' has no fitted strength parameters — likely "
                    f"promoted/relegated and absent from training seasons."
                )

        lam = np.exp(attack[home_team] - defense[away_team] + home_adv)
        mu = np.exp(attack[away_team] - defense[home_team])

        home_win = draw = away_win = 0.0
        for x in range(MAX_GOALS + 1):
            for y in range(MAX_GOALS + 1):
                p = poisson.pmf(x, lam) * poisson.pmf(y, mu) * tau(x, y, lam, mu, rho)
                if x > y:
                    home_win += p
                elif x == y:
                    draw += p
                else:
                    away_win += p

        total = home_win + draw + away_win
        return home_win / total, draw / total, away_win / total

    def save(self, path: Path):
        attack, defense, home_adv, rho = self._unpack(self.params)
        payload = {
            "reference_team": self.reference_team,
            "attack": attack,
            "defense": defense,
            "home_advantage": home_adv,
            "rho": rho,
            "time_decay_xi": XI,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        log.info(f"Saved model parameters to {path}")


# -----------------------------
# Evaluate the model
# -----------------------------
def evaluate(model: DixonColesModel, holdout: pd.DataFrame) -> dict:
    log_losses = []
    correct = 0
    skipped = []
    calibration_rows = []

    for _, m in holdout.iterrows():
        try:
            p_home, p_draw, p_away = model.predict_match_probs(m["home_team"], m["away_team"])
        except KeyError as e:
            skipped.append(str(e))
            continue

        probs = {"home_win": p_home, "draw": p_draw, "away_win": p_away}
        actual = m["outcome"]
        p_actual = max(probs[actual], 1e-10)
        log_losses.append(-np.log(p_actual))

        predicted_outcome = max(probs, key=probs.get)
        if predicted_outcome == actual:
            correct += 1

        calibration_rows.append({
            "p_home_win": p_home, "p_draw": p_draw, "p_away_win": p_away,
            "actual_outcome": actual,
        })

    n_evaluated = len(calibration_rows)
    avg_log_loss = float(np.mean(log_losses)) if log_losses else None
    accuracy = correct / n_evaluated if n_evaluated else None

    calib_df = pd.DataFrame(calibration_rows)
    calib_df["p_home_bin"] = pd.cut(calib_df["p_home_win"], bins=[0, .2, .4, .6, .8, 1.0])
    calibration_summary = (
        calib_df.groupby("p_home_bin", observed=True)
        .apply(lambda g: pd.Series({
            "avg_predicted": g["p_home_win"].mean(),
            "actual_home_win_rate": (g["actual_outcome"] == "home_win").mean(),
            "n_matches": len(g),
        }), include_groups=False)
        .reset_index()
    )

    log.info(f"Holdout ({HOLDOUT_SEASON}) log loss: {avg_log_loss:.4f} "
             f"({n_evaluated}/{len(holdout)} matches evaluated)")
    log.info(f"Holdout accuracy: {accuracy:.3f}")
    if skipped:
        log.warning(f"{len(skipped)} holdout matches skipped (promoted/relegated teams "
                    f"with no training history): {set(skipped)}")
    log.info(f"Calibration:\n{calibration_summary}")

    return {
        "holdout_season": HOLDOUT_SEASON,
        "log_loss": avg_log_loss,
        "accuracy": accuracy,
        "n_evaluated": n_evaluated,
        "n_skipped": len(skipped),
        "calibration": calibration_summary.to_dict(orient="records"),
    }


def main():
    df = load_data()
    train, holdout = train_holdout_split(df, HOLDOUT_SEASON)
    weights = compute_time_weights(train, XI)

    all_teams = pd.concat([df["home_team"], df["away_team"]]).unique()
    model = DixonColesModel(teams=all_teams)
    model.fit(train, weights)

    attack, defense, home_adv, rho = model.team_strengths()
    log.info(f"Home advantage: {home_adv:.3f} | rho: {rho:.3f}")

    top_attack = sorted(attack.items(), key=lambda kv: kv[1], reverse=True)[:5]
    log.info(f"Top 5 attack strength (recency-weighted): {top_attack}")

    report = evaluate(model, holdout)

    model.save(MODEL_DIR / "dixon_coles_recent_params.json")
    with open(MODEL_DIR / "validation_report_recent.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"Saved validation report to {MODEL_DIR / 'validation_report_recent.json'}")


if __name__ == "__main__":
    main()