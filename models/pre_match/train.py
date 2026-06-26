"""
This script trains a Dixon-Coles model for pre-match predictions and checks
how well it performs on a holdout set.

It fits one attack value and one defense value for each team, then uses the
Dixon-Coles adjustment to better handle low-scoring games like 0-0, 1-0, 0-1,
and 1-1.

Because the model is only identifiable up to differences in team strength, one
team is fixed as the reference point with attack and defense set to zero. That
is a standard way to make the model work.

Output:
    models/pre_match/dixon_coles_params.json   — fitted team strengths
    models/pre_match/validation_report.json    — log loss, accuracy, calibration
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

INPUT_PATH = PROCESSED_DATA_DIR / "prematch_matches.csv"
MODEL_DIR = Path("models/pre_match")
MAX_GOALS = 10  # How far up the scoreline grid goes when we estimate match probabilities.

# We keep the last few match weeks aside for testing so the model is judged on
# data it did not see during training.
HOLDOUT_MATCH_WEEKS = 6


# -----------------------------
# Load and split the data
# -----------------------------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(INPUT_PATH)
    df["match_date"] = pd.to_datetime(df["match_date"])
    return df.sort_values("match_date").reset_index(drop=True)


def train_holdout_split(df: pd.DataFrame, holdout_weeks: int):
    max_week = df["match_week"].max()
    cutoff = max_week - holdout_weeks
    train = df[df["match_week"] <= cutoff].reset_index(drop=True)
    holdout = df[df["match_week"] > cutoff].reset_index(drop=True)
    log.info(f"Train: {len(train)} matches (through week {cutoff}) | "
              f"Holdout: {len(holdout)} matches (weeks {cutoff + 1}-{max_week})")
    return train, holdout


# -----------------------------
# Dixon-Coles likelihood
# -----------------------------
def tau(x: int, y: int, lam: float, mu: float, rho: float) -> float:
    """Low-score correlation adjustment from Dixon & Coles (1997)."""
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
        self.fit_teams = self.teams[1:]  # All teams except the reference team.
        self.team_to_idx = {t: i for i, t in enumerate(self.fit_teams)}
        self.n_fit_teams = len(self.fit_teams)
        # The parameter layout is: attack values, defense values, home advantage, rho.
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

    def _neg_log_likelihood(self, params, matches: pd.DataFrame) -> float:
        attack, defense, home_adv, rho = self._unpack(params)
        ll = 0.0
        for _, m in matches.iterrows():
            lam = np.exp(attack[m["home_team"]] - defense[m["away_team"]] + home_adv)
            mu = np.exp(attack[m["away_team"]] - defense[m["home_team"]])
            x, y = int(m["home_score"]), int(m["away_score"])

            prob = poisson.pmf(x, lam) * poisson.pmf(y, mu) * tau(x, y, lam, mu, rho)
            prob = max(prob, 1e-10)  # This avoids taking the log of zero in edge cases.
            ll += np.log(prob)
        return -ll

    def fit(self, matches: pd.DataFrame):
        n = self.n_fit_teams
        # Start with a simple guess: neutral attack and defense, a small home advantage,
        # and rho close to zero.
        x0 = np.concatenate([
            np.zeros(n),       # attack
            np.zeros(n),       # defense
            [0.2],             # home_adv
            [0.0],             # rho
        ])

        log.info("Fitting Dixon-Coles via MLE (this can take a minute)...")
        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(matches,),
            method="L-BFGS-B",
            options={"maxiter": 500},
        )

        if not result.success:
            log.warning(f"Optimizer did not fully converge: {result.message}")
        else:
            log.info(f"Converged. Final negative log-likelihood: {result.fun:.2f}")

        self.params = result.x
        return result

    def team_strengths(self):
        attack, defense, home_adv, rho = self._unpack(self.params)
        return attack, defense, home_adv, rho

    def predict_match_probs(self, home_team: str, away_team: str):
        """Return the probabilities for home win, draw, and away win by summing
        over possible scorelines up to MAX_GOALS goals each."""
        attack, defense, home_adv, rho = self._unpack(self.params)
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

        total = home_win + draw + away_win  # This should be close to 1.0, and we normalize just in case.
        return home_win / total, draw / total, away_win / total

    def save(self, path: Path):
        attack, defense, home_adv, rho = self._unpack(self.params)
        payload = {
            "reference_team": self.reference_team,
            "attack": attack,
            "defense": defense,
            "home_advantage": home_adv,
            "rho": rho,
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
    calibration_rows = []

    for _, m in holdout.iterrows():
        p_home, p_draw, p_away = model.predict_match_probs(m["home_team"], m["away_team"])
        probs = {"home_win": p_home, "draw": p_draw, "away_win": p_away}

        actual = m["outcome"]
        p_actual = max(probs[actual], 1e-10)
        log_losses.append(-np.log(p_actual))

        predicted_outcome = max(probs, key=probs.get)
        if predicted_outcome == actual:
            correct += 1

        calibration_rows.append({
            "match_id": m["match_id"],
            "p_home_win": p_home,
            "p_draw": p_draw,
            "p_away_win": p_away,
            "actual_outcome": actual,
        })

    avg_log_loss = float(np.mean(log_losses))
    accuracy = correct / len(holdout)

    calib_df = pd.DataFrame(calibration_rows)
    # Split the predicted home-win probabilities into bins and compare them to
    # the actual home-win rate in each bin.
    calib_df["p_home_bin"] = pd.cut(calib_df["p_home_win"], bins=[0, .2, .4, .6, .8, 1.0])
    calibration_summary = (
        calib_df.groupby("p_home_bin")
        .apply(lambda g: pd.Series({
            "avg_predicted": g["p_home_win"].mean(),
            "actual_home_win_rate": (g["actual_outcome"] == "home_win").mean(),
            "n_matches": len(g),
        }))
        .reset_index()
    )

    log.info(f"Holdout log loss: {avg_log_loss:.4f}")
    log.info(f"Holdout accuracy (most-likely-outcome): {accuracy:.3f}")
    log.info(f"Calibration by predicted home-win probability bin:\n{calibration_summary}")

    return {
        "log_loss": avg_log_loss,
        "accuracy": accuracy,
        "n_holdout_matches": len(holdout),
        "calibration": calibration_summary.to_dict(orient="records"),
    }


def main():
    df = load_data()
    train, holdout = train_holdout_split(df, HOLDOUT_MATCH_WEEKS)

    all_teams = pd.concat([df["home_team"], df["away_team"]]).unique()
    model = DixonColesModel(teams=all_teams)
    model.fit(train)

    attack, defense, home_adv, rho = model.team_strengths()
    log.info(f"Home advantage: {home_adv:.3f} | rho: {rho:.3f}")

    top_attack = sorted(attack.items(), key=lambda kv: kv[1], reverse=True)[:5]
    top_defense = sorted(defense.items(), key=lambda kv: kv[1])[:5]  # Lower defense values mean stronger teams.
    log.info(f"Top 5 attack strength: {top_attack}")
    log.info(f"Top 5 defense strength (lower = stronger): {top_defense}")

    report = evaluate(model, holdout)

    model.save(MODEL_DIR / "dixon_coles_params.json")
    with open(MODEL_DIR / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    log.info(f"Saved validation report to {MODEL_DIR / 'validation_report.json'}")


if __name__ == "__main__":
    main()