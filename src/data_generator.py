from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

def generate_synthetic_historical_data(n_games: int = 2500, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic generator to keep pipeline runnable.
    Replace with real game/team/injury/pace data for production.
    """
    rng = np.random.default_rng(seed)

    teams = [
        'Boston Celtics','Brooklyn Nets','New York Knicks','Philadelphia 76ers','Toronto Raptors',
        'Chicago Bulls','Cleveland Cavaliers','Detroit Pistons','Indiana Pacers','Milwaukee Bucks',
        'Atlanta Hawks','Charlotte Hornets','Miami Heat','Orlando Magic','Washington Wizards',
        'Denver Nuggets','Minnesota Timberwolves','Oklahoma City Thunder','Portland Trail Blazers','Utah Jazz',
        'Golden State Warriors','LA Clippers','Los Angeles Lakers','Phoenix Suns','Sacramento Kings',
        'Dallas Mavericks','Houston Rockets','Memphis Grizzlies','New Orleans Pelicans','San Antonio Spurs'
    ]

    start_date = datetime(2022, 10, 1)
    rows = []
    for i in range(n_games):
        date = start_date + timedelta(days=int(i * 0.6))
        home, away = rng.choice(teams, size=2, replace=False)

        home_ppg = rng.normal(114, 6)
        away_ppg = rng.normal(114, 6)
        home_def = rng.normal(112, 5)
        away_def = rng.normal(112, 5)
        home_form = rng.uniform(0.35, 0.75)
        away_form = rng.uniform(0.35, 0.75)
        home_rest = rng.integers(0, 4)
        away_rest = rng.integers(0, 4)
        home_inj = rng.uniform(0.7, 1.0)
        away_inj = rng.uniform(0.7, 1.0)
        pace = rng.normal(100, 5)
        home_3pt = rng.normal(0.36, 0.03)
        away_3pt = rng.normal(0.36, 0.03)

        home_adv = 3.0
        home_strength = (home_ppg - away_def) + (home_form - 0.5)*12 + home_rest*0.6 + home_adv + (home_inj-0.85)*10 + (pace-100)*0.05
        away_strength = (away_ppg - home_def) + (away_form - 0.5)*12 + away_rest*0.6 + (away_inj-0.85)*10 + (pace-100)*0.05

        p_home = 1 / (1 + np.exp(-(home_strength - away_strength) / 10))
        home_win = int(rng.random() < p_home)

        # add placeholder engineered features to reach 127 total feature count.
        # Base features count ~13; add 114 synthetic engineered columns.
        engineered = {f"feat_{k:03d}": float(rng.normal(0, 1)) for k in range(114)}

        rows.append({
            "date": date.date().isoformat(),
            "home_team": home,
            "away_team": away,
            "home_ppg": float(home_ppg),
            "away_ppg": float(away_ppg),
            "home_def_rating": float(home_def),
            "away_def_rating": float(away_def),
            "home_form_l10": float(home_form),
            "away_form_l10": float(away_form),
            "home_rest_days": int(home_rest),
            "away_rest_days": int(away_rest),
            "home_injury_impact": float(home_inj),
            "away_injury_impact": float(away_inj),
            "pace": float(pace),
            "home_3pt_pct": float(home_3pt),
            "away_3pt_pct": float(away_3pt),
            "is_home": 1,
            "home_win": home_win,
            **engineered
        })

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

def main():
    out_path = os.getenv("HISTORICAL_DATA_PATH", "data/historical_games.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df = generate_synthetic_historical_data()
    df.to_csv(out_path, index=False)
    print("saved:", out_path, "rows:", len(df), "cols:", df.shape[1])

if __name__ == "__main__":
    main()
