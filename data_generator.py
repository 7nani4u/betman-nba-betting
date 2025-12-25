import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

def generate_historical_data():
    """Generate synthetic historical game data for training"""
    print("GENERATING HISTORICAL TRAINING DATA")
    
    
    # NBA teams
    teams = [
        'Boston Celtics', 'Brooklyn Nets', 'New York Knicks', 'Philadelphia 76ers', 'Toronto Raptors',
        'Chicago Bulls', 'Cleveland Cavaliers', 'Detroit Pistons', 'Indiana Pacers', 'Milwaukee Bucks',
        'Atlanta Hawks', 'Charlotte Hornets', 'Miami Heat', 'Orlando Magic', 'Washington Wizards',
        'Denver Nuggets', 'Minnesota Timberwolves', 'Oklahoma City Thunder', 'Portland Trail Blazers', 'Utah Jazz',
        'Golden State Warriors', 'LA Clippers', 'Los Angeles Lakers', 'Phoenix Suns', 'Sacramento Kings',
        'Dallas Mavericks', 'Houston Rockets', 'Memphis Grizzlies', 'New Orleans Pelicans', 'San Antonio Spurs'
    ]
    
    np.random.seed(42)
    
    games = []
    start_date = datetime(2023, 10, 1)
    
    print("\nGenerating 2000 historical games...")
    
    for i in range(2000):
        home_team = np.random.choice(teams)
        away_team = np.random.choice([t for t in teams if t != home_team])
        
        game_date = start_date + timedelta(days=i % 180)
        
        # Generate realistic stats
        home_ppg = np.random.normal(112, 8)
        away_ppg = np.random.normal(112, 8)
        
        home_def_rating = np.random.normal(110, 5)
        away_def_rating = np.random.normal(110, 5)
        
        home_form = np.random.uniform(0.3, 0.8)
        away_form = np.random.uniform(0.3, 0.8)
        
        home_rest = np.random.choice([0, 1, 2, 3, 4], p=[0.15, 0.35, 0.25, 0.15, 0.10])
        away_rest = np.random.choice([0, 1, 2, 3, 4], p=[0.15, 0.35, 0.25, 0.15, 0.10])
        
        home_injury_impact = np.random.uniform(0.7, 1.0)
        away_injury_impact = np.random.uniform(0.7, 1.0)
        
        pace = np.random.normal(100, 5)
        
        home_3pt_pct = np.random.normal(0.36, 0.03)
        away_3pt_pct = np.random.normal(0.36, 0.03)
        
        # Determine winner (home court advantage)
        home_advantage = 3.5
        home_strength = home_ppg - away_def_rating + home_form * 10 + home_rest * 0.5 + home_advantage
        away_strength = away_ppg - home_def_rating + away_form * 10 + away_rest * 0.5
        
        home_win_prob = 1 / (1 + np.exp(-(home_strength - away_strength) / 10))
        home_win = 1 if np.random.random() < home_win_prob else 0
        
        # Generate score
        if home_win:
            home_score = int(np.random.normal(home_ppg + 3, 5))
            away_score = int(np.random.normal(away_ppg - 3, 5))
        else:
            home_score = int(np.random.normal(home_ppg - 3, 5))
            away_score = int(np.random.normal(away_ppg + 3, 5))
        
        total_points = home_score + away_score
        
        # Generate odds (implied probability with vig)
        true_prob = home_win_prob
        implied_prob = true_prob * 1.05  # Add bookmaker vig
        
        if implied_prob > 0.5:
            home_ml = int(-100 * implied_prob / (1 - implied_prob))
            away_ml = int(100 * (1 - implied_prob) / implied_prob)
        else:
            home_ml = int(100 * (1 - implied_prob) / implied_prob)
            away_ml = int(-100 * implied_prob / (1 - implied_prob))
        
        games.append({
            'game_date': game_date,
            'home_team': home_team,
            'away_team': away_team,
            'home_ppg': home_ppg,
            'away_ppg': away_ppg,
            'home_def_rating': home_def_rating,
            'away_def_rating': away_def_rating,
            'home_form_l10': home_form,
            'away_form_l10': away_form,
            'home_rest_days': home_rest,
            'away_rest_days': away_rest,
            'home_injury_impact': home_injury_impact,
            'away_injury_impact': away_injury_impact,
            'pace': pace,
            'home_3pt_pct': home_3pt_pct,
            'away_3pt_pct': away_3pt_pct,
            'is_home': 1,
            'home_win': home_win,
            'home_score': home_score,
            'away_score': away_score,
            'total_points': total_points,
            'home_ml_odds': home_ml,
            'away_ml_odds': away_ml,
            'vegas_home_prob': implied_prob
        })
    
    df = pd.DataFrame(games)
    
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    df.to_csv(config.HISTORICAL_DATA_PATH, index=False)
    
    print(f"✓ Generated {len(df)} games")
    print(f"✓ Saved to: {config.HISTORICAL_DATA_PATH}")
    
    # Display stats
    
    print("DATASET STATISTICS")
    
    print(f"Total games: {len(df)}")
    print(f"Home win rate: {df['home_win'].mean() * 100:.1f}%")
    print(f"Average total points: {df['total_points'].mean():.1f}")
    print(f"Date range: {df['game_date'].min()} to {df['game_date'].max()}")
    print("\nSample data:")
    print(df.head())
    
    print("\n✓ Data generation complete!")
    

if __name__ == "__main__":
    generate_historical_data()