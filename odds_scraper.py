import requests
import pandas as pd
from datetime import datetime
import sqlite3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class OddsScraper:
    def __init__(self):
        self.api_key = config.ODDS_API_KEY
        self.base_url = "https://api.the-odds-api.com/v4"
        
    def get_sports(self):
        """Get list of available sports"""
        url = f"{self.base_url}/sports/"
        params = {'apiKey': self.api_key}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Error fetching sports: {e}")
            return []
    
    def get_odds(self, sport='basketball_nba'):
        """Fetch odds for a specific sport"""
        url = f"{self.base_url}/sports/{sport}/odds/"
        
        params = {
            'apiKey': self.api_key,
            'regions': 'us',
            'markets': 'h2h,spreads,totals',
            'oddsFormat': 'american'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            remaining = response.headers.get('x-requests-remaining')
            used = response.headers.get('x-requests-used')
            print(f"API Requests - Used: {used}, Remaining: {remaining}")
            
            return response.json()
        except Exception as e:
            print(f"Error fetching odds: {e}")
            return []
    
    def parse_odds(self, raw_data):
        """Parse raw odds data into structured DataFrame"""
        if not raw_data:
            print("No data to parse")
            return pd.DataFrame()
        
        games = []
        
        for game in raw_data:
            game_info = {
                'game_id': game['id'],
                'sport': game['sport_key'],
                'commence_time': game['commence_time'],
                'home_team': game['home_team'],
                'away_team': game['away_team']
            }
            
            # Extract bookmaker odds
            for bookmaker in game.get('bookmakers', []):
                book_key = bookmaker['key'].replace('_', '')
                
                for market in bookmaker.get('markets', []):
                    market_key = market['key']
                    
                    if market_key == 'h2h':  # Moneyline
                        for outcome in market['outcomes']:
                            team_type = 'home' if outcome['name'] == game['home_team'] else 'away'
                            game_info[f'{book_key}_{team_type}_ml'] = outcome['price']
                    
                    elif market_key == 'spreads':  # Spreads
                        for outcome in market['outcomes']:
                            team_type = 'home' if outcome['name'] == game['home_team'] else 'away'
                            game_info[f'{book_key}_{team_type}_spread'] = outcome['point']
                            game_info[f'{book_key}_{team_type}_spread_odds'] = outcome['price']
                    
                    elif market_key == 'totals':  # Totals
                        for outcome in market['outcomes']:
                            total_type = outcome['name'].lower()
                            game_info[f'{book_key}_total_points'] = outcome['point']
                            game_info[f'{book_key}_{total_type}_odds'] = outcome['price']
            
            games.append(game_info)
        
        df = pd.DataFrame(games)
        df['fetch_timestamp'] = datetime.now()
        return df
    
    def save_to_db(self, df):
        """Save odds to database"""
        if df.empty:
            print("No data to save")
            return False
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(config.ODDS_DB_PATH), exist_ok=True)
        
        conn = sqlite3.connect(config.ODDS_DB_PATH)
        df.to_sql('odds_history', conn, if_exists='append', index=False)
        conn.close()
        
        print(f"✓ Saved {len(df)} games to database")
        return True
    
    def get_latest_odds(self):
        """Get most recent odds from database"""
        if not os.path.exists(config.ODDS_DB_PATH):
            print("No database found. Run scraper first.")
            return pd.DataFrame()
        
        conn = sqlite3.connect(config.ODDS_DB_PATH)
        query = """
            SELECT * FROM odds_history 
            WHERE fetch_timestamp = (SELECT MAX(fetch_timestamp) FROM odds_history)
        """
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df

def main():
    """Main execution"""
    
    print("SPORTS BETTING ODDS SCRAPER")
   
    scraper = OddsScraper()
    
    # Test API connection
    print("\n1. Testing API connection...")
    sports = scraper.get_sports()
    
    if not sports:
        print("✗ API connection failed. Check your API key in config.py")
        return
    
    print(f"✓ API connected. Found {len(sports)} available sports")
    
    # Show available sports
    print("\nAvailable sports:")
    for sport in sports[:10]:
        print(f"  • {sport['key']}: {sport['title']}")
    
    # Fetch NBA odds
    print("\n2. Fetching NBA odds...")
    odds_data = scraper.get_odds('basketball_nba')
    
    if not odds_data:
        print("✗ No NBA games found or API error")
        return
    
    print(f"✓ Found {len(odds_data)} NBA games")
    
    # Parse odds
    print("\n3. Parsing odds data...")
    df = scraper.parse_odds(odds_data)
    print(f"✓ Parsed {len(df)} games")
    
    # Display sample
    print("\n4. Sample data:")
    for idx, row in df.head(3).iterrows():
        print(f"\n{row['away_team']} @ {row['home_team']}")
        print(f"Time: {row['commence_time']}")
        if 'draftkings_home_ml' in row:
            print(f"DraftKings ML: Home {row.get('draftkings_home_ml', 'N/A')} | Away {row.get('draftkings_away_ml', 'N/A')}")
    
    # Save to database
    print("\n5. Saving to database...")
    scraper.save_to_db(df)
    

    print("✓ SCRAPING COMPLETE!")
    print(f"\nData saved to: {config.ODDS_DB_PATH}")
    print("Next step: Run 'python src/data_generator.py' to create training data")

if __name__ == "__main__":
    main()