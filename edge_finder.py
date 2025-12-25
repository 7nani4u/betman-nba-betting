import pandas as pd
import numpy as np
import joblib
import sqlite3
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class EdgeFinder:
    def __init__(self):
        self.model = None
        self.feature_names = None
        self.load_model()
    
    def load_model(self):
        """Load trained model"""
        model_path = 'models/betting_model.pkl'
        
        if not os.path.exists(model_path):
            print(f"✗ Model not found. Run 'python src/model_training.py' first")
            return False
        
        data = joblib.load(model_path)
        self.model = data['model']
        self.feature_names = data['feature_names']
        print("✓ Model loaded successfully")
        return True
    
    def american_to_prob(self, odds):
        """Convert American odds to implied probability"""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
    
    def american_to_decimal(self, odds):
        """Convert American odds to decimal"""
        if odds > 0:
            return (odds / 100) + 1
        else:
            return (100 / abs(odds)) + 1
    
    def kelly_criterion(self, prob, odds, fraction=0.25):
        """Calculate Kelly Criterion bet size"""
        decimal_odds = self.american_to_decimal(odds)
        q = 1 - prob
        kelly = (prob * decimal_odds - 1) / (decimal_odds - 1)
        
        # Use fractional Kelly for safety
        return max(0, kelly * fraction)
    
    def calculate_edge(self, our_prob, market_prob):
        """Calculate betting edge"""
        return (our_prob - market_prob) * 100
    
    def expected_value(self, prob, odds):
        """Calculate expected value"""
        decimal_odds = self.american_to_decimal(odds)
        ev = (prob * (decimal_odds - 1)) - (1 - prob)
        return ev * 100
    
    def get_confidence(self, edge):
        """Determine confidence level"""
        if edge >= 10:
            return 'Very High'
        elif edge >= 6:
            return 'High'
        elif edge >= 3:
            return 'Medium'
        return 'Low'
    
    def create_mock_opportunities(self):
        """Create mock betting opportunities for demo"""
        print("\n📊 Generating mock betting opportunities...")
        
        opportunities = []
        
        games = [
            {
                'home': 'Boston Celtics', 'away': 'Miami Heat',
                'time': '7:30 PM ET', 'our_prob': 0.623, 'odds': -140, 'book': 'DraftKings'
            },
            {
                'home': 'Los Angeles Lakers', 'away': 'Golden State Warriors',
                'time': '10:00 PM ET', 'our_prob': 0.581, 'odds': -115, 'book': 'FanDuel'
            },
            {
                'home': 'Milwaukee Bucks', 'away': 'Philadelphia 76ers',
                'time': '8:00 PM ET', 'our_prob': 0.547, 'odds': -105, 'book': 'BetMGM'
            },
            {
                'home': 'Denver Nuggets', 'away': 'Phoenix Suns',
                'time': '9:00 PM ET', 'our_prob': 0.692, 'odds': -180, 'book': 'DraftKings'
            },
            {
                'home': 'Dallas Mavericks', 'away': 'Minnesota Timberwolves',
                'time': '8:30 PM ET', 'our_prob': 0.558, 'odds': -120, 'book': 'FanDuel'
            }
        ]
        
        for game in games:
            market_prob = self.american_to_prob(game['odds'])
            edge = self.calculate_edge(game['our_prob'], market_prob)
            
            if edge >= config.MIN_EDGE:
                ev = self.expected_value(game['our_prob'], game['odds'])
                kelly_size = self.kelly_criterion(game['our_prob'], game['odds'])
                
                opportunities.append({
                    'game': f"{game['away']} @ {game['home']}",
                    'time': game['time'],
                    'prediction': f"{game['home']} Win",
                    'our_prob': game['our_prob'] * 100,
                    'market_prob': market_prob * 100,
                    'edge': edge,
                    'recommended_bet': f"{game['home']} ML",
                    'odds': game['odds'],
                    'bookmaker': game['book'],
                    'confidence': self.get_confidence(edge),
                    'kelly_size': kelly_size * 100,
                    'expected_value': ev
                })
        
        return pd.DataFrame(opportunities)
    
    def find_opportunities(self):
        """Find all betting opportunities"""
        if self.model is None:
            return pd.DataFrame()
        
        # Try to load real odds data
        if os.path.exists(config.ODDS_DB_PATH):
            conn = sqlite3.connect(config.ODDS_DB_PATH)
            query = "SELECT * FROM odds_history WHERE fetch_timestamp = (SELECT MAX(fetch_timestamp) FROM odds_history)"
            
            try:
                odds_df = pd.read_sql_query(query, conn)
                conn.close()
                
                if not odds_df.empty:
                    print(f"✓ Found {len(odds_df)} games in database")
                    # Process real odds here...
                    # For now, return mock data
                    return self.create_mock_opportunities()
            except:
                conn.close()
        
        # Return mock opportunities
        return self.create_mock_opportunities()

def main():
    """Main execution"""
    print("=" * 60)
    print("BETTING EDGE FINDER")
    print("=" * 60)
    
    finder = EdgeFinder()
    
    if finder.model is None:
        return
    
    print(f"\nSearching for edges > {config.MIN_EDGE}%...")
    
    opportunities = finder.find_opportunities()
    
    if opportunities.empty:
        print("\n✗ No opportunities found with sufficient edge")
        return
    
    print(f"\n✓ Found {len(opportunities)} opportunities!")
    print("\n" + "=" * 60)
    print("TOP BETTING OPPORTUNITIES")
    print("=" * 60)
    
    # Sort by edge
    opportunities = opportunities.sort_values('edge', ascending=False)
    
    for idx, opp in opportunities.iterrows():
        print(f"\n{'='*60}")
        print(f"🏀 {opp['game']}")
        print(f"   Time: {opp['time']}")
        print(f"   Confidence: {opp['confidence']}")
        print(f"\n   Our Probability:    {opp['our_prob']:.1f}%")
        print(f"   Market Probability: {opp['market_prob']:.1f}%")
        print(f"   Edge:               +{opp['edge']:.1f}%")
        print(f"\n   Recommended: {opp['recommended_bet']}")
        print(f"   Odds: {opp['odds']} ({opp['bookmaker']})")
        print(f"   Kelly Size: {opp['kelly_size']:.1f}% of bankroll")
        print(f"   Expected Value: +{opp['expected_value']:.1f}%")
    
    print("\n" + "=" * 60)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\nTotal opportunities: {len(opportunities)}")
    print(f"Average edge: +{opportunities['edge'].mean():.1f}%")
    print(f"Average EV: +{opportunities['expected_value'].mean():.1f}%")
    print("\nNext step: Run 'streamlit run dashboard/app.py' to view dashboard")

if __name__ == "__main__":
    main()
    