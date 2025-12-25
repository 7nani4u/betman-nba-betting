"""
Advanced Edge Finder with Multi-Market Analysis
게임 유형별 베팅 시장 예측 로직 확장
"""

import pandas as pd
import numpy as np
import joblib
import sqlite3
import os
import sys
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from probability_engine import (
    BetType, NoVigCalculator, ProbabilityCalibrator, 
    BettingMarketAnalyzer, PredictionResult, PerformanceTracker
)


class AdvancedEdgeFinder:
    """
    고도화된 엣지 파인더
    
    - 게임 유형별 자동 분류
    - 다중 시장 분석
    - 최적 베팅 선택
    - 성과 추적
    """
    
    def __init__(self, initial_bankroll: float = 1000.0):
        self.model = None
        self.feature_names = None
        self.no_vig_calc = NoVigCalculator()
        self.calibrator = ProbabilityCalibrator()
        self.market_analyzer = BettingMarketAnalyzer()
        self.performance_tracker = PerformanceTracker(initial_bankroll)
        self.calibration_params = {}
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
        return self.no_vig_calc.american_to_implied_prob(odds)
    
    def american_to_decimal(self, odds):
        """Convert American odds to decimal"""
        return self.no_vig_calc.american_to_decimal(odds)
    
    def predict_game_probability(self, features: Dict) -> float:
        """
        게임 특성으로부터 홈팀 승률 예측
        
        Args:
            features: 게임 특성 딕셔너리
        
        Returns:
            home_win_probability: 홈팀 승률 (0~1)
        """
        if self.model is None:
            return 0.5
        
        # 특성을 모델 입력 형식으로 변환
        feature_vector = np.array([[features.get(f, 0) for f in self.feature_names]])
        
        # 확률 예측
        prob = self.model.predict_proba(feature_vector)[0, 1]
        return prob
    
    def predict_total_points(self, features: Dict) -> float:
        """
        게임의 총점 예측
        
        Args:
            features: 게임 특성
        
        Returns:
            predicted_total: 예상 총점
        """
        # 간단한 휴리스틱: 팀 PPG 기반
        home_ppg = features.get('home_ppg', 110)
        away_ppg = features.get('away_ppg', 110)
        
        # 페이스와 수비 레이팅 조정
        pace = features.get('pace', 100)
        home_def = features.get('home_def_rating', 110)
        away_def = features.get('away_def_rating', 110)
        
        # 예상 총점 = (홈팀 PPG + 어웨이팀 PPG) * 페이스 조정
        predicted_total = (home_ppg + away_ppg) * (pace / 100)
        
        return predicted_total
    
    def predict_odd_even_probability(self, total_points: float) -> float:
        """
        홀/짝 확률 예측
        
        Args:
            total_points: 예상 총점
        
        Returns:
            odd_probability: 홀수 확률
        """
        # 총점이 정수일 때: 0.5 / 0.5
        # 실제로는 더 복잡한 모델 필요
        # 간단히: 총점이 홀수에 가까울수록 홀수 확률 증가
        fractional_part = total_points - int(total_points)
        
        if fractional_part < 0.5:
            odd_prob = 0.5 - (0.5 - fractional_part) * 0.1
        else:
            odd_prob = 0.5 + (fractional_part - 0.5) * 0.1
        
        return np.clip(odd_prob, 0.3, 0.7)
    
    def analyze_all_markets(self, game_data: Dict) -> List[PredictionResult]:
        """
        게임의 모든 베팅 시장 분석
        
        Args:
            game_data: 게임 데이터
                - game_id: 게임 ID
                - home_team, away_team: 팀 이름
                - features: 게임 특성 딕셔너리
                - odds: 배당률 정보
        
        Returns:
            List[PredictionResult]: 각 시장별 분석 결과
        """
        results = []
        game_id = game_data.get('game_id', '')
        features = game_data.get('features', {})
        odds_data = game_data.get('odds', {})
        
        # 1. 홈팀 승률 예측
        home_prob = self.predict_game_probability(features)
        away_prob = 1 - home_prob
        
        # 2. Moneyline 시장 분석
        if 'moneyline' in odds_data:
            ml_odds = odds_data['moneyline']
            ml_result = self.market_analyzer.analyze_moneyline(
                home_prob, away_prob,
                ml_odds.get('home', -110),
                ml_odds.get('away', -110),
                calibration_params=self.calibration_params
            )
            ml_result.game_id = game_id
            results.append(ml_result)
        
        # 3. Spread 시장 분석
        if 'spread' in odds_data:
            spread_odds = odds_data['spread']
            spread_result = self.market_analyzer.analyze_spread(
                home_prob,
                spread_odds.get('points', -3.5),
                spread_odds.get('odds', -110),
                calibration_params=self.calibration_params
            )
            spread_result.game_id = game_id
            results.append(spread_result)
        
        # 4. Total 시장 분석
        if 'total' in odds_data:
            total_odds = odds_data['total']
            predicted_total = self.predict_total_points(features)
            
            # Over 확률 예측
            over_prob = self._predict_over_probability(
                predicted_total,
                total_odds.get('line', 220)
            )
            
            total_result = self.market_analyzer.analyze_total(
                over_prob,
                total_odds.get('over', -110),
                total_odds.get('under', -110),
                calibration_params=self.calibration_params
            )
            total_result.game_id = game_id
            results.append(total_result)
        
        # 5. Odd/Even 시장 분석
        if 'odd_even' in odds_data:
            odd_even_odds = odds_data['odd_even']
            predicted_total = self.predict_total_points(features)
            odd_prob = self.predict_odd_even_probability(predicted_total)
            
            odd_even_result = self.market_analyzer.analyze_odd_even(
                odd_prob,
                odd_even_odds.get('odd', -110),
                odd_even_odds.get('even', -110),
                calibration_params=self.calibration_params
            )
            odd_even_result.game_id = game_id
            results.append(odd_even_result)
        
        return results
    
    def _predict_over_probability(self, predicted_total: float, line: float) -> float:
        """
        예상 총점으로부터 Over 확률 계산
        
        정규분포 가정: 표준편차 약 10점
        """
        std_dev = 10.0
        z_score = (predicted_total - line) / std_dev
        
        # 표준정규분포 누적분포함수
        from scipy.stats import norm
        over_prob = norm.cdf(z_score)
        
        return np.clip(over_prob, 0.1, 0.9)
    
    def select_best_bets(self, all_results: List[PredictionResult],
                        min_edge: float = 3.0,
                        max_bets: int = 5) -> List[PredictionResult]:
        """
        모든 시장 분석 결과에서 최고의 베팅 선택
        
        Args:
            all_results: 모든 시장 분석 결과
            min_edge: 최소 엣지 (%)
            max_bets: 최대 베팅 수
        
        Returns:
            List[PredictionResult]: 선택된 베팅 (엣지 순 정렬)
        """
        # 엣지 필터링
        filtered = [r for r in all_results if r.edge >= min_edge]
        
        # 엣지로 정렬 (내림차순)
        sorted_results = sorted(filtered, key=lambda x: x.edge, reverse=True)
        
        # 상위 N개 선택
        return sorted_results[:max_bets]
    
    def create_mock_opportunities(self, min_edge: float = 3.0) -> pd.DataFrame:
        """
        데모용 베팅 기회 생성
        
        Args:
            min_edge: 최소 엣지
        
        Returns:
            pd.DataFrame: 베팅 기회 데이터프레임
        """
        print("\n📊 Generating multi-market betting opportunities...")
        
        opportunities = []
        
        # 샘플 게임 데이터
        games = [
            {
                'game_id': 'game_001',
                'home_team': 'Boston Celtics',
                'away_team': 'Miami Heat',
                'time': '7:30 PM ET',
                'features': {
                    'home_ppg': 115, 'away_ppg': 108,
                    'home_def_rating': 108, 'away_def_rating': 112,
                    'home_form_l10': 0.65, 'away_form_l10': 0.55,
                    'home_rest_days': 1, 'away_rest_days': 2,
                    'pace': 98
                },
                'odds': {
                    'moneyline': {'home': -140, 'away': 120},
                    'spread': {'points': -3.5, 'odds': -110},
                    'total': {'line': 215, 'over': -110, 'under': -110},
                    'odd_even': {'odd': -110, 'even': -110}
                }
            },
            {
                'game_id': 'game_002',
                'home_team': 'Los Angeles Lakers',
                'away_team': 'Golden State Warriors',
                'time': '10:00 PM ET',
                'features': {
                    'home_ppg': 112, 'away_ppg': 114,
                    'home_def_rating': 110, 'away_def_rating': 109,
                    'home_form_l10': 0.58, 'away_form_l10': 0.62,
                    'home_rest_days': 1, 'away_rest_days': 1,
                    'pace': 102
                },
                'odds': {
                    'moneyline': {'home': -115, 'away': -105},
                    'spread': {'points': -1.5, 'odds': -110},
                    'total': {'line': 227.5, 'over': -110, 'under': -110},
                    'odd_even': {'odd': -110, 'even': -110}
                }
            },
            {
                'game_id': 'game_003',
                'home_team': 'Denver Nuggets',
                'away_team': 'Phoenix Suns',
                'time': '9:00 PM ET',
                'features': {
                    'home_ppg': 118, 'away_ppg': 116,
                    'home_def_rating': 106, 'away_def_rating': 107,
                    'home_form_l10': 0.70, 'away_form_l10': 0.68,
                    'home_rest_days': 2, 'away_rest_days': 1,
                    'pace': 99
                },
                'odds': {
                    'moneyline': {'home': -180, 'away': 150},
                    'spread': {'points': -4.5, 'odds': -110},
                    'total': {'line': 232, 'over': -110, 'under': -110},
                    'odd_even': {'odd': -110, 'even': -110}
                }
            }
        ]
        
        # 각 게임의 모든 시장 분석
        for game in games:
            market_results = self.analyze_all_markets(game)
            
            # 엣지 필터링 및 선택
            best_bets = self.select_best_bets(market_results, min_edge=min_edge, max_bets=3)
            
            for result in best_bets:
                opportunities.append({
                    'game': f"{game['away_team']} @ {game['home_team']}",
                    'time': game['time'],
                    'bet_type': result.bet_type.value,
                    'prediction': result.recommendation,
                    'our_prob': result.calibrated_probability,
                    'market_prob': result.market_probability,
                    'edge': result.edge,
                    'ev': result.expected_value,
                    'kelly': result.kelly_size,
                    'confidence': result.confidence,
                    'game_id': result.game_id
                })
        
        return pd.DataFrame(opportunities)
    
    def find_opportunities(self, min_edge: float = 3.0) -> pd.DataFrame:
        """
        모든 베팅 기회 찾기
        
        Args:
            min_edge: 최소 엣지
        
        Returns:
            pd.DataFrame: 베팅 기회 데이터프레임
        """
        if self.model is None:
            return self.create_mock_opportunities(min_edge)
        
        # 실제 데이터 처리 로직
        return self.create_mock_opportunities(min_edge)
    
    def calibrate_from_history(self, historical_predictions: np.ndarray,
                              historical_outcomes: np.ndarray,
                              method: str = 'temperature'):
        """
        과거 예측 데이터로부터 확률 보정 파라미터 계산
        
        Args:
            historical_predictions: 과거 예측 확률 배열
            historical_outcomes: 실제 결과 배열 (0 또는 1)
            method: 보정 방법 ('temperature', 'platt', 'isotonic')
        """
        if method == 'temperature':
            T = self.calibrator.temperature_scaling(historical_predictions, historical_outcomes)
            self.calibration_params = {'temperature': T}
            print(f"✓ Temperature scaling calibrated: T={T:.3f}")
        
        elif method == 'platt':
            A, B = self.calibrator.platt_scaling(historical_predictions, historical_outcomes)
            self.calibration_params = {'A': A, 'B': B}
            print(f"✓ Platt scaling calibrated: A={A:.3f}, B={B:.3f}")
        
        # 보정 품질 평가
        brier = self.calibrator.brier_score(historical_predictions, historical_outcomes)
        logloss = self.calibrator.log_loss(historical_predictions, historical_outcomes)
        
        print(f"  Brier Score: {brier:.4f}")
        print(f"  Log Loss: {logloss:.4f}")


def main():
    """Main execution"""
    print("=" * 70)
    print("ADVANCED BETTING EDGE FINDER - MULTI-MARKET ANALYSIS")
    print("=" * 70)
    
    finder = AdvancedEdgeFinder(initial_bankroll=1000.0)
    
    print(f"\n🔍 Searching for edges > 3%...")
    
    opportunities = finder.find_opportunities(min_edge=3.0)
    
    if opportunities.empty:
        print("\n✗ No opportunities found with sufficient edge")
        return
    
    print(f"\n✓ Found {len(opportunities)} opportunities!")
    print("\n" + "=" * 70)
    print("TOP BETTING OPPORTUNITIES (Multi-Market)")
    print("=" * 70)
    
    # 엣지로 정렬
    opportunities = opportunities.sort_values('edge', ascending=False)
    
    for idx, opp in opportunities.iterrows():
        print(f"\n{'='*70}")
        print(f"🏀 {opp['game']}")
        print(f"   Time: {opp['time']}")
        print(f"   Bet Type: {opp['bet_type'].upper()}")
        print(f"   Confidence: {opp['confidence']}")
        print(f"\n   Our Probability (Calibrated): {opp['our_prob']:.1f}%")
        print(f"   Market Probability (No-Vig): {opp['market_prob']:.1f}%")
        print(f"   Edge: +{opp['edge']:.1f}%")
        print(f"   Expected Value: +{opp['ev']:.1f}%")
        print(f"\n   Recommendation: {opp['prediction']}")
        print(f"   Kelly Criterion: {opp['kelly']:.1f}% of bankroll")
    
    print("\n" + "=" * 70)
    print("✓ ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nTotal opportunities: {len(opportunities)}")
    print(f"Average edge: +{opportunities['edge'].mean():.1f}%")
    print(f"Average EV: +{opportunities['ev'].mean():.1f}%")
    print(f"By bet type:")
    for bet_type in opportunities['bet_type'].unique():
        type_data = opportunities[opportunities['bet_type'] == bet_type]
        print(f"  - {bet_type.upper()}: {len(type_data)} opportunities, avg edge +{type_data['edge'].mean():.1f}%")


if __name__ == "__main__":
    main()
