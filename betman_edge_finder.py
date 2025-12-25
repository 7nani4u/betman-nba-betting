"""
배트맨 통합 엣지 파인더

배트맨 사이트의 NBA 승부식 배당률을 수집하고,
고도화된 확률 엔진과 통합하여 최적의 베팅 기회를 식별합니다.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from probability_engine import (
    NoVigCalculator, ProbabilityCalibrator, BettingMarketAnalyzer,
    PerformanceTracker, BetType, PredictionResult
)
from betman_scraper import BetmanNBAScraper, BetmanOddsAnalyzer

logger = logging.getLogger(__name__)


class BetmanIntegratedEdgeFinder:
    """
    배트맨 데이터를 통합한 엣지 파인더
    
    배트맨의 실시간 배당률과 우리의 예측 모델을 결합하여
    최고의 베팅 기회를 식별합니다.
    """
    
    def __init__(self, initial_bankroll: float = 1000.0):
        """
        초기화
        
        Args:
            initial_bankroll: 초기 자본
        """
        self.scraper = BetmanNBAScraper(headless=True)
        self.analyzer = BetmanOddsAnalyzer()
        self.no_vig_calc = NoVigCalculator()
        self.calibrator = ProbabilityCalibrator()
        self.performance_tracker = PerformanceTracker(initial_bankroll)
        self.calibration_params = {}
        
        # NBA 팀 매핑 (배트맨 팀명 -> 표준 팀명)
        self.team_mapping = {
            'Lakers': 'Los Angeles Lakers',
            'Celtics': 'Boston Celtics',
            'Warriors': 'Golden State Warriors',
            'Suns': 'Phoenix Suns',
            'Nuggets': 'Denver Nuggets',
            'Bucks': 'Milwaukee Bucks',
            'Heat': 'Miami Heat',
            'Mavericks': 'Dallas Mavericks',
            'Timberwolves': 'Minnesota Timberwolves',
            '76ers': 'Philadelphia 76ers',
            'Nets': 'Brooklyn Nets',
            'Knicks': 'New York Knicks',
            'Raptors': 'Toronto Raptors',
            'Bulls': 'Chicago Bulls',
            'Cavaliers': 'Cleveland Cavaliers',
            'Pistons': 'Detroit Pistons',
            'Pacers': 'Indiana Pacers',
            'Hawks': 'Atlanta Hawks',
            'Hornets': 'Charlotte Hornets',
            'Magic': 'Orlando Magic',
            'Wizards': 'Washington Wizards',
            'Grizzlies': 'Memphis Grizzlies',
            'Pelicans': 'New Orleans Pelicans',
            'Spurs': 'San Antonio Spurs',
            'Kings': 'Sacramento Kings',
            'Clippers': 'LA Clippers',
            'Trail Blazers': 'Portland Trail Blazers',
            'Jazz': 'Utah Jazz',
            'Thunder': 'Oklahoma City Thunder',
            'Rockets': 'Houston Rockets'
        }
    
    def fetch_betman_odds(self) -> pd.DataFrame:
        """
        배트맨에서 실시간 배당률 수집
        
        Returns:
            pd.DataFrame: 경기 배당률 데이터
        """
        logger.info("배트맨 사이트에서 NBA 경기 데이터 수집 중...")
        matches_df = self.scraper.scrape_nba_odds()
        
        if matches_df.empty:
            logger.warning("배트맨에서 수집된 데이터가 없습니다. 샘플 데이터를 사용합니다.")
            return self._create_sample_betman_data()
        
        return matches_df
    
    def _create_sample_betman_data(self) -> pd.DataFrame:
        """
        테스트용 샘플 배트맨 데이터 생성
        
        Returns:
            pd.DataFrame: 샘플 경기 데이터
        """
        sample_data = [
            {
                'match_id': '1001',
                'home_team': 'Lakers',
                'away_team': 'Celtics',
                'home_odds': 1.95,
                'draw_odds': 15.00,
                'away_odds': 1.95,
                'match_time': '10:30 PM',
                'fetch_time': datetime.now()
            },
            {
                'match_id': '1002',
                'home_team': 'Warriors',
                'away_team': 'Suns',
                'home_odds': 2.10,
                'draw_odds': 15.00,
                'away_odds': 1.80,
                'match_time': '08:00 PM',
                'fetch_time': datetime.now()
            },
            {
                'match_id': '1003',
                'home_team': 'Nuggets',
                'away_team': 'Heat',
                'home_odds': 1.85,
                'draw_odds': 15.00,
                'away_odds': 2.05,
                'match_time': '09:00 PM',
                'fetch_time': datetime.now()
            },
            {
                'match_id': '1004',
                'home_team': 'Bucks',
                'away_team': 'Mavericks',
                'home_odds': 1.90,
                'draw_odds': 15.00,
                'away_odds': 2.00,
                'match_time': '07:30 PM',
                'fetch_time': datetime.now()
            },
            {
                'match_id': '1005',
                'home_team': 'Timberwolves',
                'away_team': '76ers',
                'home_odds': 2.05,
                'draw_odds': 15.00,
                'away_odds': 1.85,
                'match_time': '08:30 PM',
                'fetch_time': datetime.now()
            }
        ]
        
        return pd.DataFrame(sample_data)
    
    def predict_game_outcome(self, home_team: str, away_team: str) -> Dict:
        """
        게임 결과 예측
        
        Args:
            home_team: 홈팀
            away_team: 원정팀
        
        Returns:
            Dict: 예측 확률
        """
        # 실제 모델 사용 시 여기에 XGBoost 모델 예측 코드 추가
        # 현재는 휴리스틱 기반 예측
        
        # 팀 강도 (실제로는 모델에서 계산)
        team_strength = {
            'Lakers': 0.65,
            'Celtics': 0.68,
            'Warriors': 0.62,
            'Suns': 0.64,
            'Nuggets': 0.70,
            'Bucks': 0.66,
            'Heat': 0.60,
            'Mavericks': 0.63,
            'Timberwolves': 0.61,
            '76ers': 0.59,
            'Nets': 0.45,
            'Knicks': 0.58,
            'Raptors': 0.50,
            'Bulls': 0.48,
            'Cavaliers': 0.55,
            'Pistons': 0.42,
            'Pacers': 0.52,
            'Hawks': 0.51,
            'Hornets': 0.45,
            'Magic': 0.47,
            'Wizards': 0.44,
            'Grizzlies': 0.54,
            'Pelicans': 0.53,
            'Spurs': 0.46,
            'Kings': 0.49,
            'Clippers': 0.60,
            'Trail Blazers': 0.48,
            'Jazz': 0.52,
            'Thunder': 0.56,
            'Rockets': 0.50
        }
        
        home_strength = team_strength.get(home_team, 0.50)
        away_strength = team_strength.get(away_team, 0.50)
        
        # 홈 코트 어드밴티지 (약 3%)
        home_advantage = 0.03
        
        # 승률 계산
        total_strength = home_strength + away_strength
        home_prob = (home_strength + home_advantage) / (total_strength + home_advantage)
        away_prob = 1 - home_prob
        
        # 무승부 확률 (NBA는 무승부가 거의 없음, 약 0.5%)
        draw_prob = 0.005
        home_prob = home_prob * (1 - draw_prob)
        away_prob = away_prob * (1 - draw_prob)
        
        return {
            'home_prob': home_prob,
            'draw_prob': draw_prob,
            'away_prob': away_prob
        }
    
    def analyze_betman_match(self, match_data: Dict, 
                            predictions: Dict = None) -> Dict:
        """
        배트맨 경기 분석
        
        Args:
            match_data: 경기 배당률 데이터
            predictions: 모델 예측 (선택사항)
        
        Returns:
            Dict: 분석 결과
        """
        home_team = match_data.get('home_team', '')
        away_team = match_data.get('away_team', '')
        
        # 예측이 없으면 계산
        if predictions is None:
            predictions = self.predict_game_outcome(home_team, away_team)
        
        # 배당률 분석
        analysis = self.analyzer.analyze_match(match_data)
        
        # 예측 확률 추가
        analysis['model_home_prob'] = predictions['home_prob'] * 100
        analysis['model_draw_prob'] = predictions['draw_prob'] * 100
        analysis['model_away_prob'] = predictions['away_prob'] * 100
        
        # 엣지 계산
        analysis['home_edge'] = (predictions['home_prob'] - 
                                analysis['no_vig_home'] / 100) * 100
        analysis['draw_edge'] = (predictions['draw_prob'] - 
                                analysis['no_vig_draw'] / 100) * 100
        analysis['away_edge'] = (predictions['away_prob'] - 
                                analysis['no_vig_away'] / 100) * 100
        
        # 최고 엣지 결과
        edges = {
            'home': analysis['home_edge'],
            'draw': analysis['draw_edge'],
            'away': analysis['away_edge']
        }
        best_outcome = max(edges, key=edges.get)
        analysis['best_bet'] = best_outcome
        analysis['best_edge'] = edges[best_outcome]
        
        return analysis
    
    def find_best_opportunities(self, min_edge: float = 3.0, 
                               max_opportunities: int = 10) -> pd.DataFrame:
        """
        최고의 베팅 기회 식별
        
        Args:
            min_edge: 최소 엣지 (%)
            max_opportunities: 반환할 최대 기회 수
        
        Returns:
            pd.DataFrame: 추천 베팅 목록
        """
        # 배트맨에서 배당률 수집
        matches_df = self.fetch_betman_odds()
        
        if matches_df.empty:
            logger.warning("분석할 경기가 없습니다.")
            return pd.DataFrame()
        
        opportunities = []
        
        for idx, match in matches_df.iterrows():
            # 경기 분석
            analysis = self.analyze_betman_match(match)
            
            # 각 결과별 엣지 확인
            for outcome in ['home', 'draw', 'away']:
                edge_key = f'{outcome}_edge'
                odds_key = f'{outcome}_odds'
                
                if edge_key in analysis and analysis[edge_key] >= min_edge:
                    opp = {
                        'match_id': analysis['match_id'],
                        'home_team': analysis['home_team'],
                        'away_team': analysis['away_team'],
                        'bet_type': outcome,
                        'odds': analysis[odds_key],
                        'model_prob': analysis[f'model_{outcome}_prob'],
                        'no_vig_prob': analysis[f'no_vig_{outcome}'],
                        'edge': analysis[edge_key],
                        'vig': analysis['vig'],
                        'kelly_size': self._calculate_kelly(
                            analysis[f'model_{outcome}_prob'] / 100,
                            analysis[odds_key]
                        ),
                        'expected_value': self._calculate_ev(
                            analysis[f'model_{outcome}_prob'] / 100,
                            analysis[odds_key]
                        )
                    }
                    opportunities.append(opp)
        
        if not opportunities:
            logger.warning(f"최소 엣지 {min_edge}% 이상인 기회가 없습니다.")
            return pd.DataFrame()
        
        df = pd.DataFrame(opportunities)
        
        # 엣지로 정렬
        df = df.sort_values('edge', ascending=False)
        
        # 상위 N개만 반환
        return df.head(max_opportunities)
    
    def _calculate_kelly(self, prob: float, odds: float, 
                        fraction: float = 0.25) -> float:
        """
        켈리 기준 베팅 사이즈 계산
        
        Args:
            prob: 승률
            odds: 소수점 배당률
            fraction: 분수 켈리 (0.25 = 1/4 켈리)
        
        Returns:
            kelly_size: 베팅 사이즈 (%)
        """
        if odds <= 1:
            return 0
        
        b = odds - 1
        q = 1 - prob
        
        kelly = (prob * b - q) / b
        kelly = max(0, kelly)
        
        return kelly * fraction * 100
    
    def _calculate_ev(self, prob: float, odds: float) -> float:
        """
        기대값 계산
        
        Args:
            prob: 승률
            odds: 소수점 배당률
        
        Returns:
            ev: 기대값 (%)
        """
        if odds <= 1:
            return 0
        
        ev = (prob * (odds - 1)) - (1 - prob)
        return ev * 100
    
    def get_betman_recommendations(self, min_edge: float = 3.0) -> pd.DataFrame:
        """
        배트맨 기반 베팅 추천 (메인 인터페이스)
        
        Args:
            min_edge: 최소 엣지 (%)
        
        Returns:
            pd.DataFrame: 추천 베팅 목록
        """
        logger.info("=" * 70)
        logger.info("배트맨 NBA 승부식 베팅 추천")
        logger.info("=" * 70)
        
        recommendations = self.find_best_opportunities(min_edge=min_edge)
        
        if recommendations.empty:
            logger.info("추천할 베팅이 없습니다.")
            return recommendations
        
        logger.info(f"\n✓ {len(recommendations)}개의 베팅 기회 발견")
        logger.info("\n" + "=" * 70)
        logger.info("추천 베팅 목록 (엣지 순)")
        logger.info("=" * 70)
        
        for idx, rec in recommendations.iterrows():
            logger.info(f"\n#{idx+1}. {rec['away_team']} @ {rec['home_team']}")
            logger.info(f"   경기 ID: {rec['match_id']}")
            logger.info(f"   베팅 유형: {rec['bet_type'].upper()}")
            logger.info(f"   배당률: {rec['odds']:.2f}")
            logger.info(f"   모델 확률: {rec['model_prob']:.1f}%")
            logger.info(f"   시장 확률 (No-Vig): {rec['no_vig_prob']:.1f}%")
            logger.info(f"   엣지: +{rec['edge']:.1f}%")
            logger.info(f"   기대값: +{rec['expected_value']:.1f}%")
            logger.info(f"   켈리 사이즈: {rec['kelly_size']:.2f}% of bankroll")
        
        logger.info("\n" + "=" * 70)
        
        return recommendations


def main():
    """메인 실행 함수"""
    finder = BetmanIntegratedEdgeFinder(initial_bankroll=1000.0)
    
    # 추천 베팅 조회
    recommendations = finder.get_betman_recommendations(min_edge=3.0)
    
    if not recommendations.empty:
        print("\n" + "=" * 70)
        print("추천 베팅 요약")
        print("=" * 70)
        print(recommendations[['home_team', 'away_team', 'bet_type', 'odds', 
                              'edge', 'kelly_size']].to_string(index=False))


if __name__ == "__main__":
    main()
