"""
Advanced Probability Engine for Sports Betting
- No-Vig 확률 계산 (공통 전제 레이어)
- 확률 보정 (Calibration)
- 게임 유형별 예측 로직
- 성과 추적 시스템
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json


class BetType(Enum):
    """베팅 유형 분류"""
    MONEYLINE = "moneyline"  # 승패
    SPREAD = "spread"  # 핸디캡
    TOTAL = "total"  # 언더/오버
    ODD_EVEN = "odd_even"  # 홀/짝


@dataclass
class MarketOdds:
    """시장 배당률 정보"""
    bet_type: BetType
    odds: float  # American odds
    bookmaker: str
    market_name: str
    timestamp: datetime = None


@dataclass
class PredictionResult:
    """예측 결과"""
    game_id: str
    bet_type: BetType
    our_probability: float  # 우리 모델 확률
    market_probability: float  # 시장 확률 (no-vig)
    calibrated_probability: float  # 보정된 확률
    edge: float  # 엣지 (%)
    expected_value: float  # 기대값 (%)
    kelly_size: float  # 켈리 기준 베팅 사이즈 (%)
    confidence: str  # 신뢰도
    recommendation: str  # 베팅 권장사항


class NoVigCalculator:
    """
    No-Vig 확률 계산 표준화
    
    모든 베팅 시장에 대해 북메이커 마진(vig)을 제거한 
    공정한 확률을 계산하는 공통 함수
    """
    
    @staticmethod
    def american_to_decimal(odds: float) -> float:
        """미국식 배당률을 소수점 배당률로 변환"""
        if odds > 0:
            return (odds / 100) + 1
        else:
            return (100 / abs(odds)) + 1
    
    @staticmethod
    def american_to_implied_prob(odds: float) -> float:
        """미국식 배당률을 내재 확률로 변환 (vig 포함)"""
        if odds > 0:
            return 100 / (odds + 100)
        else:
            return abs(odds) / (abs(odds) + 100)
    
    @staticmethod
    def decimal_to_implied_prob(decimal_odds: float) -> float:
        """소수점 배당률을 내재 확률로 변환"""
        return 1 / decimal_odds
    
    @staticmethod
    def calculate_vig(odds_list: List[float]) -> float:
        """
        여러 배당률로부터 북메이커 마진(vig) 계산
        
        Args:
            odds_list: 모든 가능한 결과의 배당률 리스트
        
        Returns:
            vig: 북메이커 마진 (0~1 범위)
        """
        implied_probs = [NoVigCalculator.american_to_implied_prob(o) for o in odds_list]
        total_prob = sum(implied_probs)
        vig = total_prob - 1.0
        return max(0, vig)
    
    @staticmethod
    def remove_vig_two_way(odds1: float, odds2: float) -> Tuple[float, float]:
        """
        양방향 시장(Moneyline, Spread)에서 vig 제거
        
        Args:
            odds1: 첫 번째 결과의 배당률
            odds2: 두 번째 결과의 배당률
        
        Returns:
            (no_vig_prob1, no_vig_prob2): vig 제거된 확률
        """
        prob1 = NoVigCalculator.american_to_implied_prob(odds1)
        prob2 = NoVigCalculator.american_to_implied_prob(odds2)
        
        total = prob1 + prob2
        
        # vig 제거
        no_vig_prob1 = prob1 / total
        no_vig_prob2 = prob2 / total
        
        return no_vig_prob1, no_vig_prob2
    
    @staticmethod
    def remove_vig_three_way(odds_list: List[float]) -> List[float]:
        """
        3방향 시장(Draw 포함)에서 vig 제거
        
        Args:
            odds_list: [odds_team1, odds_draw, odds_team2]
        
        Returns:
            no_vig_probs: vig 제거된 확률 리스트
        """
        implied_probs = [NoVigCalculator.american_to_implied_prob(o) for o in odds_list]
        total = sum(implied_probs)
        no_vig_probs = [p / total for p in implied_probs]
        return no_vig_probs
    
    @staticmethod
    def remove_vig_total_market(over_odds: float, under_odds: float) -> Tuple[float, float]:
        """
        Total(Over/Under) 시장에서 vig 제거
        
        Args:
            over_odds: Over 배당률
            under_odds: Under 배당률
        
        Returns:
            (no_vig_over_prob, no_vig_under_prob): vig 제거된 확률
        """
        return NoVigCalculator.remove_vig_two_way(over_odds, under_odds)


class ProbabilityCalibrator:
    """
    확률 보정(Calibration) 레이어
    
    원시 예측 확률을 개선하여 과도한 베팅을 억제하고
    켈리 기준의 안정성을 높입니다.
    """
    
    def __init__(self):
        self.calibration_history = []
        self.brier_scores = []
        self.log_losses = []
    
    def brier_score(self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray) -> float:
        """
        Brier Score 계산
        
        낮을수록 좋음 (0 = 완벽, 1 = 최악)
        """
        return np.mean((predicted_probs - actual_outcomes) ** 2)
    
    def log_loss(self, predicted_probs: np.ndarray, actual_outcomes: np.ndarray) -> float:
        """
        Log Loss 계산
        
        낮을수록 좋음. 극단적인 확률 오류에 더 강하게 페널티
        """
        epsilon = 1e-15
        predicted_probs = np.clip(predicted_probs, epsilon, 1 - epsilon)
        return -np.mean(actual_outcomes * np.log(predicted_probs) + 
                       (1 - actual_outcomes) * np.log(1 - predicted_probs))
    
    def isotonic_calibration(self, predicted_probs: np.ndarray, 
                            actual_outcomes: np.ndarray) -> callable:
        """
        Isotonic Regression을 이용한 확률 보정
        
        예측 확률을 실제 결과와 일치하도록 변환하는 함수 반환
        """
        from sklearn.isotonic import IsotonicRegression
        
        iso_reg = IsotonicRegression(out_of_bounds='clip')
        iso_reg.fit(predicted_probs, actual_outcomes)
        
        return iso_reg.predict
    
    def platt_scaling(self, predicted_probs: np.ndarray, 
                     actual_outcomes: np.ndarray) -> Tuple[float, float]:
        """
        Platt Scaling을 이용한 확률 보정
        
        P(y=1|x) = 1 / (1 + exp(A*f(x) + B))
        
        Returns:
            (A, B): Platt scaling 파라미터
        """
        from scipy.optimize import minimize
        
        def log_likelihood(params):
            A, B = params
            predictions = 1 / (1 + np.exp(A * predicted_probs + B))
            ll = -np.mean(actual_outcomes * np.log(predictions + 1e-15) + 
                         (1 - actual_outcomes) * np.log(1 - predictions + 1e-15))
            return ll
        
        result = minimize(log_likelihood, [1, 0], method='BFGS')
        return result.x
    
    def apply_platt_scaling(self, prob: float, A: float, B: float) -> float:
        """Platt scaling 적용"""
        return 1 / (1 + np.exp(A * prob + B))
    
    def temperature_scaling(self, predicted_probs: np.ndarray, 
                           actual_outcomes: np.ndarray) -> float:
        """
        Temperature Scaling을 이용한 확률 보정
        
        P_calibrated = softmax(logits / T)
        
        Returns:
            T: Temperature 파라미터
        """
        from scipy.optimize import minimize
        
        def nll(T):
            if T <= 0:
                return 1e10
            scaled_probs = np.clip(predicted_probs ** (1/T), 1e-15, 1-1e-15)
            return -np.mean(actual_outcomes * np.log(scaled_probs) + 
                           (1 - actual_outcomes) * np.log(1 - scaled_probs))
        
        result = minimize(nll, [1.0], bounds=[(0.1, 5.0)], method='L-BFGS-B')
        return result.x[0]
    
    def apply_temperature_scaling(self, prob: float, T: float) -> float:
        """Temperature scaling 적용"""
        return prob ** (1/T) / (prob ** (1/T) + (1-prob) ** (1/T))
    
    def calibrate_probability(self, raw_prob: float, 
                             calibration_method: str = 'temperature',
                             calibration_params: Dict = None) -> float:
        """
        원시 확률을 보정된 확률로 변환
        
        Args:
            raw_prob: 원시 예측 확률 (0~1)
            calibration_method: 'temperature', 'platt', 'isotonic'
            calibration_params: 보정 파라미터 딕셔너리
        
        Returns:
            calibrated_prob: 보정된 확률
        """
        if calibration_params is None:
            return raw_prob
        
        if calibration_method == 'temperature':
            T = calibration_params.get('temperature', 1.0)
            return self.apply_temperature_scaling(raw_prob, T)
        
        elif calibration_method == 'platt':
            A = calibration_params.get('A', 1.0)
            B = calibration_params.get('B', 0.0)
            return self.apply_platt_scaling(raw_prob, A, B)
        
        else:
            return raw_prob


class BettingMarketAnalyzer:
    """
    게임 유형별 베팅 시장 분석
    
    각 베팅 유형에 최적화된 확률 모델링과 엣지 계산
    """
    
    def __init__(self):
        self.no_vig_calc = NoVigCalculator()
        self.calibrator = ProbabilityCalibrator()
    
    def analyze_moneyline(self, home_prob: float, away_prob: float,
                         home_odds: float, away_odds: float,
                         calibration_params: Dict = None) -> PredictionResult:
        """
        Moneyline(승패) 시장 분석
        
        Args:
            home_prob: 우리 모델의 홈팀 승률
            away_prob: 우리 모델의 어웨이팀 승률
            home_odds: 홈팀 배당률
            away_odds: 어웨이팀 배당률
            calibration_params: 확률 보정 파라미터
        
        Returns:
            PredictionResult: 분석 결과
        """
        # No-Vig 확률 계산
        market_home_prob, market_away_prob = self.no_vig_calc.remove_vig_two_way(
            home_odds, away_odds
        )
        
        # 확률 보정
        calibrated_home_prob = self.calibrator.calibrate_probability(
            home_prob, calibration_params=calibration_params
        )
        
        # 엣지 계산
        edge = (calibrated_home_prob - market_home_prob) * 100
        
        # 기대값 계산
        decimal_odds = self.no_vig_calc.american_to_decimal(home_odds)
        ev = (calibrated_home_prob * (decimal_odds - 1)) - (1 - calibrated_home_prob)
        ev_pct = ev * 100
        
        # 켈리 기준 베팅 사이즈
        kelly_size = self._kelly_criterion(calibrated_home_prob, home_odds) * 100
        
        # 신뢰도 판정
        confidence = self._get_confidence(edge)
        
        return PredictionResult(
            game_id="",
            bet_type=BetType.MONEYLINE,
            our_probability=home_prob * 100,
            market_probability=market_home_prob * 100,
            calibrated_probability=calibrated_home_prob * 100,
            edge=edge,
            expected_value=ev_pct,
            kelly_size=kelly_size,
            confidence=confidence,
            recommendation=f"{'BET' if edge > 3 else 'PASS'} - {edge:+.1f}% edge"
        )
    
    def analyze_spread(self, team_prob: float, spread: float, spread_odds: float,
                      calibration_params: Dict = None) -> PredictionResult:
        """
        Spread(핸디캡) 시장 분석
        
        Args:
            team_prob: 팀의 승률 (스프레드 조정 전)
            spread: 스프레드 포인트 (음수 = 선호팀)
            spread_odds: 스프레드 배당률
            calibration_params: 확률 보정 파라미터
        
        Returns:
            PredictionResult: 분석 결과
        """
        # 스프레드 조정된 확률 계산
        # 스프레드가 음수면 선호팀이므로 승률 상향 조정
        spread_adjusted_prob = self._adjust_prob_for_spread(team_prob, spread)
        
        # No-Vig 확률 (스프레드는 보통 -110 양쪽)
        market_prob, _ = self.no_vig_calc.remove_vig_two_way(spread_odds, spread_odds)
        
        # 확률 보정
        calibrated_prob = self.calibrator.calibrate_probability(
            spread_adjusted_prob, calibration_params=calibration_params
        )
        
        # 엣지 계산
        edge = (calibrated_prob - market_prob) * 100
        
        # 기대값 계산
        decimal_odds = self.no_vig_calc.american_to_decimal(spread_odds)
        ev = (calibrated_prob * (decimal_odds - 1)) - (1 - calibrated_prob)
        ev_pct = ev * 100
        
        # 켈리 기준
        kelly_size = self._kelly_criterion(calibrated_prob, spread_odds) * 100
        
        confidence = self._get_confidence(edge)
        
        return PredictionResult(
            game_id="",
            bet_type=BetType.SPREAD,
            our_probability=team_prob * 100,
            market_probability=market_prob * 100,
            calibrated_probability=calibrated_prob * 100,
            edge=edge,
            expected_value=ev_pct,
            kelly_size=kelly_size,
            confidence=confidence,
            recommendation=f"{'BET' if edge > 2 else 'PASS'} - {edge:+.1f}% edge"
        )
    
    def analyze_total(self, over_prob: float, over_odds: float, under_odds: float,
                     calibration_params: Dict = None) -> PredictionResult:
        """
        Total(Over/Under) 시장 분석
        
        Args:
            over_prob: Over 확률
            over_odds: Over 배당률
            under_odds: Under 배당률
            calibration_params: 확률 보정 파라미터
        
        Returns:
            PredictionResult: 분석 결과
        """
        # No-Vig 확률
        market_over_prob, market_under_prob = self.no_vig_calc.remove_vig_total_market(
            over_odds, under_odds
        )
        
        # 확률 보정
        calibrated_over_prob = self.calibrator.calibrate_probability(
            over_prob, calibration_params=calibration_params
        )
        
        # 엣지 계산
        edge = (calibrated_over_prob - market_over_prob) * 100
        
        # 기대값 계산
        decimal_odds = self.no_vig_calc.american_to_decimal(over_odds)
        ev = (calibrated_over_prob * (decimal_odds - 1)) - (1 - calibrated_over_prob)
        ev_pct = ev * 100
        
        # 켈리 기준
        kelly_size = self._kelly_criterion(calibrated_over_prob, over_odds) * 100
        
        confidence = self._get_confidence(edge)
        
        return PredictionResult(
            game_id="",
            bet_type=BetType.TOTAL,
            our_probability=over_prob * 100,
            market_probability=market_over_prob * 100,
            calibrated_probability=calibrated_over_prob * 100,
            edge=edge,
            expected_value=ev_pct,
            kelly_size=kelly_size,
            confidence=confidence,
            recommendation=f"{'BET' if edge > 2.5 else 'PASS'} - {edge:+.1f}% edge"
        )
    
    def analyze_odd_even(self, odd_prob: float, odd_odds: float, even_odds: float,
                        calibration_params: Dict = None) -> PredictionResult:
        """
        Odd/Even 시장 분석
        
        Args:
            odd_prob: 홀수 확률
            odd_odds: 홀수 배당률
            even_odds: 짝수 배당률
            calibration_params: 확률 보정 파라미터
        
        Returns:
            PredictionResult: 분석 결과
        """
        # No-Vig 확률
        market_odd_prob, market_even_prob = self.no_vig_calc.remove_vig_two_way(
            odd_odds, even_odds
        )
        
        # 확률 보정
        calibrated_odd_prob = self.calibrator.calibrate_probability(
            odd_prob, calibration_params=calibration_params
        )
        
        # 엣지 계산
        edge = (calibrated_odd_prob - market_odd_prob) * 100
        
        # 기대값 계산
        decimal_odds = self.no_vig_calc.american_to_decimal(odd_odds)
        ev = (calibrated_odd_prob * (decimal_odds - 1)) - (1 - calibrated_odd_prob)
        ev_pct = ev * 100
        
        # 켈리 기준
        kelly_size = self._kelly_criterion(calibrated_odd_prob, odd_odds) * 100
        
        confidence = self._get_confidence(edge)
        
        return PredictionResult(
            game_id="",
            bet_type=BetType.ODD_EVEN,
            our_probability=odd_prob * 100,
            market_probability=market_odd_prob * 100,
            calibrated_probability=calibrated_odd_prob * 100,
            edge=edge,
            expected_value=ev_pct,
            kelly_size=kelly_size,
            confidence=confidence,
            recommendation=f"{'BET' if edge > 2 else 'PASS'} - {edge:+.1f}% edge"
        )
    
    def _adjust_prob_for_spread(self, base_prob: float, spread: float) -> float:
        """스프레드 포인트에 따라 확률 조정"""
        # 간단한 선형 조정: spread 1포인트당 약 1% 확률 변화
        adjustment = abs(spread) * 0.01
        if spread < 0:  # 음수 스프레드 = 선호팀
            return min(1.0, base_prob + adjustment)
        else:
            return max(0.0, base_prob - adjustment)
    
    def _kelly_criterion(self, prob: float, odds: float, fraction: float = 0.25) -> float:
        """
        켈리 기준 베팅 사이즈 계산
        
        f* = (bp - q) / b
        여기서 b = decimal_odds - 1, p = 확률, q = 1-p
        
        fraction: 보수적 베팅을 위한 분수 켈리 (0.25 = 1/4 켈리)
        """
        decimal_odds = self.no_vig_calc.american_to_decimal(odds)
        b = decimal_odds - 1
        q = 1 - prob
        
        kelly = (prob * b - q) / b
        
        # 음수 켈리는 베팅하지 않음
        kelly = max(0, kelly)
        
        # 분수 켈리 적용 (위험 관리)
        return kelly * fraction
    
    def _get_confidence(self, edge: float) -> str:
        """엣지 크기에 따른 신뢰도 판정"""
        if edge >= 10:
            return 'Very High'
        elif edge >= 6:
            return 'High'
        elif edge >= 3:
            return 'Medium'
        elif edge >= 0:
            return 'Low'
        else:
            return 'Negative'


class PerformanceTracker:
    """
    성과 추적 및 자동 결산 시스템
    
    베팅 결과를 자동으로 추적하고 성과를 분석합니다.
    """
    
    def __init__(self, initial_bankroll: float = 1000.0):
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        self.bets = []
        self.results = []
    
    def place_bet(self, bet_id: str, game_id: str, bet_type: BetType,
                 prediction: PredictionResult, bet_amount: float,
                 bet_date: datetime = None) -> Dict:
        """베팅 기록"""
        bet = {
            'bet_id': bet_id,
            'game_id': game_id,
            'bet_type': bet_type.value,
            'prediction': prediction,
            'bet_amount': bet_amount,
            'bet_date': bet_date or datetime.now(),
            'status': 'pending'
        }
        self.bets.append(bet)
        return bet
    
    def record_result(self, bet_id: str, outcome: bool, odds: float) -> Dict:
        """베팅 결과 기록"""
        bet = next((b for b in self.bets if b['bet_id'] == bet_id), None)
        if not bet:
            raise ValueError(f"Bet {bet_id} not found")
        
        decimal_odds = NoVigCalculator.american_to_decimal(odds)
        
        if outcome:
            payout = bet['bet_amount'] * decimal_odds
            profit = payout - bet['bet_amount']
        else:
            payout = 0
            profit = -bet['bet_amount']
        
        result = {
            'bet_id': bet_id,
            'outcome': outcome,
            'payout': payout,
            'profit': profit,
            'result_date': datetime.now()
        }
        
        self.results.append(result)
        self.current_bankroll += profit
        bet['status'] = 'completed'
        
        return result
    
    def get_statistics(self) -> Dict:
        """성과 통계 계산"""
        if not self.results:
            return {
                'total_bets': 0,
                'win_rate': 0,
                'total_profit': 0,
                'roi': 0,
                'avg_odds': 0,
                'max_drawdown': 0
            }
        
        results_df = pd.DataFrame(self.results)
        
        total_bets = len(results_df)
        wins = results_df['outcome'].sum()
        win_rate = wins / total_bets if total_bets > 0 else 0
        
        total_profit = results_df['profit'].sum()
        roi = (total_profit / self.initial_bankroll) * 100 if self.initial_bankroll > 0 else 0
        
        # Max drawdown 계산
        cumulative_profit = results_df['profit'].cumsum()
        running_max = cumulative_profit.expanding().max()
        drawdown = cumulative_profit - running_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0
        
        return {
            'total_bets': total_bets,
            'wins': wins,
            'losses': total_bets - wins,
            'win_rate': win_rate * 100,
            'total_profit': total_profit,
            'roi': roi,
            'current_bankroll': self.current_bankroll,
            'max_drawdown': max_drawdown,
            'avg_profit_per_bet': total_profit / total_bets if total_bets > 0 else 0
        }
    
    def get_results_by_type(self) -> Dict[str, Dict]:
        """베팅 유형별 성과"""
        if not self.results:
            return {}
        
        results_df = pd.DataFrame(self.results)
        bets_df = pd.DataFrame(self.bets)
        
        # Merge results with bet types
        merged = results_df.merge(
            bets_df[['bet_id', 'bet_type']], 
            on='bet_id'
        )
        
        results_by_type = {}
        for bet_type in merged['bet_type'].unique():
            type_results = merged[merged['bet_type'] == bet_type]
            
            wins = type_results['outcome'].sum()
            total = len(type_results)
            
            results_by_type[bet_type] = {
                'total_bets': total,
                'wins': wins,
                'win_rate': (wins / total * 100) if total > 0 else 0,
                'total_profit': type_results['profit'].sum(),
                'roi': (type_results['profit'].sum() / self.initial_bankroll * 100) if self.initial_bankroll > 0 else 0
            }
        
        return results_by_type
    
    def get_equity_curve(self) -> pd.DataFrame:
        """자본 곡선 데이터"""
        if not self.results:
            return pd.DataFrame()
        
        results_df = pd.DataFrame(self.results)
        results_df['cumulative_profit'] = results_df['profit'].cumsum()
        results_df['equity'] = self.initial_bankroll + results_df['cumulative_profit']
        
        return results_df[['result_date', 'profit', 'cumulative_profit', 'equity']]
