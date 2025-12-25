"""
배트맨(Betman) 사이트 NBA 승부식 데이터 수집 모듈

배트맨 사이트의 NBA 승부식 배당률을 실시간으로 수집하고,
고도화된 베팅 분석 시스템과 통합하기 위한 모듈입니다.
"""

import requests
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BetmanNBAScraper:
    """
    배트맨 사이트에서 NBA 승부식 데이터를 수집하는 클래스
    
    배트맨은 한국의 대표적인 스포츠 베팅 사이트로,
    NBA 경기의 승부식(승/무/패) 배당률을 제공합니다.
    """
    
    def __init__(self, headless: bool = True):
        """
        초기화
        
        Args:
            headless: 브라우저를 숨김 모드로 실행할지 여부
        """
        self.base_url = "https://www.betman.co.kr"
        self.nba_url = "https://www.betman.co.kr/main/mainPage/gamebuy/gameSlip.do?gmId=G101&gmTs=250048"
        self.headless = headless
        self.driver = None
        self.session = requests.Session()
        
        # User-Agent 설정
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def _init_driver(self):
        """Selenium WebDriver 초기화"""
        options = Options()
        if self.headless:
            options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        self.driver = webdriver.Chrome(options=options)
    
    def _close_driver(self):
        """WebDriver 종료"""
        if self.driver:
            self.driver.quit()
            self.driver = None
    
    def scrape_nba_odds(self, timeout: int = 10) -> pd.DataFrame:
        """
        배트맨 사이트에서 NBA 승부식 배당률 수집
        
        Args:
            timeout: 페이지 로드 대기 시간 (초)
        
        Returns:
            pd.DataFrame: 수집된 경기 데이터
                - match_id: 경기 ID
                - home_team: 홈팀
                - away_team: 원정팀
                - home_odds: 홈팀 승리 배당률
                - draw_odds: 무승부 배당률
                - away_odds: 원정팀 승리 배당률
                - fetch_time: 수집 시간
        """
        try:
            self._init_driver()
            logger.info(f"배트맨 NBA 페이지 로드 중: {self.nba_url}")
            
            self.driver.get(self.nba_url)
            
            # 페이지 로드 대기
            try:
                WebDriverWait(self.driver, timeout).until(
                    EC.presence_of_all_elements_located((By.CSS_SELECTOR, "tr[data-matchseq]"))
                )
            except:
                logger.warning("페이지 로드 타임아웃, 현재 상태로 진행합니다.")
            
            time.sleep(2)  # 추가 대기
            
            # BeautifulSoup으로 파싱
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            matches = []
            rows = soup.select('tr[data-matchseq]')
            
            logger.info(f"발견된 경기 수: {len(rows)}")
            
            for row in rows:
                try:
                    match_data = self._parse_match_row(row)
                    if match_data:
                        matches.append(match_data)
                except Exception as e:
                    logger.warning(f"경기 파싱 오류: {e}")
                    continue
            
            if not matches:
                logger.warning("수집된 경기가 없습니다.")
                return pd.DataFrame()
            
            df = pd.DataFrame(matches)
            logger.info(f"✓ {len(df)}개 경기 데이터 수집 완료")
            
            return df
        
        except Exception as e:
            logger.error(f"스크래핑 오류: {e}")
            return pd.DataFrame()
        
        finally:
            self._close_driver()
    
    def _parse_match_row(self, row) -> Optional[Dict]:
        """
        경기 행(row)을 파싱하여 데이터 추출
        
        Args:
            row: BeautifulSoup row element
        
        Returns:
            Dict: 경기 데이터 또는 None
        """
        try:
            # 경기 ID
            match_id = row.get("data-matchseq", "")
            if not match_id:
                return None
            
            # 팀 이름 추출
            teams = row.select('div.scoreDiv span')
            if len(teams) < 3:
                return None
            
            home_team = teams[0].text.strip()
            away_team = teams[2].text.strip()
            
            # 배당률 추출
            buttons = row.select('div.btnChkBox button')
            if len(buttons) < 3:
                return None
            
            try:
                home_odds = float(buttons[0].select_one('span.db').text.strip())
                draw_odds = float(buttons[1].select_one('span.db').text.strip())
                away_odds = float(buttons[2].select_one('span.db').text.strip())
            except (ValueError, AttributeError):
                return None
            
            # 경기 시간 추출 (있으면)
            time_elem = row.select_one('span.time')
            match_time = time_elem.text.strip() if time_elem else ""
            
            return {
                'match_id': match_id,
                'home_team': home_team,
                'away_team': away_team,
                'home_odds': home_odds,
                'draw_odds': draw_odds,
                'away_odds': away_odds,
                'match_time': match_time,
                'fetch_time': datetime.now()
            }
        
        except Exception as e:
            logger.debug(f"행 파싱 오류: {e}")
            return None
    
    def get_nba_games_with_api(self) -> pd.DataFrame:
        """
        API 엔드포인트를 통해 데이터 수집 (Flask 서버 필요)
        
        Returns:
            pd.DataFrame: 경기 데이터
        """
        try:
            # 로컬 Flask 서버에서 데이터 수집
            response = requests.get('http://localhost:5000/odds', timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                matches = []
                
                for match in data:
                    matches.append({
                        'match_id': match.get('경기번호', ''),
                        'home_team': match.get('홈팀', ''),
                        'away_team': match.get('원정팀', ''),
                        'home_odds': match.get('배당', {}).get('승', 0),
                        'draw_odds': match.get('배당', {}).get('무', 0),
                        'away_odds': match.get('배당', {}).get('패', 0),
                        'fetch_time': datetime.now()
                    })
                
                df = pd.DataFrame(matches)
                logger.info(f"✓ API를 통해 {len(df)}개 경기 데이터 수집 완료")
                return df
            else:
                logger.error(f"API 응답 오류: {response.status_code}")
                return pd.DataFrame()
        
        except Exception as e:
            logger.error(f"API 수집 오류: {e}")
            return pd.DataFrame()


class BetmanOddsAnalyzer:
    """
    배트맨 배당률 분석 클래스
    
    배트맨의 배당률을 분석하여 No-Vig 확률을 계산하고,
    베팅 기회를 식별합니다.
    """
    
    @staticmethod
    def decimal_to_implied_prob(odds: float) -> float:
        """
        배당률을 내재 확률로 변환
        
        배트맨의 배당률은 소수점 형식입니다.
        예: 1.95 = 1/1.95 ≈ 51.3%
        
        Args:
            odds: 소수점 배당률
        
        Returns:
            implied_prob: 내재 확률 (0~1)
        """
        if odds <= 0:
            return 0
        return 1 / odds
    
    @staticmethod
    def remove_vig_three_way(home_odds: float, draw_odds: float, 
                            away_odds: float) -> Tuple[float, float, float]:
        """
        3방향 시장(승/무/패)에서 북메이커 마진(Vig) 제거
        
        Args:
            home_odds: 홈팀 승리 배당률
            draw_odds: 무승부 배당률
            away_odds: 원정팀 승리 배당률
        
        Returns:
            (home_prob, draw_prob, away_prob): Vig 제거된 확률
        """
        # 내재 확률 계산
        home_prob = BetmanOddsAnalyzer.decimal_to_implied_prob(home_odds)
        draw_prob = BetmanOddsAnalyzer.decimal_to_implied_prob(draw_odds)
        away_prob = BetmanOddsAnalyzer.decimal_to_implied_prob(away_odds)
        
        # 전체 확률 합계 (Vig 포함)
        total_prob = home_prob + draw_prob + away_prob
        
        # Vig 제거
        no_vig_home = home_prob / total_prob
        no_vig_draw = draw_prob / total_prob
        no_vig_away = away_prob / total_prob
        
        return no_vig_home, no_vig_draw, no_vig_away
    
    @staticmethod
    def calculate_vig(home_odds: float, draw_odds: float, 
                     away_odds: float) -> float:
        """
        북메이커 마진(Vig) 계산
        
        Args:
            home_odds: 홈팀 배당률
            draw_odds: 무승부 배당률
            away_odds: 원정팀 배당률
        
        Returns:
            vig: 북메이커 마진 (%)
        """
        home_prob = BetmanOddsAnalyzer.decimal_to_implied_prob(home_odds)
        draw_prob = BetmanOddsAnalyzer.decimal_to_implied_prob(draw_odds)
        away_prob = BetmanOddsAnalyzer.decimal_to_implied_prob(away_odds)
        
        total_prob = home_prob + draw_prob + away_prob
        vig = (total_prob - 1.0) * 100
        
        return max(0, vig)
    
    @staticmethod
    def analyze_match(match_data: Dict, model_home_prob: float = None,
                     model_draw_prob: float = None,
                     model_away_prob: float = None) -> Dict:
        """
        경기 배당률 분석
        
        Args:
            match_data: 경기 데이터 (배당률 포함)
            model_home_prob: 모델의 홈팀 승률 예측 (선택사항)
            model_draw_prob: 모델의 무승부 확률 예측 (선택사항)
            model_away_prob: 모델의 원정팀 승률 예측 (선택사항)
        
        Returns:
            Dict: 분석 결과
        """
        home_odds = match_data.get('home_odds', 0)
        draw_odds = match_data.get('draw_odds', 0)
        away_odds = match_data.get('away_odds', 0)
        
        # No-Vig 확률 계산
        no_vig_home, no_vig_draw, no_vig_away = \
            BetmanOddsAnalyzer.remove_vig_three_way(home_odds, draw_odds, away_odds)
        
        # 북메이커 마진
        vig = BetmanOddsAnalyzer.calculate_vig(home_odds, draw_odds, away_odds)
        
        analysis = {
            'match_id': match_data.get('match_id', ''),
            'home_team': match_data.get('home_team', ''),
            'away_team': match_data.get('away_team', ''),
            'home_odds': home_odds,
            'draw_odds': draw_odds,
            'away_odds': away_odds,
            'no_vig_home': no_vig_home * 100,
            'no_vig_draw': no_vig_draw * 100,
            'no_vig_away': no_vig_away * 100,
            'vig': vig
        }
        
        # 모델 예측이 있으면 엣지 계산
        if model_home_prob is not None:
            analysis['model_home_prob'] = model_home_prob * 100
            analysis['home_edge'] = (model_home_prob - no_vig_home) * 100
        
        if model_draw_prob is not None:
            analysis['model_draw_prob'] = model_draw_prob * 100
            analysis['draw_edge'] = (model_draw_prob - no_vig_draw) * 100
        
        if model_away_prob is not None:
            analysis['model_away_prob'] = model_away_prob * 100
            analysis['away_edge'] = (model_away_prob - no_vig_away) * 100
        
        return analysis
    
    @staticmethod
    def find_best_bets(matches_df: pd.DataFrame, 
                      predictions_df: pd.DataFrame = None,
                      min_edge: float = 3.0) -> pd.DataFrame:
        """
        배팅 기회 식별
        
        Args:
            matches_df: 경기 배당률 데이터
            predictions_df: 모델 예측 데이터 (선택사항)
            min_edge: 최소 엣지 (%)
        
        Returns:
            pd.DataFrame: 추천 베팅 목록
        """
        opportunities = []
        
        for idx, match in matches_df.iterrows():
            # 배당률 분석
            analysis = BetmanOddsAnalyzer.analyze_match(match)
            
            # 모델 예측 추가 (있으면)
            if predictions_df is not None:
                pred = predictions_df[
                    (predictions_df['home_team'] == match['home_team']) &
                    (predictions_df['away_team'] == match['away_team'])
                ]
                
                if not pred.empty:
                    pred_row = pred.iloc[0]
                    analysis['model_home_prob'] = pred_row.get('home_prob', 0) * 100
                    analysis['home_edge'] = (pred_row.get('home_prob', 0) - 
                                            analysis['no_vig_home'] / 100) * 100
            
            # 엣지가 최소값 이상인 경우만 추가
            for outcome in ['home', 'draw', 'away']:
                edge_key = f'{outcome}_edge'
                if edge_key in analysis and analysis[edge_key] >= min_edge:
                    opp = {
                        'match_id': analysis['match_id'],
                        'home_team': analysis['home_team'],
                        'away_team': analysis['away_team'],
                        'bet_type': outcome,
                        'odds': analysis[f'{outcome}_odds'],
                        'no_vig_prob': analysis[f'no_vig_{outcome}'],
                        'model_prob': analysis.get(f'model_{outcome}_prob', None),
                        'edge': analysis[edge_key],
                        'vig': analysis['vig']
                    }
                    opportunities.append(opp)
        
        if not opportunities:
            return pd.DataFrame()
        
        df = pd.DataFrame(opportunities)
        
        # 엣지로 정렬
        df = df.sort_values('edge', ascending=False)
        
        return df


def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("배트맨 NBA 승부식 데이터 수집 및 분석")
    print("=" * 70)
    
    # 데이터 수집
    scraper = BetmanNBAScraper(headless=True)
    
    print("\n📊 배트맨 사이트에서 NBA 경기 데이터 수집 중...")
    matches_df = scraper.scrape_nba_odds()
    
    if matches_df.empty:
        print("✗ 수집된 데이터가 없습니다.")
        return
    
    print(f"\n✓ {len(matches_df)}개 경기 수집 완료")
    print("\n" + "=" * 70)
    print("수집된 경기 데이터")
    print("=" * 70)
    
    for idx, match in matches_df.iterrows():
        print(f"\n경기 ID: {match['match_id']}")
        print(f"  {match['away_team']} @ {match['home_team']}")
        print(f"  배당률: 홈 {match['home_odds']:.2f} | 무 {match['draw_odds']:.2f} | 원정 {match['away_odds']:.2f}")
        
        # 배당률 분석
        analysis = BetmanOddsAnalyzer.analyze_match(match)
        print(f"  No-Vig 확률: 홈 {analysis['no_vig_home']:.1f}% | 무 {analysis['no_vig_draw']:.1f}% | 원정 {analysis['no_vig_away']:.1f}%")
        print(f"  북메이커 마진: {analysis['vig']:.1f}%")
    
    print("\n" + "=" * 70)
    print("✓ 분석 완료")
    print("=" * 70)


if __name__ == "__main__":
    main()
