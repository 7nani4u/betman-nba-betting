"""
배트맨 NBA 승부식 베팅 추천 대시보드

배트맨 사이트의 실시간 배당률과 우리의 예측 모델을 결합하여
최고의 베팅 기회를 시각적으로 제시하는 Streamlit 대시보드입니다.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import logging
from betman_edge_finder import BetmanIntegratedEdgeFinder

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="배트맨 NBA 승부식 베팅 추천",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
    }
    .bet-card-high {
        background-color: #dcfce7;
        border-left: 4px solid #10b981;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .bet-card-medium {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .bet-card-low {
        background-color: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 15px;
        border-radius: 8px;
        margin-bottom: 10px;
    }
    .recommendation-box {
        background-color: #f3f4f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'edge_finder' not in st.session_state:
    st.session_state.edge_finder = BetmanIntegratedEdgeFinder(initial_bankroll=1000.0)

if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

@st.cache_data(ttl=300)
def get_betman_recommendations(min_edge: float = 3.0, max_opportunities: int = 20):
    """배트맨 추천 베팅 조회 (캐시됨)"""
    finder = st.session_state.edge_finder
    return finder.find_best_opportunities(min_edge=min_edge, 
                                         max_opportunities=max_opportunities)


def get_odds_distribution(recommendations_df: pd.DataFrame):
    """배당률 분포 데이터"""
    if recommendations_df.empty:
        return pd.DataFrame()
    
    return recommendations_df.groupby('bet_type').agg({
        'odds': ['mean', 'min', 'max'],
        'edge': 'mean'
    }).round(2)


def get_edge_distribution(recommendations_df: pd.DataFrame):
    """엣지 분포 데이터"""
    if recommendations_df.empty:
        return pd.DataFrame()
    
    return recommendations_df.groupby(pd.cut(recommendations_df['edge'], 
                                            bins=[0, 3, 6, 10, 100])).size()


def get_team_recommendations(recommendations_df: pd.DataFrame):
    """팀별 추천 베팅 수"""
    if recommendations_df.empty:
        return pd.DataFrame()
    
    home_bets = recommendations_df.groupby('home_team').size().rename('홈 베팅')
    away_bets = recommendations_df.groupby('away_team').size().rename('원정 베팅')
    
    return pd.concat([home_bets, away_bets], axis=1).fillna(0).astype(int)


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Title
    st.title("🏀 배트맨 NBA 승부식 베팅 추천 시스템")
    st.markdown("### 실시간 배당률 분석 + AI 예측 모델 = 최고의 베팅 기회")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ 설정")
        
        min_edge = st.slider(
            "최소 엣지 (%)",
            min_value=0.0,
            max_value=20.0,
            value=3.0,
            step=0.5
        )
        
        max_opportunities = st.slider(
            "표시할 최대 기회 수",
            min_value=5,
            max_value=50,
            value=20,
            step=5
        )
        
        initial_bankroll = st.number_input(
            "초기 자본 ($)",
            min_value=100,
            max_value=100000,
            value=1000,
            step=100
        )
        
        st.markdown("---")
        
        # 데이터 새로고침
        if st.button("🔄 데이터 새로고침"):
            st.cache_data.clear()
            st.session_state.last_update = datetime.now()
            st.success("데이터가 새로고침되었습니다!")
        
        st.markdown("---")
        
        # 마지막 업데이트 시간
        st.info(f"마지막 업데이트: {st.session_state.last_update.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 추천 베팅 조회
    recommendations = get_betman_recommendations(min_edge=min_edge, 
                                                max_opportunities=max_opportunities)
    
    # Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 발견된 기회",
            value=len(recommendations),
            delta="경기" if len(recommendations) > 0 else None
        )
    
    with col2:
        avg_edge = recommendations['edge'].mean() if not recommendations.empty else 0
        st.metric(
            label="📈 평균 엣지",
            value=f"+{avg_edge:.2f}%",
            delta="추천 기준" if avg_edge >= min_edge else "미달"
        )
    
    with col3:
        avg_kelly = recommendations['kelly_size'].mean() if not recommendations.empty else 0
        st.metric(
            label="💰 평균 켈리 사이즈",
            value=f"{avg_kelly:.2f}%",
            delta="of bankroll"
        )
    
    with col4:
        avg_odds = recommendations['odds'].mean() if not recommendations.empty else 0
        st.metric(
            label="🎯 평균 배당률",
            value=f"{avg_odds:.2f}",
            delta="배당"
        )
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 추천 베팅",
        "📊 분석",
        "📈 통계",
        "ℹ️ 가이드"
    ])
    
    # ========================================================================
    # TAB 1: 추천 베팅
    # ========================================================================
    with tab1:
        st.header("🎯 추천 베팅 목록")
        
        if recommendations.empty:
            st.warning(f"최소 엣지 {min_edge}% 이상인 베팅 기회가 없습니다.")
            st.info("엣지 기준을 낮추거나 나중에 다시 시도해주세요.")
        else:
            st.markdown(f"""
            **{len(recommendations)}개의 베팅 기회**를 발견했습니다.
            
            아래 목록에서 가장 유리한 베팅을 선택할 수 있습니다.
            각 베팅은 **엣지(우위)** 순서로 정렬되어 있습니다.
            """)
            
            # 베팅 카드 표시
            for idx, rec in recommendations.iterrows():
                # 엣지에 따른 색상 결정
                if rec['edge'] >= 10:
                    card_class = "bet-card-high"
                    confidence = "🟢 매우 높음"
                elif rec['edge'] >= 6:
                    card_class = "bet-card-medium"
                    confidence = "🟡 높음"
                else:
                    card_class = "bet-card-low"
                    confidence = "🔵 중간"
                
                # 베팅 유형 한글화
                bet_type_kr = {
                    'home': '홈팀 승리',
                    'away': '원정팀 승리',
                    'draw': '무승부'
                }.get(rec['bet_type'], rec['bet_type'])
                
                with st.container():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.markdown(f"""
                        #### #{idx+1}. {rec['away_team']} @ {rec['home_team']}
                        
                        **베팅 유형:** {bet_type_kr}  
                        **배당률:** {rec['odds']:.2f}  
                        **신뢰도:** {confidence}
                        """)
                    
                    with col2:
                        st.metric("엣지", f"+{rec['edge']:.2f}%")
                        st.metric("기대값", f"+{rec['expected_value']:.2f}%")
                    
                    with col3:
                        st.metric("모델 확률", f"{rec['model_prob']:.1f}%")
                        st.metric("시장 확률", f"{rec['no_vig_prob']:.1f}%")
                    
                    # 상세 정보
                    with st.expander("📋 상세 정보"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.write(f"**경기 ID:** {rec['match_id']}")
                            st.write(f"**배당률:** {rec['odds']:.2f}")
                        
                        with col2:
                            st.write(f"**켈리 사이즈:** {rec['kelly_size']:.2f}%")
                            st.write(f"**북메이커 마진:** {rec['vig']:.2f}%")
                        
                        with col3:
                            st.write(f"**모델 확률:** {rec['model_prob']:.1f}%")
                            st.write(f"**시장 확률:** {rec['no_vig_prob']:.1f}%")
                        
                        # 베팅 버튼
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            if st.button(f"✅ 베팅 확인 #{idx+1}", key=f"bet_{idx}"):
                                st.success(f"✓ {rec['away_team']} @ {rec['home_team']} - {bet_type_kr} 베팅이 선택되었습니다!")
                    
                    st.markdown("---")
    
    # ========================================================================
    # TAB 2: 분석
    # ========================================================================
    with tab2:
        st.header("📊 베팅 기회 분석")
        
        if recommendations.empty:
            st.info("분석할 데이터가 없습니다.")
        else:
            # 엣지 분포
            st.subheader("엣지 분포")
            
            fig = go.Figure()
            fig.add_trace(go.Histogram(
                x=recommendations['edge'],
                nbinsx=20,
                marker_color='#3b82f6',
                name='엣지'
            ))
            fig.update_layout(
                template='plotly_white',
                height=400,
                xaxis_title="엣지 (%)",
                yaxis_title="베팅 수",
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 배당률 vs 엣지
            st.subheader("배당률 vs 엣지")
            
            fig = px.scatter(
                recommendations,
                x='odds',
                y='edge',
                color='bet_type',
                size='kelly_size',
                hover_data=['home_team', 'away_team', 'model_prob'],
                title='배당률과 엣지의 관계',
                labels={'odds': '배당률', 'edge': '엣지 (%)'}
            )
            fig.update_layout(
                template='plotly_white',
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 베팅 유형별 통계
            st.subheader("베팅 유형별 통계")
            
            bet_type_stats = recommendations.groupby('bet_type').agg({
                'edge': ['count', 'mean', 'max'],
                'odds': 'mean',
                'kelly_size': 'mean'
            }).round(2)
            
            st.dataframe(bet_type_stats, use_container_width=True)
    
    # ========================================================================
    # TAB 3: 통계
    # ========================================================================
    with tab3:
        st.header("📈 상세 통계")
        
        if recommendations.empty:
            st.info("통계 데이터가 없습니다.")
        else:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("엣지 범위별 분포")
                
                edge_ranges = pd.cut(
                    recommendations['edge'],
                    bins=[0, 3, 6, 10, 100],
                    labels=['3-6%', '6-10%', '10%+', '기타']
                )
                
                fig = px.pie(
                    values=edge_ranges.value_counts(),
                    names=edge_ranges.value_counts().index,
                    title='엣지 범위별 베팅 수'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("베팅 유형별 분포")
                
                fig = px.pie(
                    values=recommendations['bet_type'].value_counts(),
                    names=recommendations['bet_type'].value_counts().index,
                    title='베팅 유형별 분포'
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # 상세 테이블
            st.subheader("전체 추천 베팅 목록")
            
            display_df = recommendations[[
                'home_team', 'away_team', 'bet_type', 'odds', 
                'model_prob', 'no_vig_prob', 'edge', 'kelly_size'
            ]].copy()
            
            display_df.columns = [
                '홈팀', '원정팀', '베팅유형', '배당률',
                '모델확률(%)', '시장확률(%)', '엣지(%)', '켈리(%)'
            ]
            
            st.dataframe(display_df, use_container_width=True)
    
    # ========================================================================
    # TAB 4: 가이드
    # ========================================================================
    with tab4:
        st.header("ℹ️ 사용 가이드")
        
        st.markdown("""
        ## 배트맨 NBA 승부식 베팅 추천 시스템 사용 가이드
        
        ### 📌 시스템 개요
        
        이 시스템은 배트맨 사이트의 실시간 NBA 승부식 배당률과 
        AI 예측 모델을 결합하여 최고의 베팅 기회를 자동으로 식별합니다.
        
        ### 🎯 주요 개념
        
        **엣지 (Edge)**
        - 우리의 예측 확률과 시장의 공정한 확률(No-Vig) 간의 차이
        - 양수 엣지 = 우리가 시장보다 더 정확하게 예측
        - 엣지가 클수록 더 유리한 베팅
        
        **기대값 (Expected Value, EV)**
        - 장기적으로 해당 베팅에서 기대할 수 있는 평균 수익률
        - 양수 EV = 장기적으로 수익이 기대됨
        
        **켈리 기준 (Kelly Criterion)**
        - 최적의 베팅 사이즈를 계산하는 수학 공식
        - 자본을 최대한 효율적으로 사용하면서 파산 위험을 최소화
        - 시스템은 보수적인 1/4 켈리를 사용
        
        **No-Vig 확률**
        - 배당률에서 북메이커의 이익(마진)을 제거한 공정한 확률
        - 시장의 진정한 평가를 반영
        
        ### 💡 사용 팁
        
        1. **최소 엣지 설정**: 일반적으로 3% 이상의 엣지를 추천합니다.
        2. **베팅 금액**: 켈리 사이즈를 참고하여 베팅 금액을 결정하세요.
        3. **다양화**: 여러 경기에 분산 베팅하는 것이 좋습니다.
        4. **장기 관점**: 단기 손실에 흔들리지 않고 장기 수익성을 추구하세요.
        
        ### ⚠️ 주의사항
        
        - 과거 성과는 미래 결과를 보장하지 않습니다.
        - 베팅은 항상 위험을 수반합니다.
        - 여유 자금으로만 베팅하세요.
        - 책임감 있는 베팅을 실천하세요.
        
        ### 📊 시스템 구성
        
        1. **데이터 수집**: 배트맨 사이트에서 실시간 배당률 수집
        2. **No-Vig 계산**: 배당률에서 북메이커 마진 제거
        3. **확률 예측**: AI 모델로 게임 결과 예측
        4. **엣지 분석**: 예측과 시장 확률 비교
        5. **베팅 추천**: 최고의 기회 자동 식별
        
        ### 🔄 데이터 업데이트
        
        - 시스템은 5분마다 자동으로 데이터를 캐시합니다.
        - "데이터 새로고침" 버튼으로 즉시 업데이트할 수 있습니다.
        - 경기 시작 전까지 배당률이 변할 수 있습니다.
        """)


if __name__ == "__main__":
    main()
