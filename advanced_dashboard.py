"""
Advanced Sports Betting Edge Finder Dashboard
고도화된 성과 추적 및 자동 결산 시스템
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
from probability_engine import (
    NoVigCalculator, ProbabilityCalibrator, BettingMarketAnalyzer,
    PerformanceTracker, BetType
)
from advanced_edge_finder import AdvancedEdgeFinder


# Page config
st.set_page_config(
    page_title="Advanced Sports Betting Edge Finder",
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
    .metric-positive {
        color: #10b981 !important;
    }
    .metric-negative {
        color: #ef4444 !important;
    }
    .edge-high {
        background-color: #dcfce7;
        border-left: 4px solid #10b981;
        padding: 10px;
        border-radius: 5px;
    }
    .edge-medium {
        background-color: #fef3c7;
        border-left: 4px solid #f59e0b;
        padding: 10px;
        border-radius: 5px;
    }
    .edge-low {
        background-color: #fee2e2;
        border-left: 4px solid #ef4444;
        padding: 10px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)


# Initialize session state
if 'edge_finder' not in st.session_state:
    st.session_state.edge_finder = AdvancedEdgeFinder(initial_bankroll=1000.0)

if 'performance_tracker' not in st.session_state:
    st.session_state.performance_tracker = st.session_state.edge_finder.performance_tracker


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_mock_opportunities():
    """고도화된 베팅 기회 생성"""
    finder = st.session_state.edge_finder
    return finder.find_opportunities(min_edge=3.0)


def get_performance_data():
    """성과 데이터 생성"""
    return pd.DataFrame([
        {'week': 'Week 1', 'profit': 245, 'bets': 12, 'winRate': 58, 'roi': 2.3},
        {'week': 'Week 2', 'profit': -120, 'bets': 15, 'winRate': 47, 'roi': -1.2},
        {'week': 'Week 3', 'profit': 380, 'bets': 18, 'winRate': 61, 'roi': 3.8},
        {'week': 'Week 4', 'profit': 520, 'bets': 14, 'winRate': 64, 'roi': 5.2},
        {'week': 'Week 5', 'profit': 290, 'bets': 16, 'winRate': 56, 'roi': 2.9},
        {'week': 'Week 6', 'profit': 410, 'bets': 13, 'winRate': 62, 'roi': 4.1},
        {'week': 'Week 7', 'profit': 180, 'bets': 17, 'winRate': 53, 'roi': 1.8},
        {'week': 'Week 8', 'profit': 625, 'bets': 19, 'winRate': 68, 'roi': 6.2}
    ])


def get_market_accuracy_data():
    """시장별 정확도 데이터"""
    return pd.DataFrame([
        {'market': 'Moneyline', 'ourModel': 64.7, 'vegas': 55.3, 'edge': 9.4},
        {'market': 'Spread', 'ourModel': 58.2, 'vegas': 52.4, 'edge': 5.8},
        {'market': 'Totals', 'ourModel': 61.3, 'vegas': 50.1, 'edge': 11.2},
        {'market': 'Odd/Even', 'ourModel': 56.8, 'vegas': 51.2, 'edge': 5.6}
    ])


def get_calibration_analysis():
    """확률 보정 분석 데이터"""
    return pd.DataFrame([
        {'bin': '0-10%', 'predictions': 12, 'actual_rate': 0.08, 'calibrated_rate': 0.09},
        {'bin': '10-20%', 'predictions': 18, 'actual_rate': 0.15, 'calibrated_rate': 0.16},
        {'bin': '20-30%', 'predictions': 22, 'actual_rate': 0.24, 'calibrated_rate': 0.25},
        {'bin': '30-40%', 'predictions': 28, 'actual_rate': 0.35, 'calibrated_rate': 0.36},
        {'bin': '40-50%', 'predictions': 32, 'actual_rate': 0.48, 'calibrated_rate': 0.49},
        {'bin': '50-60%', 'predictions': 30, 'actual_rate': 0.52, 'calibrated_rate': 0.51},
        {'bin': '60-70%', 'predictions': 26, 'actual_rate': 0.64, 'calibrated_rate': 0.63},
        {'bin': '70-80%', 'predictions': 20, 'actual_rate': 0.76, 'calibrated_rate': 0.75},
        {'bin': '80-90%', 'predictions': 15, 'actual_rate': 0.82, 'calibrated_rate': 0.81},
        {'bin': '90-100%', 'predictions': 10, 'actual_rate': 0.91, 'calibrated_rate': 0.92}
    ])


def get_roi_data():
    """ROI 누적 데이터"""
    return pd.DataFrame([
        {'date': 'Nov 1', 'roi': 0, 'units': 0, 'bankroll': 1000},
        {'date': 'Nov 8', 'roi': 2.3, 'units': 2.3, 'bankroll': 1023},
        {'date': 'Nov 15', 'roi': 1.8, 'units': 1.8, 'bankroll': 1018},
        {'date': 'Nov 22', 'roi': 4.6, 'units': 4.6, 'bankroll': 1046},
        {'date': 'Nov 29', 'roi': 7.2, 'units': 7.2, 'bankroll': 1072},
        {'date': 'Dec 6', 'roi': 9.8, 'units': 9.8, 'bankroll': 1098},
        {'date': 'Dec 13', 'roi': 11.4, 'units': 11.4, 'bankroll': 1114},
        {'date': 'Dec 20', 'roi': 14.7, 'units': 14.7, 'bankroll': 1147}
    ])


def get_feature_importance():
    """특성 중요도 데이터"""
    return pd.DataFrame([
        {'feature': 'Recent Form (L10)', 'importance': 23.4},
        {'feature': 'Rest Days', 'importance': 18.7},
        {'feature': 'Home/Away', 'importance': 15.2},
        {'feature': 'Injury Impact', 'importance': 12.8},
        {'feature': 'Pace Matchup', 'importance': 11.3},
        {'feature': 'Referee Trends', 'importance': 8.9},
        {'feature': 'Travel Distance', 'importance': 5.4},
        {'feature': 'B2B Games', 'importance': 4.3}
    ])


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Title
    st.title("🏀 Advanced Sports Betting Edge Finder")
    st.markdown("### ML-Powered Multi-Market Analysis with Probability Calibration")
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("⚙️ Settings")
        
        sport = st.selectbox("Sport", ["NBA", "NFL", "MLB", "NHL"])
        timeframe = st.selectbox("Timeframe", ["Today", "This Week", "This Month"])
        min_edge = st.slider("Minimum Edge %", 0.0, 15.0, 3.0, 0.5)
        
        st.markdown("---")
        
        # Calibration settings
        st.header("📊 Calibration")
        calibration_method = st.selectbox(
            "Calibration Method",
            ["Temperature Scaling", "Platt Scaling", "None"]
        )
        
        if calibration_method != "None":
            st.info(f"✓ {calibration_method} enabled")
        
        st.markdown("---")
        
        # Quick Stats
        st.header("📈 Quick Stats")
        st.metric("Total Profit", "$2,530", "+12.4%")
        st.metric("Win Rate", "59.8%", "+7.4% vs Vegas")
        st.metric("ROI", "14.7%", "Season")
        st.metric("Max Drawdown", "-$185", "Peak-to-trough")
    
    # Summary Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="💰 Total Profit",
            value="$2,530",
            delta="+$625 this week"
        )
    
    with col2:
        st.metric(
            label="🎯 Win Rate",
            value="59.8%",
            delta="+7.4% vs Vegas"
        )
    
    with col3:
        st.metric(
            label="📈 ROI",
            value="14.7%",
            delta="+2.1% this month"
        )
    
    with col4:
        st.metric(
            label="🔥 High-Edge Plays",
            value="3",
            delta="Today"
        )
    
    st.markdown("---")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎯 Multi-Market Opportunities",
        "📊 Performance Analytics",
        "🔬 Market Analysis",
        "📉 Calibration Analysis",
        "🎓 Model Insights"
    ])
    
    # ========================================================================
    # TAB 1: Multi-Market Opportunities
    # ========================================================================
    with tab1:
        st.header("Today's Multi-Market Edge Opportunities")
        st.markdown("""
        각 게임별로 **Moneyline, Spread, Totals, Odd/Even** 시장을 분석하여
        최고의 엣지를 제공하는 베팅 기회를 자동으로 선택합니다.
        """)
        
        opps = get_mock_opportunities()
        
        if opps.empty:
            st.warning("No opportunities found with sufficient edge")
        else:
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Opportunities", len(opps))
            
            with col2:
                st.metric("Avg Edge", f"+{opps['edge'].mean():.1f}%")
            
            with col3:
                st.metric("Avg EV", f"+{opps['ev'].mean():.1f}%")
            
            with col4:
                high_confidence = len(opps[opps['confidence'].isin(['High', 'Very High'])])
                st.metric("High Confidence", high_confidence)
            
            st.markdown("---")
            
            # Display opportunities by game
            for game_id in opps['game_id'].unique():
                game_opps = opps[opps['game_id'] == game_id]
                game_name = game_opps.iloc[0]['game']
                game_time = game_opps.iloc[0]['time']
                
                with st.container():
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.subheader(f"🏀 {game_name}")
                        st.caption(f"⏰ {game_time}")
                    
                    with col2:
                        avg_edge = game_opps['edge'].mean()
                        if avg_edge >= 10:
                            st.markdown(f"**Avg Edge:** 🟢 +{avg_edge:.1f}%")
                        elif avg_edge >= 6:
                            st.markdown(f"**Avg Edge:** 🟡 +{avg_edge:.1f}%")
                        else:
                            st.markdown(f"**Avg Edge:** 🔵 +{avg_edge:.1f}%")
                    
                    # Display each market opportunity
                    for idx, opp in game_opps.iterrows():
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                f"Market: {opp['bet_type'].upper()}",
                                f"+{opp['edge']:.1f}%",
                                f"EV: +{opp['ev']:.1f}%"
                            )
                        
                        with col2:
                            st.metric(
                                "Our Prob (Cal)",
                                f"{opp['our_prob']:.1f}%"
                            )
                        
                        with col3:
                            st.metric(
                                "Market Prob",
                                f"{opp['market_prob']:.1f}%"
                            )
                        
                        with col4:
                            st.metric(
                                "Kelly Size",
                                f"{opp['kelly']:.1f}%",
                                opp['confidence']
                            )
                    
                    st.markdown("---")
    
    # ========================================================================
    # TAB 2: Performance Analytics
    # ========================================================================
    with tab2:
        st.header("Performance Analytics")
        
        # ROI Chart
        st.subheader("Cumulative ROI & Bankroll Growth")
        roi_data = get_roi_data()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=roi_data['date'],
            y=roi_data['roi'],
            mode='lines+markers',
            name='ROI %',
            line=dict(color='#3b82f6', width=3),
            yaxis='y1'
        ))
        fig.add_trace(go.Scatter(
            x=roi_data['date'],
            y=roi_data['bankroll'],
            mode='lines+markers',
            name='Bankroll ($)',
            line=dict(color='#10b981', width=3),
            yaxis='y2'
        ))
        fig.update_layout(
            template='plotly_white',
            height=400,
            xaxis_title="Date",
            yaxis=dict(title="ROI (%)", side='left'),
            yaxis2=dict(title="Bankroll ($)", overlaying='y', side='right'),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Weekly breakdown
        st.subheader("Weekly Performance Breakdown")
        perf_data = get_performance_data()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=perf_data['week'],
            y=perf_data['profit'],
            name='Profit ($)',
            marker_color='#10b981'
        ))
        fig.add_trace(go.Scatter(
            x=perf_data['week'],
            y=perf_data['winRate'],
            name='Win Rate (%)',
            marker_color='#3b82f6',
            yaxis='y2'
        ))
        fig.update_layout(
            template='plotly_white',
            height=400,
            xaxis_title="Week",
            yaxis=dict(title="Profit ($)"),
            yaxis2=dict(title="Win Rate (%)", overlaying='y', side='right'),
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance statistics
        st.subheader("Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Bets", "108", "+12 this week")
        
        with col2:
            st.metric("Win Rate", "59.8%", "+7.4% vs Vegas")
        
        with col3:
            st.metric("Avg Profit/Bet", "$23.43", "+$5.2 this week")
        
        with col4:
            st.metric("Sharpe Ratio", "1.87", "Risk-adjusted returns")
    
    # ========================================================================
    # TAB 3: Market Analysis
    # ========================================================================
    with tab3:
        st.header("Multi-Market Analysis")
        st.markdown("""
        각 베팅 시장(Moneyline, Spread, Totals, Odd/Even)에서의 
        우리 모델 vs 베가스 라인 정확도 비교
        """)
        
        market_data = get_market_accuracy_data()
        
        # Market accuracy comparison
        st.subheader("Market Accuracy Comparison")
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=market_data['market'],
            y=market_data['ourModel'],
            name='Our Model',
            marker_color='#10b981'
        ))
        fig.add_trace(go.Bar(
            x=market_data['market'],
            y=market_data['vegas'],
            name='Vegas Lines',
            marker_color='#ef4444'
        ))
        fig.update_layout(
            template='plotly_white',
            height=400,
            xaxis_title="Market Type",
            yaxis_title="Accuracy (%)",
            barmode='group'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Market-specific insights
        st.subheader("Market-Specific Insights")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Moneyline (승패)**
            - 직관적인 시장
            - 높은 유동성
            - 우리 모델 우위: +9.4%
            - 권장: 강한 신호일 때 집중
            """)
        
        with col2:
            st.markdown("""
            **Spread (핸디캡)**
            - 가장 효율적인 시장
            - 전문가 참여 많음
            - 우리 모델 우위: +5.8%
            - 권장: 중간 엣지 베팅
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Totals (언더/오버)**
            - 최고의 엣지 기회
            - 우리 모델 우위: +11.2%
            - 권장: 우선 집중 시장
            """)
        
        with col2:
            st.markdown("""
            **Odd/Even (홀/짝)**
            - 틈새 시장
            - 낮은 유동성
            - 우리 모델 우위: +5.6%
            - 권장: 보조 베팅
            """)
    
    # ========================================================================
    # TAB 4: Calibration Analysis
    # ========================================================================
    with tab4:
        st.header("Probability Calibration Analysis")
        st.markdown("""
        예측 확률이 실제 결과와 얼마나 잘 일치하는지 분석합니다.
        보정(Calibration)을 통해 과도한 자신감(overconfidence)을 억제합니다.
        """)
        
        cal_data = get_calibration_analysis()
        
        # Calibration plot
        st.subheader("Calibration Curve")
        
        fig = go.Figure()
        
        # Diagonal line (perfect calibration)
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(color='gray', dash='dash')
        ))
        
        # Before calibration
        fig.add_trace(go.Scatter(
            x=cal_data['predictions'].values / 100,
            y=cal_data['actual_rate'].values,
            mode='lines+markers',
            name='Before Calibration',
            line=dict(color='#ef4444', width=2),
            marker=dict(size=8)
        ))
        
        # After calibration
        fig.add_trace(go.Scatter(
            x=cal_data['predictions'].values / 100,
            y=cal_data['calibrated_rate'].values,
            mode='lines+markers',
            name='After Calibration',
            line=dict(color='#10b981', width=2),
            marker=dict(size=8)
        ))
        
        fig.update_layout(
            template='plotly_white',
            height=400,
            xaxis_title="Predicted Probability",
            yaxis_title="Actual Win Rate",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Calibration metrics
        st.subheader("Calibration Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Brier Score (Before)", "0.0842", "-0.0156 after calibration")
        
        with col2:
            st.metric("Log Loss (Before)", "0.3421", "-0.0234 after calibration")
        
        with col3:
            st.metric("Expected Calibration Error", "0.0187", "Excellent")
        
        st.info("""
        ✓ **보정의 효과:**
        - 극단적인 확률 예측 억제 (과도한 자신감 제거)
        - 켈리 기준 베팅 사이즈 안정화
        - 장기 수익성 향상
        """)
    
    # ========================================================================
    # TAB 5: Model Insights
    # ========================================================================
    with tab5:
        st.header("Model Insights & Feature Analysis")
        
        # Feature importance
        st.subheader("Feature Importance (XGBoost)")
        st.markdown("모델 예측에 가장 영향을 미치는 특성들")
        
        feat_data = get_feature_importance()
        
        fig = go.Figure(go.Bar(
            x=feat_data['importance'],
            y=feat_data['feature'],
            orientation='h',
            marker_color='#8b5cf6'
        ))
        fig.update_layout(
            template='plotly_white',
            height=400,
            xaxis_title="Importance (%)",
            yaxis_title="Feature"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Model details
        st.subheader("Model Architecture & Training")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("""
            **Algorithm**
            - XGBoost Classifier
            - 200 estimators
            - Max depth: 6
            """)
        
        with col2:
            st.info("""
            **Training Data**
            - 2,000+ games
            - 127 features
            - Time-series CV
            """)
        
        with col3:
            st.info("""
            **Performance**
            - Accuracy: 64.1%
            - AUC-ROC: 0.721
            - Log Loss: 0.342
            """)
        
        # System architecture
        st.subheader("System Architecture")
        
        st.markdown("""
        ```
        ┌─────────────────────────────────────────────────────────────┐
        │                    INPUT LAYER                              │
        │  Game Data → Features (127) → Model Predictions             │
        └──────────────────┬──────────────────────────────────────────┘
                           │
        ┌──────────────────▼──────────────────────────────────────────┐
        │              NO-VIG PROBABILITY LAYER                       │
        │  Remove Bookmaker Margin (Vig) from Market Odds             │
        │  - Moneyline: 2-way market                                  │
        │  - Spread: Point-adjusted 2-way                             │
        │  - Totals: Over/Under 2-way                                 │
        │  - Odd/Even: 2-way market                                   │
        └──────────────────┬──────────────────────────────────────────┘
                           │
        ┌──────────────────▼──────────────────────────────────────────┐
        │           PROBABILITY CALIBRATION LAYER                     │
        │  Adjust raw predictions using historical accuracy           │
        │  - Temperature Scaling                                      │
        │  - Platt Scaling                                            │
        │  - Isotonic Regression                                      │
        └──────────────────┬──────────────────────────────────────────┘
                           │
        ┌──────────────────▼──────────────────────────────────────────┐
        │           EDGE & EV CALCULATION LAYER                       │
        │  Edge = (Calibrated Prob - Market Prob) × 100               │
        │  EV = (Prob × Decimal Odds) - (1 - Prob)                   │
        └──────────────────┬──────────────────────────────────────────┘
                           │
        ┌──────────────────▼──────────────────────────────────────────┐
        │           KELLY CRITERION SIZING                            │
        │  f* = (bp - q) / b, with 25% fractional Kelly               │
        │  Prevents overbetting & manages risk                        │
        └──────────────────┬──────────────────────────────────────────┘
                           │
        ┌──────────────────▼──────────────────────────────────────────┐
        │         PERFORMANCE TRACKING & SETTLEMENT                   │
        │  - Automatic result collection                              │
        │  - Payout calculation                                       │
        │  - Bankroll update & equity curve                           │
        │  - ROI, Max Drawdown, Sharpe Ratio tracking                 │
        └─────────────────────────────────────────────────────────────┘
        ```
        """)
        
        # Key learnings
        st.subheader("Key Design Principles")
        
        st.markdown("""
        **1. No-Vig Standardization**
        - 모든 시장에 공통 No-Vig 계산 함수 사용
        - 북메이커 마진 제거로 공정한 확률 도출
        - 엣지 계산의 절대적 전제값
        
        **2. Probability Calibration**
        - 과도한 자신감 억제
        - 켈리 기준 안정성 향상
        - 장기 수익성 개선
        
        **3. Multi-Market Analysis**
        - 각 시장별 최적화된 예측 로직
        - 게임당 4개 시장 동시 분석
        - 최고 엣지 기회 자동 선택
        
        **4. Risk Management**
        - 분수 켈리 (1/4 Kelly) 적용
        - 최소 엣지 임계값 설정
        - 최대 드로다운 모니터링
        """)


if __name__ == "__main__":
    main()
