import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
ODDS_API_KEY = st.secrets["ODDS_API_KEY"]

# Page config
st.set_page_config(
    page_title="Sports Betting Edge Finder",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - FIXED COLORS
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
    .stMetric label {
        color: #262730 !important;
    }
    .stMetric .css-1xarl3l {
        color: #262730 !important;
    }
    div[data-testid="stMetricValue"] {
        color: #262730 !important;
        font-size: 2rem !important;
    }
    div[data-testid="stMetricDelta"] {
        color: #09ab3b !important;
    }
    </style>
""", unsafe_allow_html=True)

# Mock data functions
def get_opportunities():
    """Get current betting opportunities"""
    return pd.DataFrame([
        {
            'game': 'Lakers vs Celtics',
            'time': '7:30 PM ET',
            'prediction': 'Lakers Win',
            'our_prob': 58.3,
            'market_prob': 52.1,
            'edge': 6.2,
            'bet': 'Lakers ML',
            'odds': -115,
            'book': 'DraftKings',
            'confidence': 'High',
            'kelly': 3.2,
            'ev': 8.4
        },
        {
            'game': 'Warriors vs Suns',
            'time': '10:00 PM ET',
            'prediction': 'Under 227.5',
            'our_prob': 61.2,
            'market_prob': 50.0,
            'edge': 11.2,
            'bet': 'Under 227.5',
            'odds': -110,
            'book': 'FanDuel',
            'confidence': 'Very High',
            'kelly': 5.8,
            'ev': 12.3
        },
        {
            'game': 'Heat vs Bucks',
            'time': '8:00 PM ET',
            'prediction': 'Bucks -4.5',
            'our_prob': 55.8,
            'market_prob': 52.4,
            'edge': 3.4,
            'bet': 'Bucks -4.5',
            'odds': -108,
            'book': 'BetMGM',
            'confidence': 'Medium',
            'kelly': 1.8,
            'ev': 4.2
        }
    ])

def get_performance_data():
    """Get historical performance data"""
    return pd.DataFrame([
        {'week': 'Week 1', 'profit': 245, 'bets': 12, 'winRate': 58},
        {'week': 'Week 2', 'profit': -120, 'bets': 15, 'winRate': 47},
        {'week': 'Week 3', 'profit': 380, 'bets': 18, 'winRate': 61},
        {'week': 'Week 4', 'profit': 520, 'bets': 14, 'winRate': 64},
        {'week': 'Week 5', 'profit': 290, 'bets': 16, 'winRate': 56},
        {'week': 'Week 6', 'profit': 410, 'bets': 13, 'winRate': 62},
        {'week': 'Week 7', 'profit': 180, 'bets': 17, 'winRate': 53},
        {'week': 'Week 8', 'profit': 625, 'bets': 19, 'winRate': 68}
    ])

def get_accuracy_data():
    """Get model accuracy comparison"""
    return pd.DataFrame([
        {'category': 'Spreads', 'ourModel': 58.2, 'vegas': 52.4},
        {'category': 'Totals', 'ourModel': 61.3, 'vegas': 50.1},
        {'category': 'Moneylines', 'ourModel': 64.7, 'vegas': 55.3},
        {'category': 'Player Props', 'ourModel': 56.8, 'vegas': 51.2}
    ])

def get_feature_importance():
    """Get feature importance data"""
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

def get_roi_data():
    """Get cumulative ROI data"""
    return pd.DataFrame([
        {'date': 'Nov 1', 'roi': 0, 'units': 0},
        {'date': 'Nov 8', 'roi': 2.3, 'units': 2.3},
        {'date': 'Nov 15', 'roi': 1.8, 'units': 1.8},
        {'date': 'Nov 22', 'roi': 4.6, 'units': 4.6},
        {'date': 'Nov 29', 'roi': 7.2, 'units': 7.2},
        {'date': 'Dec 6', 'roi': 9.8, 'units': 9.8},
        {'date': 'Dec 13', 'roi': 11.4, 'units': 11.4},
        {'date': 'Dec 20', 'roi': 14.7, 'units': 14.7}
    ])

# Main app
def main():
    # Title
    st.title("🏀 Sports Betting Edge Finder")
    st.markdown("### ML-Powered System to Identify Mispriced Lines & Generate Alpha")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        sport = st.selectbox("Sport", ["NBA", "NFL", "MLB", "NHL"])
        timeframe = st.selectbox("Timeframe", ["Today", "This Week", "This Month"])
        min_edge = st.slider("Minimum Edge %", 0.0, 15.0, 3.0, 0.5)
        
        st.markdown("---")
        st.header("📊 Quick Stats")
        st.metric("Total Profit", "$2,530", "+12.4%")
        st.metric("Win Rate", "59.8%", "+7.4% vs Vegas")
        st.metric("ROI", "14.7%", "Season")
    
    # Summary metrics
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
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Opportunities",
        "📊 Performance",
        "🎓 Model Accuracy",
        "🔬 Feature Analysis"
    ])
    
    # Tab 1: Opportunities
    with tab1:
        st.header("Today's Edge Opportunities")
        
        opps = get_opportunities()
        
        for idx, opp in opps.iterrows():
            with st.container():
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.subheader(f"🏀 {opp['game']}")
                    st.caption(f"⏰ {opp['time']}")
                
                with col2:
                    confidence_color = {
                        'Very High': 'green',
                        'High': 'blue',
                        'Medium': 'orange',
                        'Low': 'red'
                    }
                    st.markdown(f"**Confidence:** :{confidence_color[opp['confidence']]}[{opp['confidence']}]")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Our Probability", f"{opp['our_prob']:.1f}%")
                
                with col2:
                    st.metric("Market Probability", f"{opp['market_prob']:.1f}%")
                
                with col3:
                    st.metric("Edge", f"+{opp['edge']:.1f}%", delta="Edge")
                
                with col4:
                    st.metric("Expected Value", f"+{opp['ev']:.1f}%")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.info(f"**Recommended Bet:** {opp['bet']}")
                    st.caption(f"{opp['book']} • {opp['odds']}")
                
                with col2:
                    st.success(f"**Kelly Criterion:** {opp['kelly']:.1f}% of bankroll")
                
                st.markdown("---")
    
    # Tab 2: Performance
    with tab2:
        st.header("Performance Analytics")
        
        # ROI Chart
        st.subheader("Cumulative ROI & Units Won")
        roi_data = get_roi_data()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=roi_data['date'],
            y=roi_data['roi'],
            mode='lines+markers',
            name='ROI %',
            line=dict(color='#3b82f6', width=3)
        ))
        fig.add_trace(go.Scatter(
            x=roi_data['date'],
            y=roi_data['units'],
            mode='lines+markers',
            name='Units Won',
            line=dict(color='#10b981', width=3)
        ))
        fig.update_layout(
            template='plotly_white',
            height=400,
            xaxis_title="Date",
            yaxis_title="Value"
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
        fig.add_trace(go.Bar(
            x=perf_data['week'],
            y=perf_data['winRate'],
            name='Win Rate (%)',
            marker_color='#3b82f6'
        ))
        fig.update_layout(
            template='plotly_white',
            height=400,
            xaxis_title="Week",
            yaxis_title="Value"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Tab 3: Accuracy
    with tab3:
        st.header("Model Accuracy vs Vegas Lines")
        st.markdown("Our model consistently outperforms market odds across all bet types")
        
        acc_data = get_accuracy_data()
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=acc_data['category'],
            y=acc_data['ourModel'],
            name='Our Model',
            marker_color='#10b981'
        ))
        fig.add_trace(go.Bar(
            x=acc_data['category'],
            y=acc_data['vegas'],
            name='Vegas Lines',
            marker_color='#ef4444'
        ))
        fig.update_layout(
            template='plotly_white',
            height=400,
            xaxis_title="Bet Type",
            yaxis_title="Accuracy (%)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary stats
        col1, col2, col3, col4 = st.columns(4)
        
        for i, row in acc_data.iterrows():
            diff = row['ourModel'] - row['vegas']
            with [col1, col2, col3, col4][i]:
                st.metric(
                    row['category'],
                    f"+{diff:.1f}%",
                    "advantage"
                )
    
    # Tab 4: Features
    with tab4:
        st.header("Feature Importance Analysis")
        st.markdown("Key factors driving our predictions (from XGBoost model)")
        
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
        st.subheader("Model Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.info("**Algorithm:** XGBoost Ensemble")
            st.info("**Training Data:** 15,000+ games")
        
        with col2:
            st.info("**Features Used:** 127 variables")
            st.info("**Update Frequency:** Real-time")
        
        with col3:
            st.info("**Validation:** Time-series CV")
            st.info("**Backtest Period:** 2023-24 Season")

if __name__ == "__main__":
    main()