'''
from flask import Flask, render_template
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import plotly

app = Flask(__name__, template_folder='../templates')

# ============================================================================
# Data Functions (Copied from original dashboard.py)
# ============================================================================

SPORTS_CATEGORIES = {
    "농구": {
        "icon": "🏀",
        "api_source": "DraftKings / FanDuel / BetMGM",
        "leagues": {
            "NBA": {"name": "미국 NBA", "supported": True, "bookmakers": ["DraftKings", "FanDuel", "BetMGM"]},
            "KBL": {"name": "한국 KBL", "supported": True, "bookmakers": ["DraftKings", "FanDuel", "BetMGM"]},
            "WKBL": {"name": "한국 여자 KBL", "supported": True, "bookmakers": ["DraftKings", "FanDuel", "BetMGM"]}
        }
    },
    "축구": {
        "icon": "⚽",
        "api_source": "DraftKings / FanDuel / BetMGM",
        "leagues": {
            "EPL": {"name": "영국 프리미어리그", "supported": True, "bookmakers": ["DraftKings", "FanDuel", "BetMGM"]},
            "라리가": {"name": "스페인 라리가", "supported": True, "bookmakers": ["DraftKings", "FanDuel", "BetMGM"]}
        }
    },
    "야구": {
        "icon": "⚾",
        "api_source": "DraftKings / FanDuel / BetMGM",
        "leagues": {
            "MLB": {"name": "미국 메이저리그", "supported": True, "bookmakers": ["DraftKings", "FanDuel", "BetMGM"]}
        }
    }
}

def get_opportunities(league_code):
    all_opps = pd.DataFrame([
        {'league': 'NBA', '경기': 'Lakers vs Celtics', '시간': '7:30 PM ET', '예측': 'Lakers 승리', '우리확률': 58.3, '시장확률': 52.1, '엣지': 6.2, '베팅': 'Lakers ML', '배당률': -115, '북메이커': 'DraftKings', '신뢰도': '높음', '켈리': 3.2, '기대값': 8.4},
        {'league': 'NBA', '경기': 'Warriors vs Suns', '시간': '10:00 PM ET', '예측': 'Under 227.5', '우리확률': 61.2, '시장확률': 50.0, '엣지': 11.2, '베팅': 'Under 227.5', '배당률': -110, '북메이커': 'FanDuel', '신뢰도': '매우높음', '켈리': 5.8, '기대값': 12.3},
        {'league': 'KBL', '경기': 'KCC vs SK', '시간': '7:00 PM KST', '예측': 'KCC -3.5', '우리확률': 59.1, '시장확률': 53.5, '엣지': 5.6, '베팅': 'KCC -3.5', '배당률': -112, '북메이커': 'DraftKings', '신뢰도': '높음', '켈리': 2.9, '기대값': 7.1},
        {'league': 'WKBL', '경기': 'Woori WON vs Yongin', '시간': '5:00 AM ET', '예측': 'Woori WON 승리', '우리확률': 56.8, '시장확률': 51.3, '엣지': 5.5, '베팅': 'Woori WON ML', '배당률': -110, '북메이커': 'FanDuel', '신뢰도': '중간', '켈리': 2.8, '기대값': 6.9},
        {'league': 'EPL', '경기': 'Manchester City vs Liverpool', '시간': '3:00 PM GMT', '예측': 'Manchester City 승리', '우리확률': 62.1, '시장확률': 55.3, '엣지': 6.8, '베팅': 'Manchester City ML', '배당률': -155, '북메이커': 'BetMGM', '신뢰도': '높음', '켈리': 3.5, '기대값': 8.9},
        {'league': '라리가', '경기': 'Real Madrid vs Barcelona', '시간': '8:45 PM CET', '예측': 'Real Madrid 승리', '우리확률': 59.8, '시장확률': 52.4, '엣지': 7.4, '베팅': 'Real Madrid ML', '배당률': -125, '북메이커': 'DraftKings', '신뢰도': '높음', '켈리': 3.8, '기대값': 9.2},
        {'league': 'MLB', '경기': 'Yankees vs Red Sox', '시간': '7:05 PM ET', '예측': 'Yankees 승리', '우리확률': 57.6, '시장확률': 51.2, '엣지': 6.4, '베팅': 'Yankees ML', '배당률': -120, '북메이커': 'FanDuel', '신뢰도': '높음', '켈리': 3.3, '기대값': 8.1}
    ])
    if league_code:
        return all_opps[all_opps['league'] == league_code]
    return pd.DataFrame() # Return empty dataframe if no league is selected

def get_performance_data():
    return pd.DataFrame([
        {'주': '1주', '수익': 245, '베팅수': 12, '승률': 58},
        {'주': '2주', '수익': -120, '베팅수': 15, '승률': 47},
        {'주': '3주', '수익': 380, '베팅수': 18, '승률': 61},
        {'주': '4주', '수익': 520, '베팅수': 14, '승률': 64},
        {'주': '5주', '수익': 290, '베팅수': 16, '승률': 56},
        {'주': '6주', '수익': 410, '베팅수': 13, '승률': 62},
        {'주': '7주', '수익': 180, '베팅수': 17, '승률': 53},
        {'주': '8주', '수익': 625, '베팅수': 19, '승률': 68}
    ])

def get_accuracy_data():
    return pd.DataFrame([
        {'카테고리': '스프레드', '우리모델': 58.2, '라스베가스': 52.4},
        {'카테고리': '토탈', '우리모델': 61.3, '라스베가스': 50.1},
        {'카테고리': '머니라인', '우리모델': 64.7, '라스베가스': 55.3},
        {'카테고리': '플레이어소품', '우리모델': 56.8, '라스베가스': 51.2}
    ])

def get_feature_importance():
    return pd.DataFrame([
        {'특성': '최근 폼 (L10)', '중요도': 23.4},
        {'특성': '휴식일', '중요도': 18.7},
        {'특성': '홈/원정', '중요도': 15.2},
        {'특성': '부상 영향', '중요도': 12.8},
        {'특성': '페이스 매치업', '중요도': 11.3},
        {'특성': '심판 트렌드', '중요도': 8.9},
        {'특성': '이동 거리', '중요도': 5.4},
        {'특성': 'B2B 경기', '중요도': 4.3}
    ])

def get_roi_data():
    return pd.DataFrame([
        {'날짜': '11월 1일', 'ROI': 0, '유닛': 0},
        {'날짜': '11월 8일', 'ROI': 2.3, '유닛': 2.3},
        {'날짜': '11월 15일', 'ROI': 1.8, '유닛': 1.8},
        {'날짜': '11월 22일', 'ROI': 4.6, '유닛': 4.6},
        {'날짜': '11월 29일', 'ROI': 7.2, '유닛': 7.2},
        {'날짜': '12월 6일', 'ROI': 9.8, '유닛': 9.8},
        {'날짜': '12월 13일', 'ROI': 11.4, '유닛': 11.4},
        {'날짜': '12월 20일', 'ROI': 14.7, '유닛': 14.7}
    ])

# ============================================================================
# Plotting Functions
# ============================================================================

def create_roi_chart(data):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['날짜'], y=data['ROI'], mode='lines+markers', name='ROI %', line=dict(color='#3b82f6', width=3)))
    fig.add_trace(go.Scatter(x=data['날짜'], y=data['유닛'], mode='lines+markers', name='유닛 획득', line=dict(color='#10b981', width=3)))
    fig.update_layout(template='plotly_white', height=400, xaxis_title="날짜", yaxis_title="값", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_performance_chart(data):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data['주'], y=data['수익'], name='수익 ($)', marker_color='#10b981'))
    fig.add_trace(go.Bar(x=data['주'], y=data['승률'], name='승률 (%)', marker_color='#3b82f6'))
    fig.update_layout(template='plotly_white', height=400, xaxis_title="주", yaxis_title="값", barmode='group', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_accuracy_chart(data):
    fig = go.Figure()
    fig.add_trace(go.Bar(x=data['카테고리'], y=data['우리모델'], name='우리 모델', marker_color='#10b981'))
    fig.add_trace(go.Bar(x=data['카테고리'], y=data['라스베가스'], name='라스베가스 라인', marker_color='#ef4444'))
    fig.update_layout(template='plotly_white', height=400, xaxis_title="베팅 유형", yaxis_title="정확도 (%)", barmode='group', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_feature_importance_chart(data):
    fig = go.Figure(go.Bar(x=data['중요도'], y=data['특성'], orientation='h', marker_color='#8b5cf6'))
    fig.update_layout(template='plotly_white', height=400, xaxis_title="중요도 (%)", yaxis_title="특성")
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

# ============================================================================
# Flask Route
# ============================================================================

@app.route('/', defaults={'sport': None})
@app.route('/<sport>')
def index(sport):
    # Data for summary metrics
    summary_metrics = {
        "total_profit": "$2,530",
        "profit_delta": "+625 이번 주",
        "win_rate": "59.8%",
        "win_rate_delta": "+7.4% vs 라스베가스",
        "roi": "14.7%",
        "roi_delta": "+2.1% 이번 달",
        "high_edge_plays": "3",
        "high_edge_delta": "오늘"
    }

    # Get data for the selected sport
    selected_sport_name = sport if sport in SPORTS_CATEGORIES else None
    opportunities = {}
    if selected_sport_name:
        sport_leagues = SPORTS_CATEGORIES[selected_sport_name]['leagues']
        for league_code, league_info in sport_leagues.items():
            opps_df = get_opportunities(league_code)
            if not opps_df.empty:
                opportunities[league_code] = {
                    "info": league_info,
                    "opps": opps_df.to_dict('records')
                }

    # Generate charts
    roi_chart_json = create_roi_chart(get_roi_data())
    performance_chart_json = create_performance_chart(get_performance_data())
    accuracy_chart_json = create_accuracy_chart(get_accuracy_data())
    feature_importance_chart_json = create_feature_importance_chart(get_feature_importance())
    
    accuracy_metrics_data = get_accuracy_data()
    accuracy_metrics_data['diff'] = (accuracy_metrics_data['우리모델'] - accuracy_metrics_data['라스베가스']).round(1)
    accuracy_metrics = accuracy_metrics_data.to_dict('records')

    return render_template(
        'index.html',
        sports_categories=SPORTS_CATEGORIES,
        selected_sport=selected_sport_name,
        summary_metrics=summary_metrics,
        opportunities=opportunities,
        roi_chart_json=roi_chart_json,
        performance_chart_json=performance_chart_json,
        accuracy_chart_json=accuracy_chart_json,
        feature_importance_chart_json=feature_importance_chart_json,
        accuracy_metrics=accuracy_metrics
    )

if __name__ == "__main__":
    app.run(debug=True)
'''
