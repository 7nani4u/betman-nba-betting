from flask import Flask, render_template
import pandas as pd
import plotly.graph_objects as go
import json
import plotly

# Flask가 기본으로 찾는 templates 폴더는 이 파일(index.py)과 같은 디렉터리 안입니다.
# Vercel에서도 api/templates/ 구조로 배치하면 별도 경로 설정 없이 동작합니다.
app = Flask(__name__)

# ============================================================================
# 스포츠 카테고리 데이터
# ============================================================================

SPORTS_CATEGORIES = {
    "농구": {
        "icon": "🏀",
        "api_source": "DraftKings / FanDuel / BetMGM",
        "leagues": {
            "NBA":  {"name": "미국 NBA",      "bookmakers": ["DraftKings", "FanDuel", "BetMGM"]},
            "KBL":  {"name": "한국 KBL",      "bookmakers": ["DraftKings", "FanDuel", "BetMGM"]},
            "WKBL": {"name": "한국 여자 KBL", "bookmakers": ["DraftKings", "FanDuel", "BetMGM"]},
        }
    },
    "축구": {
        "icon": "⚽",
        "api_source": "DraftKings / FanDuel / BetMGM",
        "leagues": {
            "EPL":   {"name": "영국 프리미어리그", "bookmakers": ["DraftKings", "FanDuel", "BetMGM"]},
            "라리가": {"name": "스페인 라리가",    "bookmakers": ["DraftKings", "FanDuel", "BetMGM"]},
        }
    },
    "야구": {
        "icon": "⚾",
        "api_source": "DraftKings / FanDuel / BetMGM",
        "leagues": {
            "MLB": {"name": "미국 메이저리그", "bookmakers": ["DraftKings", "FanDuel", "BetMGM"]},
        }
    }
}

# ============================================================================
# 데이터
# ============================================================================

ALL_OPPORTUNITIES = [
    {"league": "NBA",    "경기": "Lakers vs Celtics",        "시간": "7:30 PM ET",  "베팅": "Lakers ML",      "배당률": -115, "북메이커": "DraftKings", "신뢰도": "높음",    "우리확률": 58.3, "시장확률": 52.1, "엣지": 6.2,  "켈리": 3.2, "기대값": 8.4},
    {"league": "NBA",    "경기": "Warriors vs Suns",         "시간": "10:00 PM ET", "베팅": "Under 227.5",    "배당률": -110, "북메이커": "FanDuel",    "신뢰도": "매우높음", "우리확률": 61.2, "시장확률": 50.0, "엣지": 11.2, "켈리": 5.8, "기대값": 12.3},
    {"league": "KBL",    "경기": "KCC vs SK",                "시간": "7:00 PM KST", "베팅": "KCC -3.5",       "배당률": -112, "북메이커": "DraftKings", "신뢰도": "높음",    "우리확률": 59.1, "시장확률": 53.5, "엣지": 5.6,  "켈리": 2.9, "기대값": 7.1},
    {"league": "WKBL",   "경기": "Woori WON vs Yongin",     "시간": "5:00 AM ET",  "베팅": "Woori WON ML",   "배당률": -110, "북메이커": "FanDuel",    "신뢰도": "중간",    "우리확률": 56.8, "시장확률": 51.3, "엣지": 5.5,  "켈리": 2.8, "기대값": 6.9},
    {"league": "EPL",    "경기": "Man City vs Liverpool",    "시간": "3:00 PM GMT", "베팅": "Man City ML",    "배당률": -155, "북메이커": "BetMGM",     "신뢰도": "높음",    "우리확률": 62.1, "시장확률": 55.3, "엣지": 6.8,  "켈리": 3.5, "기대값": 8.9},
    {"league": "라리가", "경기": "Real Madrid vs Barcelona", "시간": "8:45 PM CET", "베팅": "Real Madrid ML", "배당률": -125, "북메이커": "DraftKings", "신뢰도": "높음",    "우리확률": 59.8, "시장확률": 52.4, "엣지": 7.4,  "켈리": 3.8, "기대값": 9.2},
    {"league": "MLB",    "경기": "Yankees vs Red Sox",       "시간": "7:05 PM ET",  "베팅": "Yankees ML",     "배당률": -120, "북메이커": "FanDuel",    "신뢰도": "높음",    "우리확률": 57.6, "시장확률": 51.2, "엣지": 6.4,  "켈리": 3.3, "기대값": 8.1},
]

def get_opportunities(league_code):
    return [o for o in ALL_OPPORTUNITIES if o["league"] == league_code]

def get_accuracy_data():
    return [
        {"카테고리": "스프레드",     "우리모델": 58.2, "라스베가스": 52.4},
        {"카테고리": "토탈",         "우리모델": 61.3, "라스베가스": 50.1},
        {"카테고리": "머니라인",     "우리모델": 64.7, "라스베가스": 55.3},
        {"카테고리": "플레이어소품", "우리모델": 56.8, "라스베가스": 51.2},
    ]

# ============================================================================
# 차트 생성 (Plotly → JSON)
# ============================================================================

def _j(fig):
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

def create_roi_chart():
    labels = ["11/1","11/8","11/15","11/22","11/29","12/6","12/13","12/20"]
    roi    = [0, 2.3, 1.8, 4.6, 7.2, 9.8, 11.4, 14.7]
    units  = [0, 2.3, 1.8, 4.6, 7.2, 9.8, 11.4, 14.7]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels, y=roi,   mode="lines+markers", name="ROI %",   line=dict(color="#3b82f6", width=3)))
    fig.add_trace(go.Scatter(x=labels, y=units, mode="lines+markers", name="유닛 획득", line=dict(color="#10b981", width=3)))
    fig.update_layout(template="plotly_white", height=380, margin=dict(l=40,r=20,t=30,b=40))
    return _j(fig)

def create_performance_chart():
    weeks   = ["1주","2주","3주","4주","5주","6주","7주","8주"]
    revenue = [245,-120,380,520,290,410,180,625]
    wr      = [58,47,61,64,56,62,53,68]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=weeks, y=revenue, name="수익 ($)", marker_color="#10b981"))
    fig.add_trace(go.Bar(x=weeks, y=wr,      name="승률 (%)", marker_color="#3b82f6"))
    fig.update_layout(template="plotly_white", height=380, barmode="group", margin=dict(l=40,r=20,t=30,b=40))
    return _j(fig)

def create_accuracy_chart():
    d     = get_accuracy_data()
    cats  = [r["카테고리"]  for r in d]
    ours  = [r["우리모델"]  for r in d]
    vegas = [r["라스베가스"] for r in d]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cats, y=ours,  name="우리 모델",      marker_color="#10b981"))
    fig.add_trace(go.Bar(x=cats, y=vegas, name="라스베가스 라인", marker_color="#ef4444"))
    fig.update_layout(template="plotly_white", height=380, barmode="group", margin=dict(l=40,r=20,t=30,b=40))
    return _j(fig)

def create_feature_chart():
    names  = ["최근 폼 (L10)","휴식일","홈/원정","부상 영향","페이스 매치업","심판 트렌드","이동 거리","B2B 경기"]
    values = [23.4, 18.7, 15.2, 12.8, 11.3, 8.9, 5.4, 4.3]
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color="#8b5cf6"))
    fig.update_layout(template="plotly_white", height=380, margin=dict(l=120,r=20,t=30,b=40))
    return _j(fig)

# ============================================================================
# Flask 라우트
# ============================================================================

@app.route("/", defaults={"sport": None})
@app.route("/<sport>")
def index(sport):
    selected = sport if sport in SPORTS_CATEGORIES else None

    opportunities = {}
    if selected:
        for code, info in SPORTS_CATEGORIES[selected]["leagues"].items():
            opps = get_opportunities(code)
            if opps:
                opportunities[code] = {"info": info, "opps": opps}

    accuracy_data = get_accuracy_data()
    for row in accuracy_data:
        row["diff"] = round(row["우리모델"] - row["라스베가스"], 1)

    return render_template(
        "index.html",
        sports_categories=SPORTS_CATEGORIES,
        selected_sport=selected,
        opportunities=opportunities,
        accuracy_metrics=accuracy_data,
        roi_chart_json=create_roi_chart(),
        performance_chart_json=create_performance_chart(),
        accuracy_chart_json=create_accuracy_chart(),
        feature_importance_chart_json=create_feature_chart(),
        summary={
            "total_profit":  "$2,530",
            "profit_delta":  "+625 이번 주",
            "win_rate":      "59.8%",
            "win_rate_delta":"+7.4% vs 라스베가스",
            "roi":           "14.7%",
            "roi_delta":     "+2.1% 이번 달",
            "high_edge":     "3",
            "high_edge_delta":"오늘",
        },
    )

if __name__ == "__main__":
    app.run(debug=True)
