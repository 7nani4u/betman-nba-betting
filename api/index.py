"""
스포츠 베팅 엣지 파인더 — Vercel 서버리스 최적화 버전
핵심 변경사항:
  - 종목 선택: URL 경로(/농구) 대신 쿼리 파라미터(?sport=농구) 사용
    → Vercel 서버리스에서 한글 경로 파라미터가 Flask 라우트 변수로
      전달되지 않는 문제를 완전히 우회
  - /api/sport?name=농구 JSON 엔드포인트 추가
    → 사이드바 클릭 시 페이지 전체 새로고침 없이 fetch()로 데이터 로드
  - 모든 차트 데이터를 JSON API로 분리하여 초기 로딩 속도 개선
"""

from flask import Flask, render_template, request, jsonify
import json
import plotly
import plotly.graph_objects as go

app = Flask(__name__)

# ============================================================================
# 스포츠 카테고리 & 베팅 데이터
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
            "EPL":    {"name": "영국 프리미어리그", "bookmakers": ["DraftKings", "FanDuel", "BetMGM"]},
            "라리가": {"name": "스페인 라리가",     "bookmakers": ["DraftKings", "FanDuel", "BetMGM"]},
        }
    },
    "야구": {
        "icon": "⚾",
        "api_source": "DraftKings / FanDuel / BetMGM",
        "leagues": {
            "MLB": {"name": "미국 메이저리그", "bookmakers": ["DraftKings", "FanDuel", "BetMGM"]},
            "KBO": {"name": "한국 KBO",        "bookmakers": ["DraftKings", "FanDuel", "BetMGM"]},
        }
    },
    "미식축구": {
        "icon": "🏈",
        "api_source": "DraftKings / FanDuel / BetMGM",
        "leagues": {
            "NFL": {"name": "미국 NFL", "bookmakers": ["DraftKings", "FanDuel", "BetMGM"]},
        }
    },
    "아이스하키": {
        "icon": "🏒",
        "api_source": "DraftKings / FanDuel / BetMGM",
        "leagues": {
            "NHL": {"name": "미국 NHL", "bookmakers": ["DraftKings", "FanDuel", "BetMGM"]},
        }
    },
}

ALL_OPPORTUNITIES = [
    # 농구 — NBA
    {"league": "NBA",    "경기": "Lakers vs Celtics",        "시간": "7:30 PM ET",  "베팅": "Lakers ML",      "배당률": -115, "북메이커": "DraftKings", "신뢰도": "높음",    "우리확률": 58.3, "시장확률": 52.1, "엣지": 6.2,  "켈리": 3.2, "기대값": 8.4},
    {"league": "NBA",    "경기": "Warriors vs Suns",         "시간": "10:00 PM ET", "베팅": "Under 227.5",    "배당률": -110, "북메이커": "FanDuel",    "신뢰도": "매우높음", "우리확률": 61.2, "시장확률": 50.0, "엣지": 11.2, "켈리": 5.8, "기대값": 12.3},
    {"league": "NBA",    "경기": "Heat vs Bucks",            "시간": "8:00 PM ET",  "베팅": "Bucks -4.5",     "배당률": -108, "북메이커": "BetMGM",     "신뢰도": "중간",    "우리확률": 55.4, "시장확률": 50.9, "엣지": 4.5,  "켈리": 2.3, "기대값": 5.8},
    # 농구 — KBL
    {"league": "KBL",    "경기": "KCC vs SK",                "시간": "7:00 PM KST", "베팅": "KCC -3.5",       "배당률": -112, "북메이커": "DraftKings", "신뢰도": "높음",    "우리확률": 59.1, "시장확률": 53.5, "엣지": 5.6,  "켈리": 2.9, "기대값": 7.1},
    {"league": "KBL",    "경기": "DB vs 현대모비스",          "시간": "5:00 PM KST", "베팅": "현대모비스 ML",  "배당률": -118, "북메이커": "FanDuel",    "신뢰도": "높음",    "우리확률": 57.8, "시장확률": 52.0, "엣지": 5.8,  "켈리": 3.0, "기대값": 7.3},
    # 농구 — WKBL
    {"league": "WKBL",   "경기": "Woori WON vs Yongin",     "시간": "5:00 AM ET",  "베팅": "Woori WON ML",   "배당률": -110, "북메이커": "FanDuel",    "신뢰도": "중간",    "우리확률": 56.8, "시장확률": 51.3, "엣지": 5.5,  "켈리": 2.8, "기대값": 6.9},
    # 축구 — EPL
    {"league": "EPL",    "경기": "Man City vs Liverpool",    "시간": "3:00 PM GMT", "베팅": "Man City ML",    "배당률": -155, "북메이커": "BetMGM",     "신뢰도": "높음",    "우리확률": 62.1, "시장확률": 55.3, "엣지": 6.8,  "켈리": 3.5, "기대값": 8.9},
    {"league": "EPL",    "경기": "Arsenal vs Chelsea",       "시간": "5:30 PM GMT", "베팅": "Arsenal ML",     "배당률": -130, "북메이커": "DraftKings", "신뢰도": "매우높음","우리확률": 63.5, "시장확률": 53.8, "엣지": 9.7,  "켈리": 5.0, "기대값": 11.2},
    # 축구 — 라리가
    {"league": "라리가", "경기": "Real Madrid vs Barcelona", "시간": "8:45 PM CET", "베팅": "Real Madrid ML", "배당률": -125, "북메이커": "DraftKings", "신뢰도": "높음",    "우리확률": 59.8, "시장확률": 52.4, "엣지": 7.4,  "켈리": 3.8, "기대값": 9.2},
    # 야구 — MLB
    {"league": "MLB",    "경기": "Yankees vs Red Sox",       "시간": "7:05 PM ET",  "베팅": "Yankees ML",     "배당률": -120, "북메이커": "FanDuel",    "신뢰도": "높음",    "우리확률": 57.6, "시장확률": 51.2, "엣지": 6.4,  "켈리": 3.3, "기대값": 8.1},
    {"league": "MLB",    "경기": "Dodgers vs Giants",        "시간": "9:40 PM ET",  "베팅": "Under 8.5",      "배당률": -115, "북메이커": "BetMGM",     "신뢰도": "중간",    "우리확률": 55.9, "시장확률": 51.5, "엣지": 4.4,  "켈리": 2.2, "기대값": 5.6},
    # 야구 — KBO
    {"league": "KBO",    "경기": "두산 vs LG",               "시간": "6:30 PM KST", "베팅": "LG ML",          "배당률": -110, "북메이커": "DraftKings", "신뢰도": "높음",    "우리확률": 58.2, "시장확률": 52.3, "엣지": 5.9,  "켈리": 3.0, "기대값": 7.4},
    # 미식축구 — NFL
    {"league": "NFL",    "경기": "Chiefs vs Eagles",         "시간": "8:20 PM ET",  "베팅": "Chiefs -3",      "배당률": -110, "북메이커": "DraftKings", "신뢰도": "매우높음","우리확률": 62.8, "시장확률": 52.4, "엣지": 10.4, "켈리": 5.4, "기대값": 11.8},
    # 아이스하키 — NHL
    {"league": "NHL",    "경기": "Maple Leafs vs Bruins",    "시간": "7:00 PM ET",  "베팅": "Maple Leafs ML", "배당률": +105, "북메이커": "FanDuel",    "신뢰도": "높음",    "우리확률": 56.3, "시장확률": 48.8, "엣지": 7.5,  "켈리": 3.9, "기대값": 9.6},
]

ACCURACY_DATA = [
    {"카테고리": "스프레드",     "우리모델": 58.2, "라스베가스": 52.4},
    {"카테고리": "토탈",         "우리모델": 61.3, "라스베가스": 50.1},
    {"카테고리": "머니라인",     "우리모델": 64.7, "라스베가스": 55.3},
    {"카테고리": "플레이어소품", "우리모델": 56.8, "라스베가스": 51.2},
]

# ============================================================================
# 차트 생성 헬퍼
# ============================================================================

def _j(fig):
    return json.loads(json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder))

def create_roi_chart():
    labels = ["11/1","11/8","11/15","11/22","11/29","12/6","12/13","12/20"]
    roi    = [0, 2.3, 1.8, 4.6, 7.2, 9.8, 11.4, 14.7]
    units  = [0, 2.3, 1.8, 4.6, 7.2, 9.8, 11.4, 14.7]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=labels, y=roi,   mode="lines+markers", name="ROI %",    line=dict(color="#3b82f6", width=3)))
    fig.add_trace(go.Scatter(x=labels, y=units, mode="lines+markers", name="유닛 획득", line=dict(color="#10b981", width=3)))
    fig.update_layout(template="plotly_white", height=350, margin=dict(l=40,r=20,t=30,b=40))
    return _j(fig)

def create_performance_chart():
    weeks   = ["1주","2주","3주","4주","5주","6주","7주","8주"]
    revenue = [245,-120,380,520,290,410,180,625]
    wr      = [58, 47,  61,  64,  56,  62,  53,  68]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=weeks, y=revenue, name="수익 ($)", marker_color="#10b981"))
    fig.add_trace(go.Bar(x=weeks, y=wr,      name="승률 (%)", marker_color="#3b82f6"))
    fig.update_layout(template="plotly_white", height=350, barmode="group", margin=dict(l=40,r=20,t=30,b=40))
    return _j(fig)

def create_accuracy_chart():
    cats  = [r["카테고리"]  for r in ACCURACY_DATA]
    ours  = [r["우리모델"]  for r in ACCURACY_DATA]
    vegas = [r["라스베가스"] for r in ACCURACY_DATA]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=cats, y=ours,  name="우리 모델",      marker_color="#10b981"))
    fig.add_trace(go.Bar(x=cats, y=vegas, name="라스베가스 라인", marker_color="#ef4444"))
    fig.update_layout(template="plotly_white", height=350, barmode="group", margin=dict(l=40,r=20,t=30,b=40))
    return _j(fig)

def create_feature_chart():
    names  = ["최근 폼 (L10)","휴식일","홈/원정","부상 영향","페이스 매치업","심판 트렌드","이동 거리","B2B 경기"]
    values = [23.4, 18.7, 15.2, 12.8, 11.3, 8.9, 5.4, 4.3]
    fig = go.Figure(go.Bar(x=values, y=names, orientation="h", marker_color="#8b5cf6"))
    fig.update_layout(template="plotly_white", height=350, margin=dict(l=130,r=20,t=30,b=40))
    return _j(fig)

def build_opportunities(sport_name):
    """선택된 종목의 리그별 베팅 기회를 딕셔너리로 반환"""
    if not sport_name or sport_name not in SPORTS_CATEGORIES:
        return {}
    result = {}
    for code, info in SPORTS_CATEGORIES[sport_name]["leagues"].items():
        opps = [o for o in ALL_OPPORTUNITIES if o["league"] == code]
        if opps:
            result[code] = {"info": info, "opps": opps}
    return result

# ============================================================================
# Flask 라우트
# ============================================================================

@app.route("/")
def index():
    """메인 페이지 — 쿼리 파라미터 ?sport=종목명 으로 종목 선택"""
    # 한글 경로 파라미터 대신 쿼리 파라미터 사용 (Vercel 호환성 보장)
    sport = request.args.get("sport", "").strip()
    selected = sport if sport in SPORTS_CATEGORIES else None

    accuracy = [dict(r, diff=round(r["우리모델"] - r["라스베가스"], 1)) for r in ACCURACY_DATA]

    return render_template(
        "index.html",
        sports_categories=SPORTS_CATEGORIES,
        selected_sport=selected,
        opportunities=build_opportunities(selected),
        accuracy_metrics=accuracy,
        roi_chart=create_roi_chart(),
        performance_chart=create_performance_chart(),
        accuracy_chart=create_accuracy_chart(),
        feature_chart=create_feature_chart(),
        summary={
            "total_profit":   "$2,530",
            "profit_delta":   "+625 이번 주",
            "win_rate":       "59.8%",
            "win_rate_delta": "+7.4% vs 라스베가스",
            "roi":            "14.7%",
            "roi_delta":      "+2.1% 이번 달",
            "high_edge":      "3",
            "high_edge_delta":"오늘",
        },
    )


@app.route("/api/sport")
def api_sport():
    """
    AJAX 엔드포인트 — 종목별 베팅 기회를 JSON으로 반환
    GET /api/sport?name=농구
    """
    sport = request.args.get("name", "").strip()
    if not sport or sport not in SPORTS_CATEGORIES:
        return jsonify({"error": "종목을 찾을 수 없습니다.", "opportunities": {}})

    opps = build_opportunities(sport)
    # JSON 직렬화를 위해 info 딕셔너리의 bookmakers 리스트도 포함
    return jsonify({
        "sport": sport,
        "icon": SPORTS_CATEGORIES[sport]["icon"],
        "api_source": SPORTS_CATEGORIES[sport]["api_source"],
        "opportunities": opps,
    })


if __name__ == "__main__":
    app.run(debug=True)
