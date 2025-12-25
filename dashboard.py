
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

# ============================================================================
# 스포츠 카테고리 매핑 (배구 제거)
# ============================================================================

SPORTS_CATEGORIES = {
    "농구": {
        "icon": "🏀",
        "api_source": "DraftKings / FanDuel / BetMGM",
        "leagues": {
            "NBA": {
                "name": "미국 NBA",
                "supported": True,
                "bookmakers": ["DraftKings", "FanDuel", "BetMGM"]
            },
            "KBL": {
                "name": "한국 KBL",
                "supported": True,
                "bookmakers": ["DraftKings", "FanDuel", "BetMGM"]
            },
            "WKBL": {
                "name": "한국 여자 KBL",
                "supported": True,
                "bookmakers": ["DraftKings", "FanDuel", "BetMGM"]
            }
        }
    },
    "축구": {
        "icon": "⚽",
        "api_source": "DraftKings / FanDuel / BetMGM",
        "leagues": {
            "EPL": {
                "name": "영국 프리미어리그",
                "supported": True,
                "bookmakers": ["DraftKings", "FanDuel", "BetMGM"]
            },
            "라리가": {
                "name": "스페인 라리가",
                "supported": True,
                "bookmakers": ["DraftKings", "FanDuel", "BetMGM"]
            }
        }
    },
    "야구": {
        "icon": "⚾",
        "api_source": "DraftKings / FanDuel / BetMGM",
        "leagues": {
            "MLB": {
                "name": "미국 메이저리그",
                "supported": True,
                "bookmakers": ["DraftKings", "FanDuel", "BetMGM"]
            }
        }
    }
}

# ============================================================================
# 페이지 설정
# ============================================================================

st.set_page_config(
    page_title="스포츠 베팅 엣지 파인더",
    page_icon="🏀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
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
    .league-card {
        background-color: #f3f4f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #3b82f6;
        margin-bottom: 15px;
    }
    .league-card-title {
        font-size: 1.3em;
        font-weight: bold;
        color: #1e40af;
        margin-bottom: 8px;
    }
    .league-card-desc {
        font-size: 1em;
        color: #4b5563;
    }
    .league-card-meta {
        font-size: 0.9em;
        color: #6b7280;
        margin-top: 8px;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# 세션 상태 초기화
# ============================================================================

if 'selected_sport' not in st.session_state:
    st.session_state.selected_sport = None

# ============================================================================
# 데이터 함수
# ============================================================================

def get_opportunities(league_code):
    """특정 리그의 베팅 기회 데이터를 조회 (샘플)"""
    all_opps = pd.DataFrame([
        {
            'league': 'NBA',
            '경기': 'Lakers vs Celtics',
            '시간': '7:30 PM ET',
            '예측': 'Lakers 승리',
            '우리확률': 58.3,
            '시장확률': 52.1,
            '엣지': 6.2,
            '베팅': 'Lakers ML',
            '배당률': -115,
            '북메이커': 'DraftKings',
            '신뢰도': '높음',
            '켈리': 3.2,
            '기대값': 8.4
        },
        {
            'league': 'NBA',
            '경기': 'Warriors vs Suns',
            '시간': '10:00 PM ET',
            '예측': 'Under 227.5',
            '우리확률': 61.2,
            '시장확률': 50.0,
            '엣지': 11.2,
            '베팅': 'Under 227.5',
            '배당률': -110,
            '북메이커': 'FanDuel',
            '신뢰도': '매우높음',
            '켈리': 5.8,
            '기대값': 12.3
        },
        {
            'league': 'KBL',
            '경기': 'KCC vs SK',
            '시간': '7:00 PM KST',
            '예측': 'KCC -3.5',
            '우리확률': 59.1,
            '시장확률': 53.5,
            '엣지': 5.6,
            '베팅': 'KCC -3.5',
            '배당률': -112,
            '북메이커': 'DraftKings',
            '신뢰도': '높음',
            '켈리': 2.9,
            '기대값': 7.1
        },
        {
            'league': 'WKBL',
            '경기': 'Woori WON vs Yongin',
            '시간': '5:00 AM ET',
            '예측': 'Woori WON 승리',
            '우리확률': 56.8,
            '시장확률': 51.3,
            '엣지': 5.5,
            '베팅': 'Woori WON ML',
            '배당률': -110,
            '북메이커': 'FanDuel',
            '신뢰도': '중간',
            '켈리': 2.8,
            '기대값': 6.9
        },
        {
            'league': 'EPL',
            '경기': 'Manchester City vs Liverpool',
            '시간': '3:00 PM GMT',
            '예측': 'Manchester City 승리',
            '우리확률': 62.1,
            '시장확률': 55.3,
            '엣지': 6.8,
            '베팅': 'Manchester City ML',
            '배당률': -155,
            '북메이커': 'BetMGM',
            '신뢰도': '높음',
            '켈리': 3.5,
            '기대값': 8.9
        },
        {
            'league': '라리가',
            '경기': 'Real Madrid vs Barcelona',
            '시간': '8:45 PM CET',
            '예측': 'Real Madrid 승리',
            '우리확률': 59.8,
            '시장확률': 52.4,
            '엣지': 7.4,
            '베팅': 'Real Madrid ML',
            '배당률': -125,
            '북메이커': 'DraftKings',
            '신뢰도': '높음',
            '켈리': 3.8,
            '기대값': 9.2
        },
        {
            'league': 'MLB',
            '경기': 'Yankees vs Red Sox',
            '시간': '7:05 PM ET',
            '예측': 'Yankees 승리',
            '우리확률': 57.6,
            '시장확률': 51.2,
            '엣지': 6.4,
            '베팅': 'Yankees ML',
            '배당률': -120,
            '북메이커': 'FanDuel',
            '신뢰도': '높음',
            '켈리': 3.3,
            '기대값': 8.1
        }
    ])
    return all_opps[all_opps['league'] == league_code]

def get_performance_data():
    """성과 데이터 조회"""
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
    """모델 정확도 데이터 조회"""
    return pd.DataFrame([
        {'카테고리': '스프레드', '우리모델': 58.2, '라스베가스': 52.4},
        {'카테고리': '토탈', '우리모델': 61.3, '라스베가스': 50.1},
        {'카테고리': '머니라인', '우리모델': 64.7, '라스베가스': 55.3},
        {'카테고리': '플레이어소품', '우리모델': 56.8, '라스베가스': 51.2}
    ])

def get_feature_importance():
    """특성 중요도 데이터 조회"""
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
    """누적 ROI 데이터 조회"""
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
# 메인 앱
# ============================================================================

def main():
    # 제목
    st.title("🏀 스포츠 베팅 엣지 파인더")
    st.markdown("### ML 기반 시스템으로 저평가된 라인 식별 및 알파 생성")

    # ========================================================================
    # 사이드바 설정
    # ========================================================================
    with st.sidebar:
        st.header("⚙️ 설정")

        # --- 기간 선택 로직 수정 ---
        today = datetime.now()
        start_of_week = today - timedelta(days=today.weekday())
        end_of_week = start_of_week + timedelta(days=6)

        # 기간 선택 옵션 생성 (표시용 레이블과 실제 값 분리)
        timeframe_options = {
            f"오늘 ({today.strftime('%Y-%m-%d')})": "오늘",
            f"이번 주 ({start_of_week.strftime('%m/%d')} ~ {end_of_week.strftime('%m/%d')})": "이번 주",
            f"이번 달 ({today.strftime('%Y-%m')})": "이번 달"
        }

        # selectbox에 표시될 레이블 리스트
        timeframe_labels = list(timeframe_options.keys())

        # 사용자가 선택한 표시용 레이블
        selected_label = st.selectbox(
            "기간",
            timeframe_labels,
            key="timeframe_selectbox"
        )

        # 선택된 레이블에 해당하는 실제 값 (예: "오늘")을 가져옴
        timeframe = timeframe_options[selected_label]
        # --------------------------

        min_edge = st.slider(
            "최소 엣지 (%)",
            0.0,
            15.0,
            3.0,
            0.5
        )

        st.markdown("---")

        # 스포츠 카테고리 선택 섹션
        st.header("🏆 종목 선택")
        st.markdown("아래에서 원하는 스포츠 종목을 선택하세요")

        # 지원되는 스포츠 종목
        for sport_name, sport_data in SPORTS_CATEGORIES.items():
            icon = sport_data["icon"]

            if st.button(
                f"{icon} {sport_name}",
                key=f"sport_{sport_name}",
                use_container_width=True
            ):
                st.session_state.selected_sport = sport_name
                st.rerun()

    # ========================================================================
    # 메인 콘텐츠 영역
    # ========================================================================

    # 요약 메트릭
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="💰 총 수익",
            value="$2,530",
            delta="+625 이번 주"
        )

    with col2:
        st.metric(
            label="🎯 승률",
            value="59.8%",
            delta="+7.4% vs 라스베가스"
        )

    with col3:
        st.metric(
            label="📈 ROI",
            value="14.7%",
            delta="+2.1% 이번 달"
        )

    with col4:
        st.metric(
            label="🔥 고엣지 플레이",
            value="3",
            delta="오늘"
        )

    st.markdown("---")

    # ========================================================================
    # 탭 구성
    # ========================================================================
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 베팅 기회",
        "📊 성과 분석",
        "🎓 모델 정확도",
        "🔬 특성 분석"
    ])

    # 탭 1: 베팅 기회
    with tab1:
        if st.session_state.selected_sport is None:
            st.info("👈 왼쪽 사이드바에서 스포츠 종목을 선택하세요")
        else:
            sport_data = SPORTS_CATEGORIES[st.session_state.selected_sport]
            leagues = sport_data["leagues"]

            st.header(f"{sport_data['icon']} {st.session_state.selected_sport} - 리그별 베팅 기회")
            st.caption(f"📡 데이터 소스: {sport_data['api_source']}")

            for league_code, league_info in leagues.items():
                with st.container():
                    # 리그 정보 카드
                    st.markdown(f"""
                    <div class="league-card">
                        <div class="league-card-title">📺 {league_code}</div>
                        <div class="league-card-desc">{league_info['name']}</div>
                        <div class="league-card-meta">
                            📊 북메이커: {', '.join(league_info['bookmakers'])}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # 해당 리그의 베팅 기회 표시
                    opps = get_opportunities(league_code)

                    if opps.empty:
                        st.warning(f"{league_info['name']}에 대한 베팅 기회가 없습니다.")
                    else:
                        for idx, opp in opps.iterrows():
                            with st.container():
                                col1, col2 = st.columns([3, 1])

                                with col1:
                                    st.subheader(f"🏀 {opp['경기']}")
                                    st.caption(f"⏰ {opp['시간']}")

                                with col2:
                                    confidence_color = {
                                        '매우높음': 'green',
                                        '높음': 'blue',
                                        '중간': 'orange',
                                        '낮음': 'red'
                                    }
                                    color = confidence_color.get(opp['신뢰도'], 'gray')
                                    st.markdown(f"**신뢰도:** :{color}[{opp['신뢰도']}]")

                                c1, c2, c3, c4 = st.columns(4)
                                c1.metric("우리 확률", f"{opp['우리확률']:.1f}%")
                                c2.metric("시장 확률", f"{opp['시장확률']:.1f}%")
                                c3.metric("엣지", f"+{opp['엣지']:.1f}%", delta="엣지")
                                c4.metric("기대값", f"+{opp['기대값']:.1f}%")

                                c1, c2 = st.columns(2)
                                c1.info(f"**추천 베팅:** {opp['베팅']}")
                                # --- 오류 수정: f-string이 한 줄에 있도록 수정 ---
                                c1.caption(f"{opp['북메이커']} • {opp['배당률']}")
                                c2.success(f"**켈리 기준:** {opp['켈리']:.1f}% of 자본")

                                st.markdown("---")

    # 탭 2: 성과 분석
    with tab2:
        st.header("성과 분석")
        st.subheader("누적 ROI & 유닛 획득")
        roi_data = get_roi_data()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roi_data['날짜'], y=roi_data['ROI'], mode='lines+markers', name='ROI %', line=dict(color='#3b82f6', width=3)))
        fig.add_trace(go.Scatter(x=roi_data['날짜'], y=roi_data['유닛'], mode='lines+markers', name='유닛 획득', line=dict(color='#10b981', width=3)))
        fig.update_layout(template='plotly_white', height=400, xaxis_title="날짜", yaxis_title="값")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("주간 성과 분석")
        perf_data = get_performance_data()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=perf_data['주'], y=perf_data['수익'], name='수익 ($)', marker_color='#10b981'))
        fig.add_trace(go.Bar(x=perf_data['주'], y=perf_data['승률'], name='승률 (%)', marker_color='#3b82f6'))
        fig.update_layout(template='plotly_white', height=400, xaxis_title="주", yaxis_title="값")
        st.plotly_chart(fig, use_container_width=True)

    # 탭 3: 모델 정확도
    with tab3:
        st.header("모델 정확도 vs 라스베가스 라인")
        st.markdown("우리 모델은 모든 베팅 유형에서 시장 배당률을 지속적으로 능가합니다")
        acc_data = get_accuracy_data()
        fig = go.Figure()
        fig.add_trace(go.Bar(x=acc_data['카테고리'], y=acc_data['우리모델'], name='우리 모델', marker_color='#10b981'))
        fig.add_trace(go.Bar(x=acc_data['카테고리'], y=acc_data['라스베가스'], name='라스베가스 라인', marker_color='#ef4444'))
        # --- 오류 수정: yaxis_title의 잘못된 백슬래시 제거 ---
        fig.update_layout(template='plotly_white', height=400, xaxis_title="베팅 유형", yaxis_title="정확도 (%)")
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        for i, row in acc_data.iterrows():
            diff = row['우리모델'] - row['라스베가스']
            with [col1, col2, col3, col4][i]:
                st.metric(row['카테고리'], f"+{diff:.1f}%", "우위")

    # 탭 4: 특성 분석
    with tab4:
        st.header("특성 중요도 분석")
        st.markdown("XGBoost 모델에서 예측을 주도하는 핵심 요소들")
        feat_data = get_feature_importance()
        fig = go.Figure(go.Bar(x=feat_data['중요도'], y=feat_data['특성'], orientation='h', marker_color='#8b5cf6'))
        fig.update_layout(template='plotly_white', height=400, xaxis_title="중요도 (%)", yaxis_title="특성")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("모델 상세 정보")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info("**알고리즘:** XGBoost 앙상블")
            st.info("**학습 데이터:** 15,000+ 경기")
        with col2:
            st.info("**사용 특성:** 127개 변수")
            st.info("**업데이트 빈도:** 실시간")
        with col3:
            st.info("**검증 방법:** 시계열 교차 검증")
            st.info("**백테스트 기간:** 2023-24 시즌")

if __name__ == "__main__":
    main()
