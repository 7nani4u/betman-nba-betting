import streamlit as st
import pandas as pd
import plotly.express as px
from src.config import settings
from src.db import init_db, connect
from src.edge_finder import EdgeFinder

st.set_page_config(page_title="NBA EV Betting Dashboard", page_icon="🏀", layout="wide")

# Secrets override
if "ODDS_API_KEY" in st.secrets and not settings.ODDS_API_KEY:
    # settings is frozen; show instruction
    st.sidebar.info("ODDS_API_KEY is present in Streamlit secrets. Set env var for CLI scripts if needed.")

init_db()

st.title("🏀 NBA EV Betting Dashboard")

with st.sidebar:
    st.header("설정")
    bankroll = st.number_input("Bankroll (paper)", min_value=0.0, value=1000.0, step=50.0)
    min_edge_pct = st.slider("최소 엣지(%)", 0.0, 15.0, float(settings.MIN_EDGE*100), 0.5)
    st.caption("머니라인(승패)만 end-to-end로 동작하도록 구성되어 있습니다. 다른 시장은 구조만 제공.")

finder = EdgeFinder()
opps = finder.find_latest_opportunities(min_edge=float(min_edge_pct/100.0))

col1, col2, col3 = st.columns(3)
col1.metric("기회 수", len(opps) if not opps.empty else 0)
col2.metric("모델(보정) LogLoss", f"{finder.metrics.get('log_loss_cal','-')}")
col3.metric("모델(보정) Brier", f"{finder.metrics.get('brier_cal','-')}")

st.subheader("+EV 기회")
if opps.empty:
    st.warning("최신 odds 스냅샷이 없거나, 조건을 만족하는 기회가 없습니다. 먼저 odds_scraper를 실행하세요.")
else:
    show = opps.copy()
    show["match"] = show["away_team"] + " @ " + show["home_team"]
    show["pick"] = show["selection"].map({"home":"HOME ML","away":"AWAY ML"})
    show["odds"] = show["odds_american"]
    show["edge(%)"] = (show["edge"]*100).round(2)
    show["EV(%)"] = (show["ev"]*100).round(2)
    show["kelly(%)"] = (show["kelly_frac"]*100).round(2)
    st.dataframe(show[["match","bookmaker","pick","odds","market_prob_no_vig","model_prob_cal","edge(%)","EV(%)","kelly(%)"]], use_container_width=True)

    st.markdown("#### 선택한 행을 Paper Bet으로 저장")
    idx = st.number_input("Row index", min_value=0, max_value=max(0, len(opps)-1), value=0, step=1)
    if st.button("Place paper bet"):
        bet_id = finder.place_bet(opps.iloc[int(idx)], bankroll=float(bankroll))
        st.success(f"저장 완료: bet_id={bet_id}")

st.subheader("베팅 내역")
with connect() as conn:
    bets = pd.read_sql_query("SELECT * FROM bets ORDER BY placed_at DESC LIMIT 500", conn)
if bets.empty:
    st.info("저장된 베팅이 없습니다.")
else:
    st.dataframe(bets, use_container_width=True)

st.subheader("Equity Curve (paper)")
if not bets.empty:
    # naive equity: bankroll_after for placed bets + pnl for settled bets
    # This is a simplified view; production should compute time-aligned equity.
    bets2 = bets.sort_values("placed_at")
    equity = [float(bets2.iloc[0]["bankroll_before"])]
    for _, r in bets2.iterrows():
        equity.append(float(r["bankroll_after"]) + (float(r["pnl"]) if pd.notna(r["pnl"]) else 0.0))
    eq_df = pd.DataFrame({"step": range(len(equity)), "equity": equity})
    fig = px.line(eq_df, x="step", y="equity")
    st.plotly_chart(fig, use_container_width=True)
