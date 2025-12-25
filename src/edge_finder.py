from __future__ import annotations
import os
import uuid
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import joblib
import pandas as pd
import numpy as np
from .config import settings
from .db import init_db, connect
from .probability import devig_proportional, expected_value_pct, kelly_fraction, american_to_decimal

@dataclass(frozen=True)
class Opportunity:
    event_id: str
    commence_time: str
    home_team: str
    away_team: str
    bookmaker: str
    market: str
    selection: str
    line: Optional[float]
    odds_american: float
    market_prob_no_vig: float
    model_prob_raw: float
    model_prob_cal: float
    edge: float
    ev: float
    kelly_frac: float

class EdgeFinder:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path or os.path.join(settings.MODEL_DIR, settings.MODEL_NAME)
        self.base_model = None
        self.calibrator = None
        self.feature_names: list[str] = []
        self.metrics: dict = {}
        self._load_model()

    def _load_model(self) -> None:
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"model not found: {self.model_path}")
        data = joblib.load(self.model_path)
        self.base_model = data["base_model"]
        self.calibrator = data["calibrator"]
        self.feature_names = data["feature_names"]
        self.metrics = data.get("metrics", {})

    def _build_features_for_game(self, row: pd.Series) -> pd.DataFrame:
        """
        Placeholder: Real implementation should join team stats, injuries, rest, pace etc.
        For now, we use minimal features from the synthetic generator (if available).
        """
        # If you have a features table keyed by (event_id, team), join here.
        # Minimal fallback: zeros for missing columns to keep pipeline executable.
        feat = {c: 0.0 for c in self.feature_names}
        feat["is_home"] = 1.0
        return pd.DataFrame([feat])

    def find_latest_opportunities(self, min_edge: float = settings.MIN_EDGE) -> pd.DataFrame:
        init_db()
        with connect() as conn:
            # latest snapshot by fetched_at (global max)
            q = """
            WITH latest AS (SELECT MAX(fetched_at) AS t FROM odds_quotes)
            SELECT * FROM odds_quotes
            WHERE fetched_at = (SELECT t FROM latest)
            """
            quotes = pd.read_sql_query(q, conn)

        if quotes.empty:
            return pd.DataFrame()

        opps: List[Opportunity] = []

        # group for devig: event+book+market+line
        grp_cols = ["event_id", "bookmaker", "market", "line"]
        for (event_id, book, market, line), g in quotes.groupby(grp_cols, dropna=False):
            # build selection->odds mapping
            sel_to_odds = {r["selection"]: float(r["odds_american"]) for _, r in g.iterrows()}
            if len(sel_to_odds) < 2:
                continue
            nv = devig_proportional(sel_to_odds)

            # predict for each selection
            for _, r in g.iterrows():
                features = self._build_features_for_game(r)
                p_raw = float(self.base_model.predict_proba(features)[:, 1][0])
                p_cal = float(self.calibrator.predict_proba(features)[:, 1][0])

                # For non-moneyline markets, p_cal must correspond to "selection happens" probability.
                # This requires market-specific training. For now, we only support moneyline(home) as an executable baseline.
                if market != "moneyline":
                    continue

                # map selection prob: if model predicts home-win, then away-win prob is 1-p
                if r["selection"] == "home":
                    p_raw_sel, p_cal_sel = p_raw, p_cal
                else:
                    p_raw_sel, p_cal_sel = (1.0 - p_raw), (1.0 - p_cal)

                market_p = nv.probs.get(r["selection"])
                if market_p is None:
                    continue

                edge = p_cal_sel - market_p
                if edge < min_edge:
                    continue

                ev = expected_value_pct(p_cal_sel, float(r["odds_american"]))
                kf = kelly_fraction(p_cal_sel, float(r["odds_american"]),
                                    fraction=settings.KELLY_FRACTION, cap=settings.MAX_BET_FRACTION)

                opps.append(Opportunity(
                    event_id=event_id,
                    commence_time=r["commence_time"],
                    home_team=r["home_team"],
                    away_team=r["away_team"],
                    bookmaker=book,
                    market=market,
                    selection=r["selection"],
                    line=None if pd.isna(r["line"]) else float(r["line"]),
                    odds_american=float(r["odds_american"]),
                    market_prob_no_vig=float(market_p),
                    model_prob_raw=float(p_raw_sel),
                    model_prob_cal=float(p_cal_sel),
                    edge=float(edge),
                    ev=float(ev),
                    kelly_frac=float(kf),
                ))

        df = pd.DataFrame([o.__dict__ for o in opps])
        if not df.empty:
            df["edge_pct"] = df["edge"] * 100.0
            df["ev_pct"] = df["ev"] * 100.0
            df["kelly_pct"] = df["kelly_frac"] * 100.0
        return df.sort_values(["edge", "ev"], ascending=False)

    def place_bet(self, opp_row: pd.Series, bankroll: float) -> str:
        """
        Persist a bet decision (paper trading).
        Stake = bankroll * kelly_frac
        """
        init_db()
        stake = bankroll * float(opp_row["kelly_frac"])
        bet_id = str(uuid.uuid4())
        bankroll_after = bankroll - stake

        with connect() as conn:
            conn.execute(
                """INSERT INTO bets
                (bet_id, placed_at, event_id, commence_time, home_team, away_team, bookmaker, market, selection, line,
                 odds_american, stake, bankroll_before, bankroll_after, model_prob_raw, model_prob_cal, market_prob_no_vig,
                 edge, ev, kelly_frac, status)
                 VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    bet_id,
                    pd.Timestamp.utcnow().isoformat(),
                    str(opp_row["event_id"]),
                    str(opp_row["commence_time"]),
                    str(opp_row["home_team"]),
                    str(opp_row["away_team"]),
                    str(opp_row["bookmaker"]),
                    str(opp_row["market"]),
                    str(opp_row["selection"]),
                    None,
                    float(opp_row["odds_american"]),
                    float(stake),
                    float(bankroll),
                    float(bankroll_after),
                    float(opp_row["model_prob_raw"]),
                    float(opp_row["model_prob_cal"]),
                    float(opp_row["market_prob_no_vig"]),
                    float(opp_row["edge"]),
                    float(opp_row["ev"]),
                    float(opp_row["kelly_frac"]),
                    "open",
                )
            )
            conn.commit()
        return bet_id

def main():
    finder = EdgeFinder()
    df = finder.find_latest_opportunities()
    print(df.head(10).to_string(index=False))

if __name__ == "__main__":
    main()
