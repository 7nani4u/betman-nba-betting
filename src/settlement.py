from __future__ import annotations
import pandas as pd
from .db import init_db, connect
from .markets import GameResult, settle_moneyline, settle_spread, settle_totals, settle_odd_even
from .probability import american_to_decimal

def settle_open_bets() -> int:
    """
    Settle open bets if result exists.
    For now, only moneyline settlement is implemented end-to-end.
    """
    init_db()
    with connect() as conn:
        open_bets = pd.read_sql_query("SELECT * FROM bets WHERE status='open'", conn)
        if open_bets.empty:
            return 0
        results = pd.read_sql_query("SELECT * FROM results", conn)

    if results.empty:
        return 0

    res_map = {r["event_id"]: r for _, r in results.iterrows()}
    settled = 0

    with connect() as conn:
        for _, b in open_bets.iterrows():
            r = res_map.get(b["event_id"])
            if r is None:
                continue
            gr = GameResult(home_score=int(r["home_score"]), away_score=int(r["away_score"]))

            market = b["market"]
            selection = b["selection"]
            line = b["line"]
            odds = float(b["odds_american"])
            stake = float(b["stake"])

            if market == "moneyline":
                win_mult = settle_moneyline(selection, gr, b["home_team"], b["away_team"])
                if win_mult == 1.0:
                    payout = stake * american_to_decimal(odds)
                    pnl = payout - stake
                    result = "win"
                    status = "settled"
                else:
                    pnl = -stake
                    result = "lose"
                    status = "settled"
            else:
                # not implemented fully
                continue

            conn.execute(
                "UPDATE bets SET status=?, result=?, pnl=? WHERE bet_id=?",
                (status, result, float(pnl), b["bet_id"])
            )
            settled += 1
        conn.commit()

    return settled

if __name__ == "__main__":
    n = settle_open_bets()
    print("settled:", n)
