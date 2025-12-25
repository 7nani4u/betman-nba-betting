from __future__ import annotations
import requests
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from .config import settings
from .db import init_db, connect

MARKET_MAP = {
    "h2h": "moneyline",
    "spreads": "spread",
    "totals": "totals",
}

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

class OddsScraper:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.api_key = (api_key or settings.ODDS_API_KEY).strip()
        self.base_url = base_url or settings.ODDS_API_BASE_URL

    def fetch_odds(self, sport: str = "basketball_nba", regions: str = "us",
                   markets: str = "h2h,spreads,totals", odds_format: str = "american") -> List[Dict[str, Any]]:
        if not self.api_key:
            raise RuntimeError("ODDS_API_KEY is empty. Set env ODDS_API_KEY or Streamlit secrets.")
        url = f"{self.base_url}/sports/{sport}/odds/"
        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": markets,
            "oddsFormat": odds_format,
        }
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        return r.json()

    def persist_quotes(self, raw: List[Dict[str, Any]]) -> int:
        """
        Normalize odds into odds_quotes (one row per outcome).
        """
        init_db()
        fetched_at = _utcnow_iso()
        rows = []
        for game in raw:
            event_id = game["id"]
            commence = game["commence_time"]
            home = game["home_team"]
            away = game["away_team"]

            for bookmaker in game.get("bookmakers", []):
                book = bookmaker.get("key", "")
                for market in bookmaker.get("markets", []):
                    mk = market.get("key")
                    if mk not in MARKET_MAP:
                        continue
                    market_name = MARKET_MAP[mk]
                    for outcome in market.get("outcomes", []):
                        name = outcome.get("name")
                        odds = outcome.get("price")
                        line = outcome.get("point")  # spreads/totals
                        if odds is None:
                            continue

                        # selection mapping
                        if market_name == "moneyline":
                            selection = "home" if name == home else "away"
                            line_val = None
                        elif market_name == "spread":
                            selection = "home" if name == home else "away"
                            line_val = float(line) if line is not None else None
                        elif market_name == "totals":
                            selection = str(name).lower()  # "over" / "under"
                            line_val = float(line) if line is not None else None
                        else:
                            continue

                        rows.append((fetched_at, event_id, commence, home, away, book, market_name, selection, line_val, float(odds)))

        if not rows:
            return 0

        with connect() as conn:
            conn.executemany(
                """INSERT INTO odds_quotes
                   (fetched_at,event_id,commence_time,home_team,away_team,bookmaker,market,selection,line,odds_american)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                rows
            )
            conn.commit()
        return len(rows)

def main():
    scraper = OddsScraper()
    raw = scraper.fetch_odds()
    n = scraper.persist_quotes(raw)
    print(f"saved odds_quotes rows: {n}")

if __name__ == "__main__":
    main()
