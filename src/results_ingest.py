from __future__ import annotations
import requests
from datetime import datetime, timezone
from typing import Optional, Dict, Any
from .db import init_db, connect

def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

class ResultsClient:
    """
    Results ingestion stub.

    NBA final scores require a results provider.
    This project leaves provider choice configurable; implement one of:
      - paid sports data APIs (reliable)
      - public endpoints (may change)
    """
    def __init__(self, base_url: Optional[str] = None, api_key: Optional[str] = None):
        self.base_url = base_url
        self.api_key = api_key

    def fetch_game_result(self, event_id: str) -> Optional[Dict[str, Any]]:
        """
        Provider-specific mapping is required.
        Return dict with: completed_at, home_score, away_score.
        """
        return None

def upsert_result(event_id: str, completed_at: str, home_score: int, away_score: int) -> None:
    init_db()
    with connect() as conn:
        conn.execute(
            """INSERT INTO results(event_id, completed_at, home_score, away_score)
               VALUES(?,?,?,?)
               ON CONFLICT(event_id) DO UPDATE SET
                 completed_at=excluded.completed_at,
                 home_score=excluded.home_score,
                 away_score=excluded.away_score
            """,
            (event_id, completed_at, int(home_score), int(away_score))
        )
        conn.commit()
