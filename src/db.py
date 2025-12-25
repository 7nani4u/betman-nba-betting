from __future__ import annotations
import sqlite3
import os
from contextlib import contextmanager
from typing import Iterator, Optional
from .config import settings

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS odds_quotes (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  fetched_at TEXT NOT NULL,
  event_id TEXT NOT NULL,
  commence_time TEXT NOT NULL,
  home_team TEXT NOT NULL,
  away_team TEXT NOT NULL,
  bookmaker TEXT NOT NULL,
  market TEXT NOT NULL,        -- moneyline|spread|totals|odd_even
  selection TEXT NOT NULL,     -- home|away|over|under|odd|even
  line REAL,                   -- spread/totals line (NULL for moneyline/odd_even)
  odds_american REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_odds_event_market ON odds_quotes(event_id, market, bookmaker, fetched_at);

CREATE TABLE IF NOT EXISTS bets (
  bet_id TEXT PRIMARY KEY,
  placed_at TEXT NOT NULL,
  event_id TEXT NOT NULL,
  commence_time TEXT NOT NULL,
  home_team TEXT NOT NULL,
  away_team TEXT NOT NULL,
  bookmaker TEXT NOT NULL,
  market TEXT NOT NULL,
  selection TEXT NOT NULL,
  line REAL,
  odds_american REAL NOT NULL,
  stake REAL NOT NULL,
  bankroll_before REAL NOT NULL,
  bankroll_after REAL NOT NULL,
  model_prob_raw REAL NOT NULL,
  model_prob_cal REAL NOT NULL,
  market_prob_no_vig REAL NOT NULL,
  edge REAL NOT NULL,
  ev REAL NOT NULL,
  kelly_frac REAL NOT NULL,
  status TEXT NOT NULL,          -- open|settled|void
  result TEXT,                   -- win|lose|push|half_win|half_lose|unknown
  pnl REAL
);

CREATE TABLE IF NOT EXISTS results (
  event_id TEXT PRIMARY KEY,
  completed_at TEXT NOT NULL,
  home_score INTEGER NOT NULL,
  away_score INTEGER NOT NULL
);
"""

def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)

@contextmanager
def connect() -> Iterator[sqlite3.Connection]:
    ensure_parent_dir(settings.DB_PATH)
    conn = sqlite3.connect(settings.DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

def init_db() -> None:
    with connect() as conn:
        conn.executescript(SCHEMA)
        conn.commit()
