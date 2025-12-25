"""
Project configuration.

This project expects ODDS_API_KEY in one of:
- environment variable ODDS_API_KEY
- Streamlit secrets (dashboard) under st.secrets["ODDS_API_KEY"]
"""
from __future__ import annotations
import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    # External APIs
    ODDS_API_BASE_URL: str = "https://api.the-odds-api.com/v4"
    ODDS_API_KEY: str = os.getenv("ODDS_API_KEY", "")

    # DB paths
    DB_PATH: str = os.getenv("NBA_BETTING_DB_PATH", "data/nba_betting.sqlite")

    # Pipeline thresholds
    MIN_EDGE: float = float(os.getenv("MIN_EDGE", "0.03"))   # 3% absolute probability edge
    KELLY_FRACTION: float = float(os.getenv("KELLY_FRACTION", "0.25"))  # fractional Kelly
    MAX_BET_FRACTION: float = float(os.getenv("MAX_BET_FRACTION", "0.05"))  # cap 5% bankroll per bet

    # Model
    MODEL_DIR: str = os.getenv("MODEL_DIR", "models")
    MODEL_NAME: str = os.getenv("MODEL_NAME", "xgb_calibrated.joblib")

settings = Settings()
