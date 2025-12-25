# NBA EV Betting (v2 refactor)

This refactor introduces:
- a standardized **no-vig** probability layer (single function)
- an explicit **calibration** step (CalibratedClassifierCV + TimeSeriesSplit)
- a normalized **SQLite** schema for odds quotes and bets
- a paper-trading loop (find opportunities -> Kelly sizing -> store bet -> settle)

> Note: Only moneyline is implemented end-to-end. Spread/totals/odd-even scaffolding is included.

## Quickstart

```bash
pip install -r requirements.txt

# 1) generate synthetic historical data (replace with real data later)
python -m src.data_generator

# 2) train + calibrate model
python -m src.model_training

# 3) scrape odds and persist to sqlite
export ODDS_API_KEY="..."
python -m src.odds_scraper

# 4) find +EV opportunities
python -m src.edge_finder

# 5) run dashboard
streamlit run dashboard.py
```
