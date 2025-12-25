from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Dict, Tuple, Optional
import math

def american_to_decimal(odds: float) -> float:
    """American -> decimal odds."""
    if odds == 0:
        raise ValueError("odds cannot be 0")
    if odds > 0:
        return (odds / 100.0) + 1.0
    return (100.0 / abs(odds)) + 1.0

def american_to_implied_prob(odds: float) -> float:
    """American -> implied probability (includes vig)."""
    if odds == 0:
        raise ValueError("odds cannot be 0")
    if odds > 0:
        return 100.0 / (odds + 100.0)
    return abs(odds) / (abs(odds) + 100.0)

def decimal_to_implied_prob(decimal_odds: float) -> float:
    if decimal_odds <= 1.0:
        raise ValueError("decimal_odds must be > 1")
    return 1.0 / decimal_odds

@dataclass(frozen=True)
class NoVigResult:
    probs: Dict[str, float]          # selection -> no-vig probability
    overround: float                 # sum(implied) - 1
    implied_sum: float               # sum(implied)
    method: str                      # "proportional"

def devig_proportional(selection_to_odds_american: Dict[str, float]) -> NoVigResult:
    """
    Remove bookmaker margin by proportional normalization:
      p_no_vig(i) = p_implied(i) / sum_j p_implied(j)
    Works for 2-way (moneyline/spread/totals/odd-even) and N-way markets.
    """
    if len(selection_to_odds_american) < 2:
        raise ValueError("need at least 2 outcomes to devig")
    implied = {k: american_to_implied_prob(v) for k, v in selection_to_odds_american.items()}
    s = sum(implied.values())
    if s <= 0:
        raise ValueError("invalid implied probability sum")
    probs = {k: v / s for k, v in implied.items()}
    return NoVigResult(probs=probs, overround=s - 1.0, implied_sum=s, method="proportional")

def expected_value_pct(p_win: float, odds_american: float) -> float:
    """
    EV (percentage of stake) for a binary bet:
      EV = p*(d-1) - (1-p)
    """
    d = american_to_decimal(odds_american)
    ev = (p_win * (d - 1.0)) - (1.0 - p_win)
    return ev

def kelly_fraction(p_win: float, odds_american: float, fraction: float = 0.25, cap: float = 0.05) -> float:
    """
    Fractional Kelly for binary bet:
      f* = (p*b - q)/b, where b = d-1
    Apply fraction and cap to reduce overbetting.
    """
    d = american_to_decimal(odds_american)
    b = d - 1.0
    q = 1.0 - p_win
    if b <= 0:
        return 0.0
    f_star = (p_win * b - q) / b
    f_star = max(0.0, f_star)
    f = f_star * fraction
    return min(f, cap)
