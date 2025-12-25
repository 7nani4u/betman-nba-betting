from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, List
import math

@dataclass(frozen=True)
class GameResult:
    home_score: int
    away_score: int

def _split_asian_quarter(line: float) -> List[float]:
    """
    For quarter lines (x.25 or x.75), split into two adjacent half-lines.
    Example:
      -0.25 -> [0.0, -0.5]
      +0.75 -> [+0.5, +1.0]
      1.75 totals -> [1.5, 2.0]
    """
    frac = abs(line) % 1.0
    # handle floating errors
    frac = round(frac, 2)
    if frac not in (0.25, 0.75):
        return [line]
    if frac == 0.25:
        return [math.copysign(math.floor(abs(line)) + 0.0, line),
                math.copysign(math.floor(abs(line)) + 0.5, line)]
    # 0.75
    return [math.copysign(math.floor(abs(line)) + 0.5, line),
            math.copysign(math.floor(abs(line)) + 1.0, line)]

def settle_moneyline(selection: str, result: GameResult, home_team: str, away_team: str) -> float:
    """Return outcome multiplier: 1=win, 0=lose."""
    if selection == "home":
        return 1.0 if result.home_score > result.away_score else 0.0
    if selection == "away":
        return 1.0 if result.away_score > result.home_score else 0.0
    raise ValueError("selection must be 'home' or 'away'")

def settle_spread(selection: str, line: float, result: GameResult) -> float:
    """
    selection: 'home' or 'away'
    line: points handicap applied to selected team.
    Supports quarter/half lines by splitting stake.
    Returns multiplier in {0,0.5,1} for asian-quarter; for push returns 0.0 but should be treated as refund.
    """
    lines = _split_asian_quarter(line)
    mults=[]
    for ln in lines:
        diff = (result.home_score - result.away_score)
        if selection == "home":
            adj = diff + ln
        elif selection == "away":
            adj = (-diff) + ln
        else:
            raise ValueError("selection must be 'home' or 'away'")
        if adj > 0:
            mults.append(1.0)
        elif adj == 0:
            mults.append(None)  # push/refund
        else:
            mults.append(0.0)
    # average the stake split
    # push => refund for that half; represent as None and handle upstream
    return _combine_split_outcomes(mults)

def settle_totals(selection: str, line: float, result: GameResult) -> float:
    """
    selection: 'over' or 'under'
    Supports quarter lines via split (e.g., 227.5 is half-line already).
    Returns combined multiplier with push handling.
    """
    total = result.home_score + result.away_score
    lines = _split_asian_quarter(line)
    mults=[]
    for ln in lines:
        if selection == "over":
            if total > ln:
                mults.append(1.0)
            elif total == ln:
                mults.append(None)
            else:
                mults.append(0.0)
        elif selection == "under":
            if total < ln:
                mults.append(1.0)
            elif total == ln:
                mults.append(None)
            else:
                mults.append(0.0)
        else:
            raise ValueError("selection must be 'over' or 'under'")
    return _combine_split_outcomes(mults)

def settle_odd_even(selection: str, result: GameResult) -> float:
    total = result.home_score + result.away_score
    is_even = (total % 2 == 0)
    if selection == "even":
        return 1.0 if is_even else 0.0
    if selection == "odd":
        return 1.0 if not is_even else 0.0
    raise ValueError("selection must be 'odd' or 'even'")

def _combine_split_outcomes(mults: List[Optional[float]]) -> float:
    """
    Combine split stake outcomes:
      - Win (1), Lose (0), Push (None)
    For a split bet of 2 halves, return:
      Win+Push => 0.5 win (and 0.5 refund)
      Lose+Push => 0.5 lose (and 0.5 refund)
      Win+Lose => 0.5 net win multiplier (0.5)
      Win+Win => 1
      Lose+Lose => 0
      Push+Push => refund only -> 0.0 (caller handles refund)
    Represent as:
      1.0 full win
      0.5 half win (and half refund/lose depending on other half)
      0.0 no win (lose or full push)
    NOTE: Caller must compute PnL with push handling by looking at push count.
    """
    if len(mults)==1:
        return 0.0 if mults[0] is None else float(mults[0])
    # two-way split
    wins = sum(1 for m in mults if m==1.0)
    loses = sum(1 for m in mults if m==0.0)
    pushes = sum(1 for m in mults if m is None)
    # returns win multiplier for the non-refunded portion
    if wins==2:
        return 1.0
    if loses==2:
        return 0.0
    if wins==1 and loses==1:
        return 0.5
    if wins==1 and pushes==1:
        return 0.5
    if loses==1 and pushes==1:
        return 0.0
    # push+push
    return 0.0
