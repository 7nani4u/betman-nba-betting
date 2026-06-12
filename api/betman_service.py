"""Betman dashboard data collection and analysis helpers."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

import plotly
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup


@dataclass
class OddsLine:
    """Normalized decimal odds for a match."""

    home: Optional[float]
    draw: Optional[float]
    away: Optional[float]


@dataclass
class TeamMetrics:
    """Model inputs derived from raw feed data or deterministic fallbacks."""

    power_rating: float
    recent_form: float
    venue_index: float
    head_to_head_index: float
    attacking_index: float
    defensive_index: float
    standings_index: float
    availability_index: float


@dataclass
class MatchRecord:
    """Canonical match record shared by collection, analysis and UI layers."""

    match_id: str
    sport: str
    league: str
    league_name: str
    round_name: str
    kickoff: str
    home_team: str
    away_team: str
    status: str
    source: str
    source_label: str
    updated_at: str
    odds: OddsLine
    home_metrics: TeamMetrics
    away_metrics: TeamMetrics


def _now_iso() -> str:
    """Return an ISO8601 UTC timestamp."""

    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> Optional[float]:
    """Convert a mixed value into a float when possible."""

    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    for token in ("%", ",", "배", "x"):
        text = text.replace(token, "")

    try:
        return float(text)
    except ValueError:
        return None


def _parse_datetime(value: Any) -> Optional[datetime]:
    """Parse several common feed date formats into timezone-aware datetimes."""

    if not value:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)

    text = str(value).strip()
    if not text:
        return None

    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    except ValueError:
        pass

    formats = [
        "%Y-%m-%d %H:%M",
        "%Y/%m/%d %H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S",
        "%m/%d/%Y %H:%M",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    return None


def _title_from_key(key: str) -> str:
    """Convert feed keys into readable labels."""

    return key.replace("_", " ").strip().title()


def _stable_metric(name: str, channel: str, minimum: float = 45.0, maximum: float = 84.0) -> float:
    """Generate deterministic fallback team metrics when source data is missing."""

    raw = hashlib.sha1(f"{name}:{channel}".encode("utf-8")).hexdigest()
    ratio = int(raw[:8], 16) / 0xFFFFFFFF
    return round(minimum + (maximum - minimum) * ratio, 1)


def _safe_path_lookup(payload: Any, path: str) -> Any:
    """Walk a dotted path inside arbitrary JSON data."""

    current = payload
    for part in path.split("."):
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current


def _figure_to_json(figure: go.Figure) -> Dict[str, Any]:
    """Serialize Plotly figures for Flask templates and JSON APIs."""

    return json.loads(json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder))


def _clamp(value: float, minimum: float, maximum: float) -> float:
    """Clamp a float into the configured range."""

    return max(minimum, min(value, maximum))


def _normalize_implied_probabilities(odds: OddsLine) -> Dict[str, Optional[float]]:
    """Normalize decimal odds into de-vigged market probabilities."""

    implied = {
        "home": (1 / odds.home) if odds.home and odds.home > 1 else None,
        "draw": (1 / odds.draw) if odds.draw and odds.draw > 1 else None,
        "away": (1 / odds.away) if odds.away and odds.away > 1 else None,
    }
    values = [value for value in implied.values() if value is not None]
    if not values:
        return {"home": None, "draw": None, "away": None}

    total = sum(values)
    return {
        outcome: (value / total if value is not None else None)
        for outcome, value in implied.items()
    }


class BetmanCollector:
    """Collect match rows from sample, HTML or JSON sources with TTL caching."""

    HEADER_ALIASES = {
        "sport": ("sport", "sports", "종목", "game_type"),
        "league": ("league", "league_code", "리그"),
        "league_name": ("league_name", "league_nm", "리그명"),
        "round_name": ("round", "round_name", "회차", "게임"),
        "kickoff": ("kickoff", "match_time", "start_time", "경기일시", "경기시간", "일시"),
        "home_team": ("home", "home_team", "home_name", "홈팀"),
        "away_team": ("away", "away_team", "away_name", "원정팀"),
        "status": ("status", "state", "상태"),
        "match_id": ("match_id", "id", "game_id", "경기번호"),
        "updated_at": ("updated_at", "last_updated", "수집시각"),
        "odds_home": ("odds_home", "home_odds", "배당_home", "home_price", "홈배당"),
        "odds_draw": ("odds_draw", "draw_odds", "배당_draw", "draw_price", "무배당"),
        "odds_away": ("odds_away", "away_odds", "배당_away", "away_price", "원정배당"),
    }

    TEAM_METRIC_ALIASES = {
        "power_rating": ("power_rating", "rating", "전력지수"),
        "recent_form": ("recent_form", "폼지수", "최근성적"),
        "venue_index": ("venue_index", "home_away_index", "홈원정지수"),
        "head_to_head_index": ("head_to_head_index", "상대전적지수"),
        "attacking_index": ("attacking_index", "득점추세", "공격지수"),
        "defensive_index": ("defensive_index", "실점억제", "수비지수"),
        "standings_index": ("standings_index", "순위지수"),
        "availability_index": ("availability_index", "부상가용성", "선수가용성"),
    }

    SPORT_LABELS = {
        "basketball": "농구",
        "soccer": "축구",
        "baseball": "야구",
        "football": "미식축구",
        "ice hockey": "아이스하키",
        "hockey": "아이스하키",
    }

    def __init__(self) -> None:
        self.session = requests.Session()
        self.cache_ttl = int(os.getenv("BETMAN_CACHE_TTL", "90"))
        self.request_timeout = int(os.getenv("BETMAN_REQUEST_TIMEOUT", "12"))
        self._cache: Dict[str, Any] = {}

    def collect_matches(self, force_refresh: bool = False) -> Tuple[List[MatchRecord], Dict[str, Any]]:
        """Collect and normalize raw match rows, using cached data when possible."""

        now = datetime.now(timezone.utc)
        cached_until = self._cache.get("expires_at")
        if not force_refresh and cached_until and cached_until > now:
            payload = copy.deepcopy(self._cache["payload"])
            payload["status"]["cache_hit"] = True
            return payload["matches"], payload["status"]

        warnings: List[str] = []
        source_mode = os.getenv("BETMAN_SOURCE_MODE", "sample").strip().lower() or "sample"
        source_label = "샘플 데이터"
        raw_rows: List[Dict[str, Any]] = []

        try:
            if source_mode == "json":
                raw_rows = self._collect_from_json()
                source_label = "JSON 피드"
            elif source_mode == "html":
                raw_rows = self._collect_from_html()
                source_label = "HTML 페이지"
            else:
                raw_rows = self._sample_rows()
        except Exception as exc:
            warnings.append(f"원격 수집에 실패하여 샘플 데이터로 대체했습니다: {exc}")
            raw_rows = self._sample_rows()
            source_mode = "sample-fallback"
            source_label = "샘플 데이터"

        normalized = [record for row in raw_rows if (record := self._normalize_row(row, source_label))]
        deduped, duplicate_count = self._deduplicate(normalized)

        missing_odds_count = sum(
            1
            for match in deduped
            if not any((match.odds.home, match.odds.draw, match.odds.away))
        )

        status = {
            "source_mode": source_mode,
            "source_label": source_label,
            "raw_count": len(raw_rows),
            "normalized_count": len(normalized),
            "deduplicated_count": len(deduped),
            "duplicate_count": duplicate_count,
            "missing_odds_count": missing_odds_count,
            "cache_hit": False,
            "warnings": warnings,
            "collected_at": _now_iso(),
        }

        payload = {"matches": deduped, "status": status}
        self._cache = {
            "expires_at": now + timedelta(seconds=self.cache_ttl),
            "payload": copy.deepcopy(payload),
        }
        return deduped, status

    def _collect_from_json(self) -> List[Dict[str, Any]]:
        """Fetch a JSON feed and extract rows from a configurable root path."""

        url = os.getenv("BETMAN_JSON_URL", "").strip()
        if not url:
            raise ValueError("BETMAN_JSON_URL 환경변수가 비어 있습니다.")

        response = self.session.get(
            url,
            timeout=self.request_timeout,
            headers={"User-Agent": os.getenv("BETMAN_USER_AGENT", "Mozilla/5.0 BetmanDashboard/1.0")},
        )
        response.raise_for_status()
        payload = response.json()

        root_path = os.getenv("BETMAN_JSON_ROOT", "").strip()
        rows = _safe_path_lookup(payload, root_path) if root_path else payload
        if not isinstance(rows, list):
            raise ValueError("BETMAN_JSON_ROOT가 리스트 데이터를 가리키지 않습니다.")
        return [row for row in rows if isinstance(row, dict)]

    def _collect_from_html(self) -> List[Dict[str, Any]]:
        """Fetch an HTML page and extract table rows using loose heuristics."""

        url = os.getenv("BETMAN_HTML_URL", "").strip()
        if not url:
            raise ValueError("BETMAN_HTML_URL 환경변수가 비어 있습니다.")

        response = self.session.get(
            url,
            timeout=self.request_timeout,
            headers={"User-Agent": os.getenv("BETMAN_USER_AGENT", "Mozilla/5.0 BetmanDashboard/1.0")},
        )
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")
        selector = os.getenv("BETMAN_HTML_ROW_SELECTOR", "table tr")
        rows = soup.select(selector)
        extracted: List[Dict[str, Any]] = []
        current_headers: List[str] = []

        for row in rows:
            headers = [cell.get_text(" ", strip=True) for cell in row.find_all("th")]
            cells = [cell.get_text(" ", strip=True) for cell in row.find_all("td")]

            if headers and not cells:
                current_headers = headers
                continue

            if not cells:
                continue

            if current_headers and len(current_headers) == len(cells):
                extracted.append(dict(zip(current_headers, cells)))
                continue

            if len(cells) >= 7:
                extracted.append(
                    {
                        "sport": cells[0],
                        "league_name": cells[1],
                        "round_name": cells[2],
                        "kickoff": cells[3],
                        "home_team": cells[4],
                        "away_team": cells[5],
                        "odds_home": cells[6],
                        "odds_draw": cells[7] if len(cells) > 8 else None,
                        "odds_away": cells[8] if len(cells) > 8 else cells[7] if len(cells) > 7 else None,
                    }
                )

        if not extracted:
            raise ValueError("HTML에서 경기 행을 찾지 못했습니다.")
        return extracted

    def _normalize_row(self, row: Dict[str, Any], source_label: str) -> Optional[MatchRecord]:
        """Convert a raw row into the canonical match record shape."""

        sport = self._first_value(row, "sport") or "기타"
        sport = self.SPORT_LABELS.get(str(sport).strip().lower(), str(sport).strip())

        league = self._first_value(row, "league") or self._first_value(row, "league_name") or "UNKNOWN"
        league_name = self._first_value(row, "league_name") or _title_from_key(str(league))
        round_name = self._first_value(row, "round_name") or "일반"
        home_team = self._first_value(row, "home_team")
        away_team = self._first_value(row, "away_team")
        if not home_team or not away_team:
            return None

        kickoff_dt = _parse_datetime(self._first_value(row, "kickoff")) or datetime.now(timezone.utc)
        updated_at = _parse_datetime(self._first_value(row, "updated_at")) or datetime.now(timezone.utc)

        match_id = self._first_value(row, "match_id")
        if not match_id:
            match_id = hashlib.sha1(
                f"{sport}|{league}|{kickoff_dt.isoformat()}|{home_team}|{away_team}".encode("utf-8")
            ).hexdigest()[:16]

        home_metrics = self._team_metrics(row, "home", home_team)
        away_metrics = self._team_metrics(row, "away", away_team)

        return MatchRecord(
            match_id=str(match_id),
            sport=str(sport),
            league=str(league),
            league_name=str(league_name),
            round_name=str(round_name),
            kickoff=kickoff_dt.isoformat(),
            home_team=str(home_team),
            away_team=str(away_team),
            status=str(self._first_value(row, "status") or "발매중"),
            source=os.getenv("BETMAN_SOURCE_MODE", "sample"),
            source_label=source_label,
            updated_at=updated_at.isoformat(),
            odds=OddsLine(
                home=_safe_float(self._first_value(row, "odds_home")),
                draw=_safe_float(self._first_value(row, "odds_draw")),
                away=_safe_float(self._first_value(row, "odds_away")),
            ),
            home_metrics=home_metrics,
            away_metrics=away_metrics,
        )

    def _team_metrics(self, row: Dict[str, Any], side: str, team_name: str) -> TeamMetrics:
        """Build or derive a full feature vector for a team."""

        def value(metric_key: str, channel: str) -> float:
            aliases = []
            for alias in self.TEAM_METRIC_ALIASES[metric_key]:
                aliases.extend((f"{side}_{alias}", f"{side}{alias}", f"{alias}_{side}"))

            for alias in aliases:
                if alias in row:
                    parsed = _safe_float(row.get(alias))
                    if parsed is not None:
                        return round(parsed, 1)
            return _stable_metric(team_name, channel)

        return TeamMetrics(
            power_rating=value("power_rating", "power"),
            recent_form=value("recent_form", "recent"),
            venue_index=value("venue_index", "venue"),
            head_to_head_index=value("head_to_head_index", "h2h"),
            attacking_index=value("attacking_index", "attack"),
            defensive_index=value("defensive_index", "defense"),
            standings_index=value("standings_index", "standing"),
            availability_index=value("availability_index", "availability"),
        )

    def _first_value(self, row: Dict[str, Any], canonical_key: str) -> Any:
        """Return the first available value among known aliases."""

        for alias in self.HEADER_ALIASES.get(canonical_key, (canonical_key,)):
            if alias in row and row.get(alias) not in ("", None):
                return row.get(alias)
        return None

    def _deduplicate(self, matches: Iterable[MatchRecord]) -> Tuple[List[MatchRecord], int]:
        """Keep only the freshest copy for each logical match."""

        unique_matches: Dict[str, MatchRecord] = {}
        duplicate_count = 0

        for match in matches:
            key = f"{match.sport}|{match.league}|{match.kickoff}|{match.home_team}|{match.away_team}"
            existing = unique_matches.get(key)
            if not existing:
                unique_matches[key] = match
                continue

            duplicate_count += 1
            if match.updated_at > existing.updated_at:
                unique_matches[key] = match

        ordered = sorted(unique_matches.values(), key=lambda item: item.kickoff)
        return ordered, duplicate_count

    def _sample_rows(self) -> List[Dict[str, Any]]:
        """Return deterministic seed data that mirrors the production schema."""

        return [
            {
                "match_id": "btm-001",
                "sport": "농구",
                "league": "NBA",
                "league_name": "미국 NBA",
                "round_name": "프로토 승부식",
                "kickoff": "2026-06-13T11:30:00+09:00",
                "home_team": "Lakers",
                "away_team": "Celtics",
                "status": "발매중",
                "odds_home": 1.82,
                "odds_away": 2.03,
                "home_power_rating": 83,
                "away_power_rating": 79,
                "home_recent_form": 76,
                "away_recent_form": 68,
                "home_venue_index": 74,
                "away_venue_index": 58,
                "home_head_to_head_index": 63,
                "away_head_to_head_index": 52,
                "home_attacking_index": 81,
                "away_attacking_index": 77,
                "home_defensive_index": 72,
                "away_defensive_index": 67,
                "home_standings_index": 78,
                "away_standings_index": 72,
                "home_availability_index": 75,
                "away_availability_index": 69,
                "updated_at": "2026-06-12T02:20:00+09:00",
            },
            {
                "match_id": "btm-002",
                "sport": "축구",
                "league": "KLEAGUE1",
                "league_name": "K리그1",
                "round_name": "축구토토 승무패",
                "kickoff": "2026-06-13T19:00:00+09:00",
                "home_team": "울산",
                "away_team": "전북",
                "status": "발매중",
                "odds_home": 2.12,
                "odds_draw": 3.24,
                "odds_away": 3.44,
                "home_power_rating": 79,
                "away_power_rating": 76,
                "home_recent_form": 72,
                "away_recent_form": 65,
                "home_venue_index": 77,
                "away_venue_index": 54,
                "home_head_to_head_index": 62,
                "away_head_to_head_index": 50,
                "home_attacking_index": 74,
                "away_attacking_index": 68,
                "home_defensive_index": 73,
                "away_defensive_index": 66,
                "home_standings_index": 82,
                "away_standings_index": 69,
                "home_availability_index": 71,
                "away_availability_index": 67,
                "updated_at": "2026-06-12T01:55:00+09:00",
            },
            {
                "match_id": "btm-003",
                "sport": "야구",
                "league": "KBO",
                "league_name": "KBO 리그",
                "round_name": "프로토 승부식",
                "kickoff": "2026-06-13T18:30:00+09:00",
                "home_team": "LG",
                "away_team": "두산",
                "status": "발매중",
                "odds_home": 1.76,
                "odds_away": 2.15,
                "home_power_rating": 81,
                "away_power_rating": 75,
                "home_recent_form": 78,
                "away_recent_form": 62,
                "home_venue_index": 73,
                "away_venue_index": 57,
                "home_head_to_head_index": 66,
                "away_head_to_head_index": 48,
                "home_attacking_index": 80,
                "away_attacking_index": 71,
                "home_defensive_index": 76,
                "away_defensive_index": 61,
                "home_standings_index": 84,
                "away_standings_index": 64,
                "home_availability_index": 74,
                "away_availability_index": 63,
                "updated_at": "2026-06-12T02:05:00+09:00",
            },
            {
                "match_id": "btm-004",
                "sport": "축구",
                "league": "EPL",
                "league_name": "잉글랜드 프리미어리그",
                "round_name": "프로토 승부식",
                "kickoff": "2026-06-14T04:00:00+09:00",
                "home_team": "Arsenal",
                "away_team": "Chelsea",
                "status": "발매중",
                "odds_home": 2.04,
                "odds_draw": 3.38,
                "odds_away": 3.72,
                "home_power_rating": 84,
                "away_power_rating": 77,
                "home_recent_form": 79,
                "away_recent_form": 69,
                "home_venue_index": 76,
                "away_venue_index": 55,
                "home_head_to_head_index": 58,
                "away_head_to_head_index": 49,
                "home_attacking_index": 82,
                "away_attacking_index": 71,
                "home_defensive_index": 77,
                "away_defensive_index": 68,
                "home_standings_index": 83,
                "away_standings_index": 70,
                "home_availability_index": 74,
                "away_availability_index": 66,
                "updated_at": "2026-06-12T02:10:00+09:00",
            },
            {
                "match_id": "btm-005",
                "sport": "미식축구",
                "league": "NFL",
                "league_name": "미국 NFL",
                "round_name": "프로토 승부식",
                "kickoff": "2026-06-15T09:20:00+09:00",
                "home_team": "Chiefs",
                "away_team": "Bills",
                "status": "발매예정",
                "odds_home": 1.91,
                "odds_away": 1.95,
                "home_power_rating": 86,
                "away_power_rating": 82,
                "home_recent_form": 81,
                "away_recent_form": 76,
                "home_venue_index": 74,
                "away_venue_index": 61,
                "home_head_to_head_index": 60,
                "away_head_to_head_index": 53,
                "home_attacking_index": 84,
                "away_attacking_index": 80,
                "home_defensive_index": 75,
                "away_defensive_index": 70,
                "home_standings_index": 86,
                "away_standings_index": 79,
                "home_availability_index": 77,
                "away_availability_index": 71,
                "updated_at": "2026-06-12T01:40:00+09:00",
            },
            {
                "match_id": "btm-006",
                "sport": "아이스하키",
                "league": "NHL",
                "league_name": "미국 NHL",
                "round_name": "프로토 승부식",
                "kickoff": "2026-06-15T08:00:00+09:00",
                "home_team": "Bruins",
                "away_team": "Rangers",
                "status": "발매중",
                "odds_home": 2.18,
                "odds_away": 1.79,
                "home_power_rating": 74,
                "away_power_rating": 79,
                "home_recent_form": 67,
                "away_recent_form": 75,
                "home_venue_index": 71,
                "away_venue_index": 60,
                "home_head_to_head_index": 49,
                "away_head_to_head_index": 58,
                "home_attacking_index": 69,
                "away_attacking_index": 76,
                "home_defensive_index": 71,
                "away_defensive_index": 74,
                "home_standings_index": 70,
                "away_standings_index": 79,
                "home_availability_index": 68,
                "away_availability_index": 75,
                "updated_at": "2026-06-12T02:00:00+09:00",
            },
            {
                "match_id": "btm-006-dup",
                "sport": "아이스하키",
                "league": "NHL",
                "league_name": "미국 NHL",
                "round_name": "프로토 승부식",
                "kickoff": "2026-06-15T08:00:00+09:00",
                "home_team": "Bruins",
                "away_team": "Rangers",
                "status": "발매중",
                "odds_home": 2.12,
                "odds_away": 1.82,
                "updated_at": "2026-06-12T02:12:00+09:00",
            },
        ]


class MatchAnalyzer:
    """Generate probabilities, edges and explanation-friendly evidence."""

    BASE_SCORE_BY_SPORT = {
        "농구": 108.0,
        "축구": 1.42,
        "야구": 4.75,
        "미식축구": 24.2,
        "아이스하키": 2.95,
    }

    FACTOR_WEIGHTS = {
        "전력 차이": 0.18,
        "최근 경기 흐름": 0.13,
        "홈/원정 적합도": 0.10,
        "상대 전적": 0.06,
        "공격 생산성": 0.12,
        "수비 안정성": 0.10,
        "득점 추세": 0.08,
        "실점 억제": 0.07,
        "리그 순위": 0.05,
        "선수 가용성": 0.04,
        "일정 휴식": 0.03,
        "이동 피로": 0.02,
        "경기 일관성": 0.02,
    }

    def analyze(self, match: MatchRecord) -> Dict[str, Any]:
        """Analyze a single match and prepare a UI-ready card."""

        market_probs = _normalize_implied_probabilities(match.odds)
        derived_context = self._derive_context(match)
        contributions = self._build_contributions(match, derived_context)
        weighted_score = sum(item["impact"] for item in contributions)
        projected = self._project_match_script(match, derived_context, weighted_score)
        raw_model_probs = self._build_model_probabilities(
            match,
            market_probs,
            weighted_score,
            projected["projected_margin"],
            derived_context,
        )
        quality = self._quality_snapshot(match, market_probs, weighted_score, raw_model_probs, derived_context)

        blended_probs = {
            outcome: self._blend_probability(
                raw_model_probs.get(outcome),
                market_probs.get(outcome),
                quality["blend_weight"],
            )
            for outcome in ("home", "draw", "away")
        }
        blended_probs = self._renormalize(blended_probs)

        probability_bands = self._build_probability_bands(blended_probs, quality["volatility_score"])
        outcome_rankings = self._rank_outcomes(match, blended_probs, market_probs, quality, probability_bands)
        recommendation = outcome_rankings[0] if outcome_rankings else self._empty_recommendation()
        top_reasons = self._top_reasons(contributions)
        evidence = self._serialize_evidence(contributions)

        return {
            "match_id": match.match_id,
            "sport": match.sport,
            "league": match.league,
            "league_name": match.league_name,
            "round_name": match.round_name,
            "kickoff": match.kickoff,
            "home_team": match.home_team,
            "away_team": match.away_team,
            "status": match.status,
            "source_label": match.source_label,
            "updated_at": match.updated_at,
            "odds": asdict(match.odds),
            "market_probabilities": self._round_probabilities(market_probs),
            "raw_model_probabilities": self._round_probabilities(raw_model_probs),
            "model_probabilities": self._round_probabilities(blended_probs),
            "probability_bands": probability_bands,
            "recommendation": recommendation,
            "confidence_label": self._confidence_label(recommendation["edge_percent"], recommendation["expected_value_percent"]),
            "top_reasons": top_reasons,
            "evidence": evidence,
            "quality": quality,
            "projection": projected,
            "derived_context": derived_context,
            "outcome_rankings": outcome_rankings,
            "team_metrics": {
                "home": asdict(match.home_metrics),
                "away": asdict(match.away_metrics),
            },
        }

    def _value_for_label(self, metrics: TeamMetrics, label: str) -> float:
        """Map a readable factor label back to the stored metric."""

        mapping = {
            "전력 차이": metrics.power_rating,
            "최근 경기 흐름": metrics.recent_form,
            "홈/원정 적합도": metrics.venue_index,
            "상대 전적": metrics.head_to_head_index,
            "공격 생산성": metrics.attacking_index,
            "수비 안정성": metrics.defensive_index,
            "리그 순위": metrics.standings_index,
            "선수 가용성": metrics.availability_index,
        }
        return mapping[label]

    def _derive_context(self, match: MatchRecord) -> Dict[str, Dict[str, float]]:
        """Derive additional schedule and volatility signals from stable inputs."""

        def build(team_name: str, metrics: TeamMetrics) -> Dict[str, float]:
            return {
                "rest_index": round((metrics.recent_form * 0.45) + (_stable_metric(team_name, "rest", 48, 92) * 0.55), 1),
                "travel_index": round((metrics.venue_index * 0.35) + (_stable_metric(team_name, "travel", 42, 88) * 0.65), 1),
                "consistency_index": round(
                    (metrics.recent_form * 0.40) + (metrics.defensive_index * 0.35) + (metrics.power_rating * 0.25), 1
                ),
                "scoring_trend": round((metrics.attacking_index * 0.62) + (metrics.recent_form * 0.38), 1),
                "conceding_control": round((metrics.defensive_index * 0.68) + (metrics.availability_index * 0.32), 1),
                "pace_index": round((_stable_metric(team_name, "pace", 46, 88) * 0.55) + (metrics.attacking_index * 0.45), 1),
                "motivation_index": round((metrics.standings_index * 0.52) + (_stable_metric(team_name, "motivation", 54, 94) * 0.48), 1),
            }

        home = build(match.home_team, match.home_metrics)
        away = build(match.away_team, match.away_metrics)
        return {"home": home, "away": away}

    def _build_contributions(self, match: MatchRecord, context: Dict[str, Dict[str, float]]) -> List[Dict[str, float]]:
        """Build weighted factors used by the prediction model."""

        home = match.home_metrics
        away = match.away_metrics

        factors = [
            ("전력 차이", home.power_rating, away.power_rating, home.power_rating - away.power_rating),
            ("최근 경기 흐름", home.recent_form, away.recent_form, home.recent_form - away.recent_form),
            ("홈/원정 적합도", home.venue_index, away.venue_index, home.venue_index - away.venue_index),
            ("상대 전적", home.head_to_head_index, away.head_to_head_index, home.head_to_head_index - away.head_to_head_index),
            (
                "공격 생산성",
                round((home.attacking_index + context["home"]["scoring_trend"]) / 2, 1),
                round((away.defensive_index + context["away"]["conceding_control"]) / 2, 1),
                ((home.attacking_index + context["home"]["scoring_trend"]) / 2)
                - ((away.defensive_index + context["away"]["conceding_control"]) / 2),
            ),
            (
                "수비 안정성",
                round((home.defensive_index + context["home"]["conceding_control"]) / 2, 1),
                round((away.attacking_index + context["away"]["scoring_trend"]) / 2, 1),
                ((home.defensive_index + context["home"]["conceding_control"]) / 2)
                - ((away.attacking_index + context["away"]["scoring_trend"]) / 2),
            ),
            ("득점 추세", context["home"]["scoring_trend"], context["away"]["scoring_trend"], context["home"]["scoring_trend"] - context["away"]["scoring_trend"]),
            (
                "실점 억제",
                context["home"]["conceding_control"],
                context["away"]["conceding_control"],
                context["home"]["conceding_control"] - context["away"]["conceding_control"],
            ),
            ("리그 순위", home.standings_index, away.standings_index, home.standings_index - away.standings_index),
            ("선수 가용성", home.availability_index, away.availability_index, home.availability_index - away.availability_index),
            ("일정 휴식", context["home"]["rest_index"], context["away"]["rest_index"], context["home"]["rest_index"] - context["away"]["rest_index"]),
            ("이동 피로", context["home"]["travel_index"], context["away"]["travel_index"], context["home"]["travel_index"] - context["away"]["travel_index"]),
            (
                "경기 일관성",
                context["home"]["consistency_index"],
                context["away"]["consistency_index"],
                context["home"]["consistency_index"] - context["away"]["consistency_index"],
            ),
        ]

        contributions = []
        for label, home_value, away_value, gap in factors:
            weight = self.FACTOR_WEIGHTS[label]
            contributions.append(
                {
                    "label": label,
                    "home_value": round(home_value, 1),
                    "away_value": round(away_value, 1),
                    "gap": round(gap, 1),
                    "weight": weight,
                    "impact": round(gap * weight, 2),
                }
            )
        return contributions

    def _project_match_script(
        self,
        match: MatchRecord,
        context: Dict[str, Dict[str, float]],
        weighted_score: float,
    ) -> Dict[str, Any]:
        """Project the expected game script for explanation and API consumers."""

        base_score = self.BASE_SCORE_BY_SPORT.get(match.sport, 1.0)
        home_attack = (
            match.home_metrics.attacking_index * 0.34
            + context["home"]["scoring_trend"] * 0.26
            + match.home_metrics.recent_form * 0.16
            + context["home"]["pace_index"] * 0.12
            + context["home"]["motivation_index"] * 0.12
        )
        away_attack = (
            match.away_metrics.attacking_index * 0.34
            + context["away"]["scoring_trend"] * 0.26
            + match.away_metrics.recent_form * 0.16
            + context["away"]["pace_index"] * 0.12
            + context["away"]["motivation_index"] * 0.12
        )
        home_defense = (
            match.home_metrics.defensive_index * 0.46
            + context["home"]["conceding_control"] * 0.28
            + context["home"]["consistency_index"] * 0.14
            + match.home_metrics.availability_index * 0.12
        )
        away_defense = (
            match.away_metrics.defensive_index * 0.46
            + context["away"]["conceding_control"] * 0.28
            + context["away"]["consistency_index"] * 0.14
            + match.away_metrics.availability_index * 0.12
        )

        pace_factor = ((context["home"]["pace_index"] + context["away"]["pace_index"]) / 2 - 60) / 100
        home_delta = ((home_attack - away_defense) / 100) + (weighted_score / 120)
        away_delta = ((away_attack - home_defense) / 100) - (weighted_score / 140)
        expected_home = max(base_score * (1 + pace_factor * 0.20 + home_delta * 0.34), base_score * 0.45)
        expected_away = max(base_score * (1 + pace_factor * 0.18 + away_delta * 0.34), base_score * 0.45)
        projected_margin = round(expected_home - expected_away, 2)
        projected_total = round(expected_home + expected_away, 2)

        if projected_margin >= 1.2:
            script_label = f"{match.home_team} 우세"
        elif projected_margin <= -1.2:
            script_label = f"{match.away_team} 우세"
        else:
            script_label = "박빙"

        if pace_factor >= 0.12:
            tempo_label = "빠른 템포"
        elif pace_factor <= -0.06:
            tempo_label = "느린 템포"
        else:
            tempo_label = "중간 템포"

        score_precision = 1
        return {
            "projected_home_score": round(expected_home, score_precision),
            "projected_away_score": round(expected_away, score_precision),
            "projected_total": round(projected_total, score_precision),
            "projected_margin": projected_margin,
            "tempo_label": tempo_label,
            "script_label": script_label,
            "scoreline_hint": f"{match.home_team} {round(expected_home, score_precision)} - {round(expected_away, score_precision)} {match.away_team}",
        }

    def _build_model_probabilities(
        self,
        match: MatchRecord,
        market_probs: Dict[str, Optional[float]],
        weighted_score: float,
        projected_margin: float,
        context: Dict[str, Dict[str, float]],
    ) -> Dict[str, Optional[float]]:
        """Build raw probabilities before market blending."""

        margin_scale = projected_margin / max(self.BASE_SCORE_BY_SPORT.get(match.sport, 1.0), 1.0)
        schedule_edge = (context["home"]["rest_index"] - context["away"]["rest_index"]) / 45
        motivation_edge = (context["home"]["motivation_index"] - context["away"]["motivation_index"]) / 65
        base_signal = (weighted_score / 12) + margin_scale + schedule_edge + motivation_edge
        home_no_draw = 1 / (1 + math.exp(-base_signal))

        if market_probs["draw"] is not None:
            market_draw = market_probs["draw"] or 0.26
            balance = 1 - min(abs(weighted_score) / 18, 1)
            draw_bias = 0.14 + (balance * 0.11)
            if match.sport in {"축구", "아이스하키"}:
                draw_bias += 0.04
            draw_model = _clamp((market_draw * 0.78) + draw_bias - abs(margin_scale) * 0.12, 0.12, 0.34)
            remaining = 1 - draw_model
            return {
                "home": remaining * home_no_draw,
                "draw": draw_model,
                "away": remaining * (1 - home_no_draw),
            }

        return {"home": home_no_draw, "draw": None, "away": 1 - home_no_draw}

    def _quality_snapshot(
        self,
        match: MatchRecord,
        market_probs: Dict[str, Optional[float]],
        weighted_score: float,
        model_probs: Dict[str, Optional[float]],
        context: Dict[str, Dict[str, float]],
    ) -> Dict[str, Any]:
        """Estimate confidence, volatility and source quality."""

        outcome_count = sum(1 for value in (match.odds.home, match.odds.draw, match.odds.away) if value is not None)
        expected_outcome_count = 3 if match.odds.draw is not None else 2
        completeness_score = 100 * (outcome_count / expected_outcome_count)

        updated_at = _parse_datetime(match.updated_at) or datetime.now(timezone.utc)
        kickoff = _parse_datetime(match.kickoff) or datetime.now(timezone.utc)
        age_minutes = max((datetime.now(timezone.utc) - updated_at).total_seconds() / 60, 0)
        time_to_kickoff_hours = max((kickoff - datetime.now(timezone.utc)).total_seconds() / 3600, 0)
        recency_score = _clamp(100 - (age_minutes / 4.5), 45, 100)
        source_score = 62 if "샘플" in match.source_label else 90
        signal_strength = _clamp(42 + abs(weighted_score) * 4.3, 0, 100)
        rest_balance = abs(context["home"]["rest_index"] - context["away"]["rest_index"])
        volatility_score = round(
            _clamp(
                72 - (signal_strength * 0.42) + (18 if "샘플" in match.source_label else 0) + (12 - min(rest_balance, 12)),
                14,
                86,
            ),
            1,
        )
        data_quality_score = round((completeness_score * 0.34) + (recency_score * 0.33) + (source_score * 0.33), 1)

        overround = round(
            max(
                sum((1 / odd) for odd in (match.odds.home, match.odds.draw, match.odds.away) if odd and odd > 1) - 1,
                0,
            )
            * 100,
            2,
        )
        disagreement = round(
            sum(
                abs((model_probs[outcome] or 0) - (market_probs[outcome] or 0))
                for outcome in ("home", "draw", "away")
                if model_probs.get(outcome) is not None and market_probs.get(outcome) is not None
            )
            * 100,
            1,
        )
        blend_weight = round(_clamp(0.56 + ((data_quality_score - 60) / 100) - (volatility_score / 300), 0.48, 0.78), 2)

        if volatility_score <= 28:
            volatility_label = "낮음"
        elif volatility_score <= 54:
            volatility_label = "보통"
        else:
            volatility_label = "높음"

        return {
            "data_quality_score": data_quality_score,
            "volatility_score": volatility_score,
            "volatility_label": volatility_label,
            "market_overround_percent": overround,
            "market_disagreement_percent": disagreement,
            "blend_weight": blend_weight,
            "time_to_kickoff_hours": round(time_to_kickoff_hours, 1),
            "age_minutes": round(age_minutes, 1),
        }

    def _build_probability_bands(
        self,
        probabilities: Dict[str, Optional[float]],
        volatility_score: float,
    ) -> Dict[str, Optional[Dict[str, float]]]:
        """Return low/base/high probability scenarios for each outcome."""

        uncertainty = _clamp((volatility_score / 100) * 0.09, 0.02, 0.10)
        bands: Dict[str, Optional[Dict[str, float]]] = {}
        for outcome, value in probabilities.items():
            if value is None:
                bands[outcome] = None
                continue
            bands[outcome] = {
                "low": round(_clamp((value - uncertainty) * 100, 1, 98), 1),
                "base": round(value * 100, 1),
                "high": round(_clamp((value + uncertainty) * 100, 1, 98), 1),
            }
        return bands

    def _rank_outcomes(
        self,
        match: MatchRecord,
        model_probs: Dict[str, Optional[float]],
        market_probs: Dict[str, Optional[float]],
        quality: Dict[str, Any],
        probability_bands: Dict[str, Optional[Dict[str, float]]],
    ) -> List[Dict[str, Any]]:
        """Score every outcome so the API can expose ranked alternatives."""

        labels = {"home": match.home_team, "draw": "무승부", "away": match.away_team}
        odds_map = {"home": match.odds.home, "draw": match.odds.draw, "away": match.odds.away}
        ranked = []

        for outcome in ("home", "draw", "away"):
            probability = model_probs.get(outcome)
            market_probability = market_probs.get(outcome)
            odds = odds_map.get(outcome)
            if probability is None or market_probability is None or odds is None or odds <= 1:
                continue

            edge_percent = (probability - market_probability) * 100
            expected_value_percent = ((probability * odds) - 1) * 100
            fair_odds = round(1 / probability, 2) if probability > 0 else None
            kelly_fraction = max((((odds - 1) * probability) - (1 - probability)) / (odds - 1), 0)
            half_kelly = round(kelly_fraction * 50, 1)
            quarter_kelly = round(kelly_fraction * 25, 1)
            if quality["volatility_score"] <= 28:
                risk_label = "공격적 가능"
            elif quality["volatility_score"] <= 54:
                risk_label = "균형"
            else:
                risk_label = "보수적 접근"

            band = probability_bands.get(outcome)
            ranked.append(
                {
                    "outcome": outcome,
                    "label": labels[outcome],
                    "odds": round(odds, 2),
                    "fair_odds": fair_odds,
                    "probability_percent": round(probability * 100, 1),
                    "market_probability_percent": round(market_probability * 100, 1),
                    "edge_percent": round(edge_percent, 1),
                    "expected_value_percent": round(expected_value_percent, 1),
                    "kelly_percent": round(kelly_fraction * 100, 1),
                    "half_kelly_percent": half_kelly,
                    "quarter_kelly_percent": quarter_kelly,
                    "risk_label": risk_label,
                    "probability_band": band,
                }
            )

        ranked.sort(key=lambda item: (item["expected_value_percent"], item["edge_percent"], item["probability_percent"]), reverse=True)
        return ranked

    def _serialize_evidence(self, contributions: List[Dict[str, float]]) -> List[Dict[str, Any]]:
        """Prepare evidence rows for templates and JSON consumers."""

        return [
            {
                "label": item["label"],
                "home_value": item["home_value"],
                "away_value": item["away_value"],
                "gap": item["gap"],
                "weight": round(item["weight"] * 100, 1),
                "impact": item["impact"],
            }
            for item in contributions
        ]

    def _blend_probability(
        self,
        model_value: Optional[float],
        market_value: Optional[float],
        model_weight: float,
    ) -> Optional[float]:
        """Blend model and market probabilities using a dynamic weight."""

        if model_value is None:
            return market_value
        if market_value is None:
            return model_value
        return (model_value * model_weight) + (market_value * (1 - model_weight))

    def _renormalize(self, probabilities: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        """Ensure probability totals sum to one after blending."""

        numeric = {key: value for key, value in probabilities.items() if value is not None}
        total = sum(numeric.values())
        if not total:
            return probabilities
        return {
            key: (value / total if value is not None else None)
            for key, value in probabilities.items()
        }

    def _empty_recommendation(self) -> Dict[str, Any]:
        """Fallback object when no valid market exists."""

        return {
            "outcome": "none",
            "label": "추천 없음",
            "odds": None,
            "fair_odds": None,
            "probability_percent": 0,
            "market_probability_percent": 0,
            "edge_percent": 0,
            "expected_value_percent": 0,
            "kelly_percent": 0,
            "half_kelly_percent": 0,
            "quarter_kelly_percent": 0,
            "risk_label": "관망",
            "probability_band": None,
        }

    def _top_reasons(self, evidence: List[Dict[str, Any]]) -> List[str]:
        """Convert the strongest factors into short human-readable reasons."""

        ranked = sorted(evidence, key=lambda item: abs(item["impact"]), reverse=True)[:3]
        reasons = []
        for item in ranked:
            direction = "홈 우위" if item["impact"] >= 0 else "원정 우위"
            reasons.append(f"{item['label']} {direction} ({item['impact']:+.1f}, 가중치 {item['weight'] * 100:.0f}%)")
        return reasons

    def _round_probabilities(self, probabilities: Dict[str, Optional[float]]) -> Dict[str, Optional[float]]:
        """Express probabilities as percentages for presentation."""

        return {
            key: (round(value * 100, 1) if value is not None else None)
            for key, value in probabilities.items()
        }

    def _confidence_label(self, edge_percent: float, expected_value_percent: float) -> str:
        """Map numeric edge quality into a concise confidence label."""

        if edge_percent >= 7 and expected_value_percent >= 8:
            return "매우 높음"
        if edge_percent >= 4 and expected_value_percent >= 4:
            return "높음"
        if edge_percent >= 2:
            return "중간"
        return "관망"


class DashboardService:
    """Compose collected data, analysis results and chart payloads."""

    def __init__(self) -> None:
        self.collector = BetmanCollector()
        self.analyzer = MatchAnalyzer()

    def build_dashboard_payload(
        self,
        selected_sport: Optional[str] = None,
        min_edge: float = 1.5,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Build the full dashboard response used by HTML and JSON endpoints."""

        analyzed, status = self._get_analyzed_matches(force_refresh=force_refresh)
        sports = self._sports_index(analyzed)
        filtered = [
            item
            for item in analyzed
            if (not selected_sport or item["sport"] == selected_sport)
            and item["recommendation"]["edge_percent"] >= min_edge
        ]

        summary = self._build_summary(analyzed, filtered, status)
        charts = self._build_charts(filtered or analyzed)
        insights = self._analysis_notes()

        return {
            "generated_at": _now_iso(),
            "filters": {
                "selected_sport": selected_sport or "전체",
                "min_edge": round(min_edge, 1),
            },
            "sports": sports,
            "summary": summary,
            "status": status,
            "matches": filtered,
            "total_matches_before_filter": len(analyzed),
            "charts": charts,
            "analysis_notes": insights,
            "api": {
                "predictions": "/api/predictions",
                "match_detail": "/api/matches/<match_id>",
                "health": "/api/health",
            },
        }

    def build_predictions_payload(
        self,
        selected_sport: Optional[str] = None,
        min_edge: float = 0.0,
        sort_by: str = "expected_value",
        limit: int = 20,
        force_refresh: bool = False,
    ) -> Dict[str, Any]:
        """Build an API-first payload focused on ranked predictions."""

        analyzed, status = self._get_analyzed_matches(force_refresh=force_refresh)
        filtered = [
            item
            for item in analyzed
            if (not selected_sport or item["sport"] == selected_sport)
            and item["recommendation"]["edge_percent"] >= min_edge
        ]

        sort_key_map = {
            "edge": lambda item: item["recommendation"]["edge_percent"],
            "confidence": lambda item: item["quality"]["data_quality_score"] - item["quality"]["volatility_score"],
            "kickoff": lambda item: item["kickoff"],
            "expected_value": lambda item: item["recommendation"]["expected_value_percent"],
        }
        sorter = sort_key_map.get(sort_by, sort_key_map["expected_value"])
        reverse = sort_by != "kickoff"
        ranked = sorted(filtered, key=sorter, reverse=reverse)[: max(1, min(limit, 100))]

        return {
            "generated_at": _now_iso(),
            "filters": {
                "selected_sport": selected_sport or "전체",
                "min_edge": round(min_edge, 1),
                "sort_by": sort_by,
                "limit": max(1, min(limit, 100)),
            },
            "status": status,
            "count": len(ranked),
            "predictions": ranked,
        }

    def build_match_detail(self, match_id: str, force_refresh: bool = False) -> Dict[str, Any]:
        """Return a single match with richer surrounding context for API clients."""

        analyzed, status = self._get_analyzed_matches(force_refresh=force_refresh)
        match = next((item for item in analyzed if item["match_id"] == match_id), None)
        if not match:
            raise KeyError(match_id)

        same_sport = [
            item
            for item in analyzed
            if item["sport"] == match["sport"] and item["match_id"] != match_id
        ]
        related = sorted(
            same_sport,
            key=lambda item: item["recommendation"]["expected_value_percent"],
            reverse=True,
        )[:3]

        return {
            "generated_at": _now_iso(),
            "status": status,
            "match": match,
            "related_matches": related,
        }

    def _get_analyzed_matches(self, force_refresh: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Collect raw matches and run them through the analyzer once per request."""

        matches, status = self.collector.collect_matches(force_refresh=force_refresh)
        return [self.analyzer.analyze(match) for match in matches], status

    def _sports_index(self, matches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build sport filter options with counts."""

        counts: Dict[str, int] = {}
        for item in matches:
            counts[item["sport"]] = counts.get(item["sport"], 0) + 1

        result = [{"name": "전체", "count": len(matches)}]
        for name in sorted(counts):
            result.append({"name": name, "count": counts[name]})
        return result

    def _build_summary(
        self,
        all_matches: List[Dict[str, Any]],
        visible_matches: List[Dict[str, Any]],
        status: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Create compact KPIs for the dashboard header."""

        actionable = [item for item in all_matches if item["recommendation"]["expected_value_percent"] > 0]
        best_pick = max(
            visible_matches or all_matches,
            key=lambda item: item["recommendation"]["expected_value_percent"],
            default=None,
        )

        average_edge = (
            round(sum(item["recommendation"]["edge_percent"] for item in visible_matches) / len(visible_matches), 1)
            if visible_matches
            else 0.0
        )

        return {
            "total_matches": len(all_matches),
            "visible_matches": len(visible_matches),
            "actionable_matches": len(actionable),
            "average_edge": average_edge,
            "sports_covered": len({item["sport"] for item in all_matches}),
            "leagues_covered": len({item["league"] for item in all_matches}),
            "best_pick": (
                {
                    "match": f"{best_pick['home_team']} vs {best_pick['away_team']}",
                    "bet": best_pick["recommendation"]["label"],
                    "edge_percent": best_pick["recommendation"]["edge_percent"],
                    "expected_value_percent": best_pick["recommendation"]["expected_value_percent"],
                }
                if best_pick
                else None
            ),
            "refresh_note": (
                f"{status['source_label']} 기준 {status['deduplicated_count']}경기 정규화"
            ),
        }

    def _build_charts(self, matches: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create responsive charts used by the dashboard."""

        edge_sorted = sorted(matches, key=lambda item: item["recommendation"]["edge_percent"], reverse=True)[:8]
        figure_edge = go.Figure(
            data=[
                go.Bar(
                    x=[f"{item['home_team']} vs {item['away_team']}" for item in edge_sorted],
                    y=[item["recommendation"]["edge_percent"] for item in edge_sorted],
                    marker_color="#2563eb",
                    name="엣지",
                )
            ]
        )
        figure_edge.update_layout(
            title="상위 추천 경기 엣지",
            height=320,
            margin=dict(l=30, r=20, t=48, b=80),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        figure_prob = go.Figure()
        figure_prob.add_trace(
            go.Scatter(
                x=[item["market_probabilities"]["home"] or 0 for item in matches],
                y=[item["model_probabilities"]["home"] or 0 for item in matches],
                mode="markers+text",
                text=[item["league"] for item in matches],
                textposition="top center",
                marker=dict(size=11, color="#10b981"),
                name="홈 승리 확률",
            )
        )
        figure_prob.update_layout(
            title="시장 확률 vs 모델 확률",
            xaxis_title="시장 확률 (%)",
            yaxis_title="모델 확률 (%)",
            height=320,
            margin=dict(l=50, r=20, t=48, b=50),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        sport_counts: Dict[str, int] = {}
        for item in matches:
            sport_counts[item["sport"]] = sport_counts.get(item["sport"], 0) + 1
        figure_sport = go.Figure(
            data=[
                go.Pie(
                    labels=list(sport_counts.keys()) or ["데이터 없음"],
                    values=list(sport_counts.values()) or [1],
                    hole=0.55,
                    marker=dict(colors=["#2563eb", "#10b981", "#f59e0b", "#8b5cf6", "#ef4444"]),
                )
            ]
        )
        figure_sport.update_layout(
            title="종목별 수집 분포",
            height=320,
            margin=dict(l=20, r=20, t=48, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        labels = list(MatchAnalyzer.FACTOR_WEIGHTS.keys())
        weights = [round(value * 100, 1) for value in MatchAnalyzer.FACTOR_WEIGHTS.values()]
        figure_factor = go.Figure(
            data=[
                go.Bar(
                    x=weights,
                    y=labels,
                    orientation="h",
                    marker_color="#8b5cf6",
                    name="가중치",
                )
            ]
        )
        figure_factor.update_layout(
            title="예측 모델 입력 가중치",
            height=320,
            margin=dict(l=110, r=20, t=48, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )

        return {
            "edge": _figure_to_json(figure_edge),
            "probability": _figure_to_json(figure_prob),
            "sport": _figure_to_json(figure_sport),
            "factor": _figure_to_json(figure_factor),
        }

    def _analysis_notes(self) -> List[Dict[str, str]]:
        """Explain the collection and prediction pipeline to the user."""

        return [
            {
                "title": "수집 파이프라인",
                "body": "HTML/JSON 원본을 공통 스키마로 정규화하고, 경기 ID·리그·시각·팀명을 기준으로 중복을 제거합니다.",
            },
            {
                "title": "예측 근거",
                "body": "전력, 최근 폼, 홈/원정 적합도, 상대 전적, 공격/수비 추세, 순위, 선수 가용성을 결합해 확률을 계산합니다.",
            },
            {
                "title": "안정성 설계",
                "body": "원격 수집 실패 시 샘플 데이터로 자동 전환하고, 수집 결과와 경고를 상태 패널에 노출합니다.",
            },
        ]


dashboard_service = DashboardService()


def build_dashboard_payload(
    selected_sport: Optional[str] = None,
    min_edge: float = 1.5,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Public helper used by Flask routes."""

    return dashboard_service.build_dashboard_payload(
        selected_sport=selected_sport,
        min_edge=min_edge,
        force_refresh=force_refresh,
    )


def build_predictions_payload(
    selected_sport: Optional[str] = None,
    min_edge: float = 0.0,
    sort_by: str = "expected_value",
    limit: int = 20,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """Public helper for the ranked predictions API."""

    return dashboard_service.build_predictions_payload(
        selected_sport=selected_sport,
        min_edge=min_edge,
        sort_by=sort_by,
        limit=limit,
        force_refresh=force_refresh,
    )


def build_match_detail(match_id: str, force_refresh: bool = False) -> Dict[str, Any]:
    """Public helper for the single-match detail API."""

    return dashboard_service.build_match_detail(match_id=match_id, force_refresh=force_refresh)
