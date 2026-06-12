"""Main Flask entrypoint for the Betman dashboard."""

from flask import Flask, jsonify, render_template, request

from api.betman_service import (
    build_dashboard_payload,
    build_match_detail,
    build_predictions_payload,
)

app = Flask(__name__)


def _parse_min_edge(raw_value: str) -> float:
    """Parse and clamp the edge filter to a safe range."""

    try:
        value = float(raw_value)
    except (TypeError, ValueError):
        value = 1.5
    return max(0.0, min(value, 20.0))


def _parse_limit(raw_value: str) -> int:
    """Parse and clamp a positive limit value."""

    try:
        value = int(raw_value)
    except (TypeError, ValueError):
        value = 20
    return max(1, min(value, 100))


@app.route("/")
def index() -> str:
    """Render the dashboard shell with server-side bootstrap data."""

    selected_sport = request.args.get("sport", "").strip() or None
    min_edge = _parse_min_edge(request.args.get("min_edge", "1.5"))
    force_refresh = request.args.get("refresh", "0") == "1"

    payload = build_dashboard_payload(
        selected_sport=selected_sport,
        min_edge=min_edge,
        force_refresh=force_refresh,
    )
    return render_template("index.html", initial_payload=payload)


@app.route("/api/dashboard")
def api_dashboard():
    """Return the dashboard payload used by the SPA-style client."""

    selected_sport = request.args.get("sport", "").strip() or None
    min_edge = _parse_min_edge(request.args.get("min_edge", "1.5"))
    force_refresh = request.args.get("refresh", "0") == "1"

    payload = build_dashboard_payload(
        selected_sport=selected_sport,
        min_edge=min_edge,
        force_refresh=force_refresh,
    )
    return jsonify(payload)


@app.route("/api/sport")
def api_sport():
    """Return the legacy per-sport payload used by the original UI."""

    selected_sport = request.args.get("name", "").strip() or None
    min_edge = _parse_min_edge(request.args.get("min_edge", "1.5"))
    force_refresh = request.args.get("refresh", "0") == "1"

    payload = build_dashboard_payload(
        selected_sport=selected_sport,
        min_edge=min_edge,
        force_refresh=force_refresh,
    )
    if selected_sport and selected_sport not in payload["sports_categories"]:
        return jsonify({"error": "종목을 찾을 수 없습니다.", "opportunities": {}}), 404

    category = payload["sports_categories"].get(selected_sport or "전체")
    return jsonify(
        {
            "sport": selected_sport or "전체",
            "icon": (category or {}).get("icon", "📊"),
            "api_source": (category or {}).get("api_source", "Betman 통합 수집 + 하이브리드 분석"),
            "opportunities": payload["legacy_opportunities"],
            "summary": payload["summary"],
        }
    )


@app.route("/api/predictions")
def api_predictions():
    """Return ranked predictions for automation or external clients."""

    selected_sport = request.args.get("sport", "").strip() or None
    min_edge = _parse_min_edge(request.args.get("min_edge", "0"))
    force_refresh = request.args.get("refresh", "0") == "1"
    sort_by = request.args.get("sort_by", "expected_value").strip() or "expected_value"
    limit = _parse_limit(request.args.get("limit", "20"))

    payload = build_predictions_payload(
        selected_sport=selected_sport,
        min_edge=min_edge,
        sort_by=sort_by,
        limit=limit,
        force_refresh=force_refresh,
    )
    return jsonify(payload)


@app.route("/api/matches/<match_id>")
def api_match_detail(match_id: str):
    """Return the detailed prediction payload for a single match."""

    force_refresh = request.args.get("refresh", "0") == "1"
    try:
        payload = build_match_detail(match_id=match_id, force_refresh=force_refresh)
    except KeyError:
        return jsonify({"error": "경기를 찾을 수 없습니다.", "match_id": match_id}), 404
    return jsonify(payload)


@app.route("/api/health")
def api_health():
    """Expose a minimal health endpoint for uptime checks."""

    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True)
