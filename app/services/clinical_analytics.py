"""Carica log/checkpoint dal DB e genera il payload clinico."""

from __future__ import annotations

import base64
import json
import sqlite3
from typing import Any

from analytics.clinical_engine import run_clinical_pipeline
from app.routine_config import all_slot_ids


# Colonne booleane note nello schema SQLite attuale.
_KNOWN_LOG_SLOTS = {
    "am_cleanser",
    "am_azid",
    "am_rederma",
    "am_spf",
    "pm_cleanser",
    "pm_differin",
    "pm_rederma",
}


def _log_select_columns() -> list[str]:
    configured = [sid for sid in all_slot_ids() if sid in _KNOWN_LOG_SLOTS]
    return configured or ["am_azid", "am_rederma", "pm_differin", "pm_rederma"]


def _load_daily_logs(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    cols = _log_select_columns()
    rows = conn.execute(
        f"""
        SELECT entry_date, {", ".join(cols)}
        FROM daily_logs
        ORDER BY entry_date ASC
        """
    ).fetchall()
    out = []
    for r in rows:
        item: dict[str, Any] = {"entry_date": r["entry_date"]}
        for c in cols:
            item[c] = bool(r[c])
        out.append(item)
    return out


def _load_checkpoints(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT entry_date, analysis_json
        FROM photo_checkpoints
        WHERE analysis_json IS NOT NULL
        ORDER BY entry_date ASC
        """
    ).fetchall()
    out = []
    for r in rows:
        try:
            analysis = json.loads(r["analysis_json"]) if r["analysis_json"] else {}
        except json.JSONDecodeError:
            analysis = {}
        out.append({"entry_date": r["entry_date"], "analysis": analysis})
    return out


def build_clinical_dashboard_payload(
    conn: sqlite3.Connection,
    as_of: str | None = None,
    window_days: int | None = None,
    results_window_days: int | None = None,
) -> dict[str, Any]:
    logs = _load_daily_logs(conn)
    checkpoints = _load_checkpoints(conn)
    result = run_clinical_pipeline(
        logs,
        checkpoints,
        as_of=as_of,
        window_days=window_days,
        results_window_days=results_window_days,
    )
    swimlanes = {
        agent: base64.b64encode(png).decode("ascii")
        for agent, png in result["swimlane_pngs"].items()
    }
    return {
        "as_of": result.get("as_of"),
        "window_days": result.get("window_days"),
        "results_window_days": result.get("results_window_days"),
        "checkpoint_count": result["checkpoint_count"],
        "latest_saturation": result["latest_saturation"],
        "swimlane_pngs_base64": swimlanes,
        "results_page1_png_base64": base64.b64encode(result["results_page1_png"]).decode(
            "ascii"
        ),
        "results_page2_png_base64": base64.b64encode(result["results_page2_png"]).decode(
            "ascii"
        )
        if result.get("results_page2_png")
        else None,
        "correlation_png_base64": base64.b64encode(result["correlation_png"]).decode(
            "ascii"
        )
        if result.get("correlation_png")
        else None,
    }
