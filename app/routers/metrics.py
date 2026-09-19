from fastapi import APIRouter, Depends, HTTPException, Query, status
from datetime import date, timedelta
import sqlite3
from typing import Optional, List, Any, Dict

from app.database import get_db_connection, get_skin_trend
from app.models import (
    TherapeuticPressureResponse,
    TherapeuticAxisMetric,
    SkinTrendPoint,
)
from app.routine_config import (
    all_slot_ids,
    get_compliance_axes,
    get_windows,
    public_config_payload,
)
from app.services.clinical_analytics import build_clinical_dashboard_payload

router = APIRouter(prefix="/api/v1/metrics", tags=["Metrics"])

def get_db():
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()


@router.get("/routine-config", summary="Routine products, actives and formula parameters")
def read_routine_config():
    """Espone la config esterna (senza path assoluti) per UI e client."""
    return public_config_payload()


@router.get("/therapeutic-pressure", response_model=TherapeuticPressureResponse, summary="Rolling therapeutic pressure metrics")
def get_therapeutic_pressure(
    target_date: Optional[date] = Query(None, description="Data target (default: oggi)"),
    db: sqlite3.Connection = Depends(get_db)
):
    """
    Pressione terapeutica su finestra mobile configurabile.
    Assi, slot e target sono definiti in routine.json.
    """
    if not target_date:
        target_date = date.today()

    windows = get_windows()
    window_days = int(windows.get("therapeutic", 14))
    axes_cfg = get_compliance_axes()
    slot_ids = all_slot_ids()

    start_date = target_date - timedelta(days=window_days - 1)
    start_str = start_date.isoformat()
    end_str = target_date.isoformat()

    window_map: Dict[str, Dict[str, Any]] = {}
    for i in range(window_days):
        d = start_date + timedelta(days=i)
        window_map[d.isoformat()] = {sid: False for sid in slot_ids}
        window_map[d.isoformat()]["recorded"] = False

    # Solo colonne presenti in schema; gli id config devono allinearsi al DB.
    select_cols = [sid for sid in slot_ids if sid in {
        "am_cleanser", "am_azid", "am_rederma", "am_spf",
        "pm_cleanser", "pm_differin", "pm_rederma",
    }]
    if not select_cols:
        select_cols = ["am_azid", "am_rederma", "pm_differin", "pm_rederma"]

    query = f"""
    SELECT entry_date, {", ".join(select_cols)}
    FROM daily_logs
    WHERE entry_date BETWEEN ? AND ?
    """

    try:
        cursor = db.execute(query, (start_str, end_str))
        rows = cursor.fetchall()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Errore nel recupero dei dati dal database: {str(e)}"
        )

    days_recorded = 0
    for row in rows:
        d_str = row["entry_date"]
        if d_str not in window_map:
            continue
        entry = window_map[d_str]
        for sid in select_cols:
            entry[sid] = bool(row[sid])
        entry["recorded"] = True
        days_recorded += 1

    axes_out: List[TherapeuticAxisMetric] = []
    for axis in axes_cfg:
        flags = list(axis["slot_flags"])
        total_slots = int(axis["total_slots"])
        target_pct = float(axis["target_pct"])
        completed = 0
        for day in window_map.values():
            for flag in flags:
                if day.get(flag):
                    completed += 1
        current_pct = round((completed / total_slots) * 100.0, 1) if total_slots else 0.0
        axes_out.append(
            TherapeuticAxisMetric(
                id=axis["id"],
                label=axis["label"],
                current_pct=current_pct,
                target_pct=target_pct,
                completed_slots=completed,
                total_slots=total_slots,
                is_compliant=current_pct >= target_pct,
            )
        )

    return TherapeuticPressureResponse(
        window_start=start_str,
        window_end=end_str,
        days_recorded=days_recorded,
        window_days=window_days,
        axes=axes_out,
    )


@router.get("/skin-trend", response_model=List[SkinTrendPoint], summary="Skin analysis metrics over photo checkpoints")
def read_skin_trend(db: sqlite3.Connection = Depends(get_db)):
    """Serie temporale delle metriche PIH/texture per i checkpoint con analisi."""
    try:
        return get_skin_trend(db)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )


@router.get("/clinical-dashboard", summary="Clinical habits + results dashboard payload")
def clinical_dashboard(
    target_date: Optional[date] = Query(None),
    window_days: Optional[int] = Query(None),
    results_window_days: Optional[int] = Query(None),
    db: sqlite3.Connection = Depends(get_db),
):
    as_of = target_date.isoformat() if target_date else None
    try:
        return build_clinical_dashboard_payload(
            db,
            as_of=as_of,
            window_days=window_days,
            results_window_days=results_window_days,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
