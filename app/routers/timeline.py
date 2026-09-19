from fastapi import APIRouter, Depends, status
import sqlite3
from typing import List
from app.database import get_db_connection, get_timeline
from app.models import TimelineItem

router = APIRouter(prefix="/api/v1/timeline", tags=["Timeline"])

def get_db():
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()

@router.get("", response_model=List[TimelineItem], status_code=status.HTTP_200_OK, summary="Fetch ordered manifest of all normalized checkpoints")
def read_timeline(db: sqlite3.Connection = Depends(get_db)):
    """
    Restituisce l'elenco ordinato cronologicamente di tutti i checkpoint fotografici normalizzati,
    con i relativi metadati clinici associati, i giorni trascorsi dal baseline (T+X)
    e i tassi di aderenza terapeutica (compliance rate) am/pm calcolati per ciascun ciclo.
    """
    return get_timeline(db)
