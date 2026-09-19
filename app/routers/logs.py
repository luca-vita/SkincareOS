from fastapi import APIRouter, Depends, HTTPException, status
from datetime import date
import sqlite3
from typing import List
from app.database import get_db_connection, upsert_daily_log, get_daily_log, get_all_daily_logs
from app.models import DailyLogCreate, DailyLogResponse

router = APIRouter(prefix="/api/v1/logs", tags=["Logs"])

def get_db():
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()

@router.post("", status_code=status.HTTP_200_OK, summary="Insert or update daily tracking log")
def create_or_update_log(log: DailyLogCreate, db: sqlite3.Connection = Depends(get_db)):
    """
    Inserisce o aggiorna un log giornaliero. Se esiste già un log per la data specificata,
    esegue un'operazione di UPSERT sovrascrivendo i valori precedenti.
    """
    try:
        upsert_daily_log(db, log)
        return {"status": "success", "message": "Log stored successfully"}
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Errore durante il salvataggio del log: {str(e)}"
        )

@router.get("", response_model=List[DailyLogResponse], summary="Fetch all daily logs")
def read_all_logs(db: sqlite3.Connection = Depends(get_db)):
    """
    Recupera lo storico completo di tutti i log giornalieri ordinati per data crescente.
    Utilizzato per popolare il calendario e i grafici di andamento.
    """
    return get_all_daily_logs(db)

@router.get("/{entry_date}", response_model=DailyLogResponse, summary="Fetch daily log by date")
def read_log(entry_date: date, db: sqlite3.Connection = Depends(get_db)):
    """
    Recupera il log giornaliero per la data specificata.
    Se non esiste alcun log, restituisce un errore 404.
    """
    log = get_daily_log(db, entry_date)
    if not log:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Nessun log trovato per la data specificata"
        )
    return log
