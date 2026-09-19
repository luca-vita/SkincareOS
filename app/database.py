import json
import sqlite3
import os
from datetime import datetime, date
from typing import List, Optional, Any, Dict
from app.config import DB_PATH, PHOTOS_DIR
from app.exceptions import DatabaseBusyException
from app.models import DailyLogCreate, DailyLogResponse, TimelineItem, GymWorkoutEnum, SkinAnalysisMetrics

def get_db_connection() -> sqlite3.Connection:
    """Apre una connessione a SQLite impostando WAL mode e Foreign Keys ON.
    Gestisce i blocchi temporanei sollevando DatabaseBusyException."""
    try:
        # Crea la cartella data se non esiste
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        
        conn = sqlite3.connect(DB_PATH, timeout=5.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        
        # Abilita WAL mode e Foreign Keys
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        
        return conn
    except sqlite3.OperationalError as e:
        if "locked" in str(e).lower() or "busy" in str(e).lower():
            raise DatabaseBusyException("Il database è temporaneamente occupato. Riprova.")
        raise e

def migrate_db_if_needed(conn: sqlite3.Connection):
    """Verifica se il database ha il vecchio schema e lo migra automaticamente."""
    # Verifica se la tabella daily_logs esiste
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='daily_logs';")
    if not cursor.fetchone():
        return

    # Recupera le colonne esistenti
    cursor = conn.execute("PRAGMA table_info(daily_logs);")
    columns = [row["name"] for row in cursor.fetchall()]
    
    if "am_routine" in columns and "am_cleanser" not in columns:
        print("[MIGRATION] Rilevato vecchio schema del database. Avvio migrazione automatica...")
        
        # Disabilita temporaneamente le foreign keys per la migrazione
        conn.execute("PRAGMA foreign_keys = OFF;")
        
        try:
            with conn:
                # 1. Rinomina la vecchia tabella
                conn.execute("ALTER TABLE daily_logs RENAME TO daily_logs_old;")
                
                # 2. Crea la nuova tabella con lo schema aggiornato
                conn.execute("""
                CREATE TABLE daily_logs (
                    entry_date TEXT PRIMARY KEY,
                    am_cleanser INTEGER NOT NULL CHECK(am_cleanser IN (0, 1)),
                    am_azid INTEGER NOT NULL CHECK(am_azid IN (0, 1)),
                    am_rederma INTEGER NOT NULL CHECK(am_rederma IN (0, 1)),
                    am_spf INTEGER NOT NULL CHECK(am_spf IN (0, 1)),
                    pm_cleanser INTEGER NOT NULL CHECK(pm_cleanser IN (0, 1)),
                    pm_differin INTEGER NOT NULL CHECK(pm_differin IN (0, 1)),
                    pm_rederma INTEGER NOT NULL CHECK(pm_rederma IN (0, 1)),
                    stinging_index INTEGER NOT NULL CHECK(stinging_index BETWEEN 0 AND 3),
                    gym_workout TEXT NOT NULL CHECK(gym_workout IN ('NONE', 'MORNING', 'AFTERNOON', 'EVENING')),
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                """)
                
                # 3. Copia i dati mappando le vecchie colonne routine booleane sui singoli step
                conn.execute("""
                INSERT INTO daily_logs (
                    entry_date, am_cleanser, am_azid, am_rederma, am_spf,
                    pm_cleanser, pm_differin, pm_rederma, stinging_index, gym_workout, notes, created_at
                )
                SELECT 
                    entry_date, am_routine, am_routine, am_routine, am_routine,
                    pm_routine, pm_routine, pm_routine, stinging_index, gym_workout, notes, created_at
                FROM daily_logs_old;
                """)
                
                # 4. Rimuovi la vecchia tabella
                conn.execute("DROP TABLE daily_logs_old;")
                
            print("[MIGRATION] Migrazione completata con successo! I dati storici sono stati preservati.")
        except Exception as e:
            print(f"[MIGRATION] Errore durante la migrazione: {str(e)}")
            raise e
        finally:
            conn.execute("PRAGMA foreign_keys = ON;")

    # Verifica se la tabella photo_checkpoints esiste e se fa riferimento a daily_logs_old
    cursor = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='photo_checkpoints';")
    row = cursor.fetchone()
    if row:
        sql = row["sql"]
        if "daily_logs_old" in sql:
            print("[MIGRATION] Rilevato riferimento a daily_logs_old in photo_checkpoints. Avvio migrazione correttiva...")
            conn.execute("PRAGMA foreign_keys = OFF;")
            try:
                with conn:
                    # 1. Rinomina la tabella photo_checkpoints
                    conn.execute("ALTER TABLE photo_checkpoints RENAME TO photo_checkpoints_old;")
                    
                    # 2. Crea la nuova tabella photo_checkpoints con la FK corretta
                    conn.execute("""
                    CREATE TABLE photo_checkpoints (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        entry_date TEXT NOT NULL UNIQUE,
                        file_path TEXT NOT NULL,
                        resolution_width INTEGER NOT NULL DEFAULT 2048,
                        resolution_height INTEGER NOT NULL DEFAULT 2048,
                        interpupillary_distance REAL NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (entry_date) REFERENCES daily_logs(entry_date) ON DELETE CASCADE
                    );
                    """)
                    
                    # 3. Copia i dati
                    conn.execute("""
                    INSERT INTO photo_checkpoints (
                        id, entry_date, file_path, resolution_width, resolution_height, interpupillary_distance, created_at
                    )
                    SELECT id, entry_date, file_path, resolution_width, resolution_height, interpupillary_distance, created_at
                    FROM photo_checkpoints_old;
                    """)
                    
                    # 4. Rimuovi la vecchia tabella
                    conn.execute("DROP TABLE photo_checkpoints_old;")
                    
                    # 5. Ricrea l'indice
                    conn.execute("CREATE INDEX IF NOT EXISTS idx_photos_date ON photo_checkpoints(entry_date);")
                    
                print("[MIGRATION] Migrazione correttiva di photo_checkpoints completata con successo!")
            except Exception as e:
                print(f"[MIGRATION] Errore durante la migrazione correttiva: {str(e)}")
                raise e
            finally:
                conn.execute("PRAGMA foreign_keys = ON;")

    # Colonne analisi PIH/texture su photo_checkpoints (idempotente)
    cursor = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='photo_checkpoints';"
    )
    if cursor.fetchone():
        cursor = conn.execute("PRAGMA table_info(photo_checkpoints);")
        photo_cols = {row["name"] for row in cursor.fetchall()}
        alter_map = {
            "pih_path": "ALTER TABLE photo_checkpoints ADD COLUMN pih_path TEXT;",
            "texture_path": "ALTER TABLE photo_checkpoints ADD COLUMN texture_path TEXT;",
            "analysis_json": "ALTER TABLE photo_checkpoints ADD COLUMN analysis_json TEXT;",
            "analysis_version": "ALTER TABLE photo_checkpoints ADD COLUMN analysis_version TEXT;",
        }
        for col, sql in alter_map.items():
            if col not in photo_cols:
                print(f"[MIGRATION] Aggiungo colonna photo_checkpoints.{col}...")
                with conn:
                    conn.execute(sql)


def init_db():
    """Inizializza il database eseguendo lo schema SQL."""
    schema_path = os.path.join(os.path.dirname(__file__), "schema.sql")
    if not os.path.exists(schema_path):
        raise FileNotFoundError(f"Schema SQL non trovato in {schema_path}")
        
    with open(schema_path, "r", encoding="utf-8") as f:
        schema_sql = f.read()
        
    conn = get_db_connection()
    try:
        # Esegui prima la migrazione se necessaria
        migrate_db_if_needed(conn)
        
        with conn:
            conn.executescript(schema_sql)
    finally:
        conn.close()

def upsert_daily_log(conn: sqlite3.Connection, log: DailyLogCreate):
    """Inserisce o aggiorna un log giornaliero (idempotente) con step granulari."""
    query = """
    INSERT INTO daily_logs (
        entry_date, am_cleanser, am_azid, am_rederma, am_spf,
        pm_cleanser, pm_differin, pm_rederma, stinging_index, gym_workout, notes
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(entry_date) DO UPDATE SET
        am_cleanser = excluded.am_cleanser,
        am_azid = excluded.am_azid,
        am_rederma = excluded.am_rederma,
        am_spf = excluded.am_spf,
        pm_cleanser = excluded.pm_cleanser,
        pm_differin = excluded.pm_differin,
        pm_rederma = excluded.pm_rederma,
        stinging_index = excluded.stinging_index,
        gym_workout = excluded.gym_workout,
        notes = excluded.notes;
    """
    with conn:
        conn.execute(
            query,
            (
                log.entry_date.isoformat(),
                1 if log.am_cleanser else 0,
                1 if log.am_azid else 0,
                1 if log.am_rederma else 0,
                1 if log.am_spf else 0,
                1 if log.pm_cleanser else 0,
                1 if log.pm_differin else 0,
                1 if log.pm_rederma else 0,
                log.stinging_index,
                log.gym_workout.value,
                log.notes
            )
        )

def get_daily_log(conn: sqlite3.Connection, entry_date: date) -> Optional[DailyLogResponse]:
    """Recupera un log giornaliero per una data specifica, includendo se ha una foto."""
    query = """
    SELECT l.entry_date, l.am_cleanser, l.am_azid, l.am_rederma, l.am_spf,
           l.pm_cleanser, l.pm_differin, l.pm_rederma, l.stinging_index, l.gym_workout, l.notes,
           p.file_path IS NOT NULL as has_photo, p.entry_date as photo_date
    FROM daily_logs l
    LEFT JOIN photo_checkpoints p ON l.entry_date = p.entry_date
    WHERE l.entry_date = ?
    """
    cursor = conn.execute(query, (entry_date.isoformat(),))
    row = cursor.fetchone()
    if not row:
        return None
        
    photo_url = f"/static/photos/{row['photo_date']}.webp" if row["has_photo"] else None
    
    return DailyLogResponse(
        entry_date=date.fromisoformat(row["entry_date"]),
        am_cleanser=bool(row["am_cleanser"]),
        am_azid=bool(row["am_azid"]),
        am_rederma=bool(row["am_rederma"]),
        am_spf=bool(row["am_spf"]),
        pm_cleanser=bool(row["pm_cleanser"]),
        pm_differin=bool(row["pm_differin"]),
        pm_rederma=bool(row["pm_rederma"]),
        stinging_index=row["stinging_index"],
        gym_workout=GymWorkoutEnum(row["gym_workout"]),
        notes=row["notes"],
        has_photo=bool(row["has_photo"]),
        photo_url=photo_url
    )

def upsert_photo_checkpoint(
    conn: sqlite3.Connection,
    entry_date: date,
    file_path: str,
    ipd: float,
    pih_path: Optional[str] = None,
    texture_path: Optional[str] = None,
    analysis_json: Optional[str] = None,
    analysis_version: Optional[str] = None,
):
    """Inserisce o aggiorna un checkpoint fotografico (+ analisi opzionale).
    Se non esiste un log giornaliero associato, ne crea uno di default con tutti i passaggi a 0."""
    date_str = entry_date.isoformat()
    
    with conn:
        # 1. Assicura l'esistenza del log giornaliero (default se mancante) per rispettare la FK
        conn.execute(
            """
            INSERT OR IGNORE INTO daily_logs (
                entry_date, am_cleanser, am_azid, am_rederma, am_spf,
                pm_cleanser, pm_differin, pm_rederma, stinging_index, gym_workout, notes
            )
            VALUES (?, 0, 0, 0, 0, 0, 0, 0, 0, 'NONE', NULL)
            """,
            (date_str,)
        )
        
        # 2. Upsert del checkpoint fotografico
        conn.execute(
            """
            INSERT INTO photo_checkpoints (
                entry_date, file_path, resolution_width, resolution_height, interpupillary_distance,
                pih_path, texture_path, analysis_json, analysis_version
            )
            VALUES (?, ?, 2048, 2048, ?, ?, ?, ?, ?)
            ON CONFLICT(entry_date) DO UPDATE SET
                file_path = excluded.file_path,
                resolution_width = excluded.resolution_width,
                resolution_height = excluded.resolution_height,
                interpupillary_distance = excluded.interpupillary_distance,
                pih_path = excluded.pih_path,
                texture_path = excluded.texture_path,
                analysis_json = excluded.analysis_json,
                analysis_version = excluded.analysis_version;
            """,
            (date_str, file_path, ipd, pih_path, texture_path, analysis_json, analysis_version)
        )

def delete_photo_checkpoint(conn: sqlite3.Connection, entry_date: date):
    """Elimina un checkpoint e restituisce dict con path file da rimuovere, o None."""
    date_str = entry_date.isoformat()
    
    cursor = conn.execute(
        "SELECT file_path, pih_path, texture_path FROM photo_checkpoints WHERE entry_date = ?",
        (date_str,),
    )
    row = cursor.fetchone()
    if not row:
        return None
        
    paths = {
        "file_path": row["file_path"],
        "pih_path": row["pih_path"],
        "texture_path": row["texture_path"],
    }
    
    with conn:
        conn.execute("DELETE FROM photo_checkpoints WHERE entry_date = ?", (date_str,))
        
    return paths

def _parse_analysis_json(raw: Optional[str]) -> Optional[SkinAnalysisMetrics]:
    if not raw:
        return None
    try:
        data = json.loads(raw)
        return SkinAnalysisMetrics(**data)
    except Exception:
        return None


def get_timeline(conn: sqlite3.Connection) -> List[TimelineItem]:
    """Genera la timeline unendo log e checkpoint ordinati per data,
    calcolando giorni trascorsi dall'inizio e compliance rate frazionaria per ciascun ciclo."""
    checkpoints_query = """
    SELECT entry_date, file_path, interpupillary_distance,
           pih_path, texture_path, analysis_json, analysis_version
    FROM photo_checkpoints
    ORDER BY entry_date ASC
    """
    cursor = conn.execute(checkpoints_query)
    checkpoints = cursor.fetchall()
    
    if not checkpoints:
        return []
        
    from app.routine_config import get_routine_steps

    steps = get_routine_steps()
    am_ids = [s["id"] for s in steps.get("am", [])]
    pm_ids = [s["id"] for s in steps.get("pm", [])]

    logs_query = """
    SELECT entry_date, am_cleanser, am_azid, am_rederma, am_spf,
           pm_cleanser, pm_differin, pm_rederma, stinging_index
    FROM daily_logs
    ORDER BY entry_date ASC
    """
    cursor = conn.execute(logs_query)
    logs = cursor.fetchall()
    
    logs_map = {}
    for log in logs:
        am_n = max(1, len(am_ids))
        pm_n = max(1, len(pm_ids))
        am_score = sum(1 for sid in am_ids if log[sid]) / am_n
        pm_score = sum(1 for sid in pm_ids if log[sid]) / pm_n
        
        logs_map[date.fromisoformat(log["entry_date"])] = {
            "am_score": am_score,
            "pm_score": pm_score,
            "stinging_index": log["stinging_index"]
        }
        
    baseline_date = date.fromisoformat(checkpoints[0]["entry_date"])
    timeline_items = []
    
    for idx, cp in enumerate(checkpoints):
        cp_date = date.fromisoformat(cp["entry_date"])
        days_from_start = (cp_date - baseline_date).days
        
        if idx == 0:
            relevant_logs = [
                log_data for log_date, log_data in logs_map.items()
                if log_date <= cp_date
            ]
        else:
            cp_date_prev = date.fromisoformat(checkpoints[idx - 1]["entry_date"])
            relevant_logs = [
                log_data for log_date, log_data in logs_map.items()
                if cp_date_prev < log_date <= cp_date
            ]
            
        total_days = len(relevant_logs)
        if total_days > 0:
            am_compliance_rate = sum(log["am_score"] for log in relevant_logs) / total_days
            pm_compliance_rate = sum(log["pm_score"] for log in relevant_logs) / total_days
        else:
            am_compliance_rate = 1.0
            pm_compliance_rate = 1.0
            
        stinging_index = logs_map.get(cp_date, {}).get("stinging_index", 0)
        date_str = cp["entry_date"]
        photo_url = f"/static/photos/{date_str}.webp"
        pih_url = f"/static/photos/{date_str}_pih.webp" if cp["pih_path"] else None
        texture_url = f"/static/photos/{date_str}_tex.webp" if cp["texture_path"] else None
        
        timeline_items.append(
            TimelineItem(
                index=idx,
                entry_date=cp_date,
                days_from_start=days_from_start,
                photo_url=photo_url,
                pih_url=pih_url,
                texture_url=texture_url,
                analysis=_parse_analysis_json(cp["analysis_json"]),
                stinging_index=stinging_index,
                am_compliance_rate=am_compliance_rate,
                pm_compliance_rate=pm_compliance_rate
            )
        )
        
    return timeline_items


def get_skin_trend(conn: sqlite3.Connection) -> List[Dict[str, Any]]:
    """Serie temporale delle metriche cute per i grafici."""
    cursor = conn.execute(
        """
        SELECT entry_date, analysis_json
        FROM photo_checkpoints
        WHERE analysis_json IS NOT NULL
        ORDER BY entry_date ASC
        """
    )
    series = []
    for row in cursor.fetchall():
        metrics = _parse_analysis_json(row["analysis_json"])
        if not metrics:
            continue
        item = {"entry_date": row["entry_date"], **metrics.model_dump()}
        series.append(item)
    return series

def get_all_daily_logs(conn: sqlite3.Connection) -> List[DailyLogResponse]:
    """Recupera tutti i log giornalieri ordinati per data crescente."""
    query = """
    SELECT l.entry_date, l.am_cleanser, l.am_azid, l.am_rederma, l.am_spf,
           l.pm_cleanser, l.pm_differin, l.pm_rederma, l.stinging_index, l.gym_workout, l.notes,
           p.file_path IS NOT NULL as has_photo, p.entry_date as photo_date
    FROM daily_logs l
    LEFT JOIN photo_checkpoints p ON l.entry_date = p.entry_date
    ORDER BY l.entry_date ASC
    """
    cursor = conn.execute(query)
    rows = cursor.fetchall()
    
    logs = []
    for row in rows:
        photo_url = f"/static/photos/{row['photo_date']}.webp" if row["has_photo"] else None
        logs.append(
            DailyLogResponse(
                entry_date=date.fromisoformat(row["entry_date"]),
                am_cleanser=bool(row["am_cleanser"]),
                am_azid=bool(row["am_azid"]),
                am_rederma=bool(row["am_rederma"]),
                am_spf=bool(row["am_spf"]),
                pm_cleanser=bool(row["pm_cleanser"]),
                pm_differin=bool(row["pm_differin"]),
                pm_rederma=bool(row["pm_rederma"]),
                stinging_index=row["stinging_index"],
                gym_workout=GymWorkoutEnum(row["gym_workout"]),
                notes=row["notes"],
                has_photo=bool(row["has_photo"]),
                photo_url=photo_url
            )
        )
    return logs

