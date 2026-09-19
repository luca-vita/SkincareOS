from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, status
from datetime import date
import sqlite3
import time
import os
import numpy as np
from PIL import Image
from app.database import get_db_connection, upsert_photo_checkpoint, delete_photo_checkpoint
from app.cv_pipeline import save_image_atomically
from cv_tools.align_photo import align_face, load_rgb_from_bytes
from app.services.photo_analysis import run_photo_analysis, delete_analysis_overlays
from app.exceptions import DatabaseBusyException

router = APIRouter(prefix="/api/v1/photos", tags=["Photos"])

def get_db():
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()

@router.post("/upload", status_code=status.HTTP_201_CREATED, summary="Ingest, align, and analyze weekly checkpoint photo")
async def upload_photo(
    entry_date: date = Query(..., description="La data del checkpoint nel formato YYYY-MM-DD"),
    file: UploadFile = File(..., description="Il file d'immagine (JPEG, PNG, HEIC)"),
    db: sqlite3.Connection = Depends(get_db)
):
    """
    Carica, allinea biometricamente (2048 WebP), estrae ROI, calcola PIH+texture,
    salva overlay pre-renderizzati e metriche. Sync end-to-end.

    Ordine obbligatorio: align → salva plain WebP → ricarica da disco → analizza.
    Così PIH/texture sono calibrati sugli stessi pixel che vede CLI/UI.
    """
    try:
        content = await file.read()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Impossibile leggere il file caricato."
        )

    img_rgb = load_rgb_from_bytes(content)
    warped_img, ipd = align_face(img_rgb)

    date_str = entry_date.isoformat()
    file_path = save_image_atomically(warped_img, date_str)

    # Analizza i pixel persistiti (stesso WebP che vede CLI/UI), non il buffer
    # in-memory pre-compressione: altrimenti metriche/overlay possono divergere.
    try:
        persisted_rgb = np.array(Image.open(file_path).convert("RGB"))
        analysis_result = run_photo_analysis(persisted_rgb, date_str)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Allineamento ok, ma analisi cute fallita: {str(e)}"
        )

    max_retries = 3
    retry_delay = 0.5

    for attempt in range(max_retries):
        try:
            upsert_photo_checkpoint(
                db,
                entry_date,
                file_path,
                ipd,
                pih_path=analysis_result["pih_path"],
                texture_path=analysis_result["texture_path"],
                analysis_json=analysis_result["analysis_json"],
                analysis_version=analysis_result["analysis_version"],
            )
            break
        except DatabaseBusyException as dbe:
            if attempt == max_retries - 1:
                raise dbe
            time.sleep(retry_delay * (2 ** attempt))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Errore imprevisto durante il salvataggio nel database: {str(e)}"
            )

    return {
        "status": "success",
        "message": "Photo aligned and analyzed",
        "entry_date": date_str,
        "file_path": file_path,
        "interpupillary_distance": ipd,
        "pih_url": analysis_result["pih_url"],
        "texture_url": analysis_result["texture_url"],
        "analysis": analysis_result["analysis"],
    }

@router.delete("/delete", status_code=status.HTTP_200_OK, summary="Delete weekly checkpoint photo")
def delete_photo(
    entry_date: date = Query(..., description="La data del checkpoint da eliminare nel formato YYYY-MM-DD"),
    db: sqlite3.Connection = Depends(get_db)
):
    """Elimina checkpoint DB + plain + overlay PIH/texture."""
    max_retries = 3
    retry_delay = 0.5
    paths = None

    for attempt in range(max_retries):
        try:
            paths = delete_photo_checkpoint(db, entry_date)
            break
        except DatabaseBusyException as dbe:
            if attempt == max_retries - 1:
                raise dbe
            time.sleep(retry_delay * (2 ** attempt))
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Errore imprevisto durante l'eliminazione dal database: {str(e)}"
            )

    if not paths:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Nessun checkpoint fotografico trovato per la data specificata."
        )

    for key in ("file_path", "pih_path", "texture_path"):
        path = paths.get(key)
        if not path:
            continue
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"[WARNING] Impossibile rimuovere fisicamente {path}: {str(e)}")

    delete_analysis_overlays(entry_date.isoformat())

    return {
        "status": "success",
        "message": "Photo checkpoint deleted successfully",
        "entry_date": entry_date.isoformat()
    }
