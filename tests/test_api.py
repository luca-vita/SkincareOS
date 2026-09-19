import os
import pytest
from fastapi.testclient import TestClient
from datetime import date, timedelta
import sqlite3
import numpy as np
from PIL import Image
import io

from app.main import app
from app.database import get_db_connection, init_db
from app.config import DB_PATH

# Forza l'uso di un database di test separato
TEST_DB_PATH = os.path.join(os.path.dirname(DB_PATH), "test_app.db")

@pytest.fixture(autouse=True)
def setup_test_db(monkeypatch):
    """Fixture che inizializza un database di test pulito prima di ogni test."""
    # Sovrascrive il DB_PATH in app.config per puntare al database di test
    monkeypatch.setattr("app.config.DB_PATH", TEST_DB_PATH)
    monkeypatch.setattr("app.database.DB_PATH", TEST_DB_PATH)
    
    # Rimuove il DB di test se esiste già
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except OSError:
            pass
            
    # Inizializza lo schema nel DB di test
    init_db()
    
    yield
    
    # Pulisce dopo il test
    if os.path.exists(TEST_DB_PATH):
        try:
            os.remove(TEST_DB_PATH)
        except OSError:
            pass

client = TestClient(app)

def test_upsert_daily_log_idempotent():
    """Verifica che l'inserimento di un log sia idempotente (UPSERT) con i 7 step granulari."""
    log_date = "2026-09-08"
    payload = {
        "entry_date": log_date,
        "am_cleanser": True,
        "am_azid": True,
        "am_rederma": True,
        "am_spf": True,
        "pm_cleanser": False,
        "pm_differin": False,
        "pm_rederma": False,
        "stinging_index": 1,
        "gym_workout": "MORNING",
        "notes": "Primo inserimento"
    }
    
    # Primo inserimento
    response = client.post("/api/v1/logs", json=payload)
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    
    # Verifica che sia stato salvato correttamente
    response = client.get(f"/api/v1/logs/{log_date}")
    assert response.status_code == 200
    data = response.json()
    assert data["notes"] == "Primo inserimento"
    assert data["am_cleanser"] is True
    assert data["am_azid"] is True
    assert data["pm_cleanser"] is False
    assert data["has_photo"] is False
    
    # Secondo inserimento (UPSERT) sulla stessa data con modifiche
    payload["notes"] = "Nota aggiornata"
    payload["pm_cleanser"] = True
    payload["pm_differin"] = True
    response = client.post("/api/v1/logs", json=payload)
    assert response.status_code == 200
    
    # Verifica l'aggiornamento
    response = client.get(f"/api/v1/logs/{log_date}")
    assert response.status_code == 200
    data = response.json()
    assert data["notes"] == "Nota aggiornata"
    assert data["pm_cleanser"] is True
    assert data["pm_differin"] is True

def test_get_log_not_found():
    """Verifica che la richiesta di un log non esistente restituisca 404."""
    response = client.get("/api/v1/logs/2026-01-01")
    assert response.status_code == 404
    assert "Nessun log trovato" in response.json()["detail"]

def test_get_timeline_empty():
    """Verifica che la timeline sia vuota all'inizio."""
    response = client.get("/api/v1/timeline")
    assert response.status_code == 200
    assert response.json() == []

# Mock per bypassare MediaPipe nei test di integrazione API
class DummyLandmark:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class DummyFaceLandmarks:
    def __init__(self, landmarks):
        self.landmark = landmarks

class DummyMultiFaceLandmarks:
    def __init__(self, faces):
        self.multi_face_landmarks = faces

class DummyFaceMeshDetector:
    def __init__(self, faces_list):
        self.faces_list = faces_list
        
    def process(self, img_rgb):
        return DummyMultiFaceLandmarks(self.faces_list)

def test_upload_photo_no_face_detected(monkeypatch):
    """Verifica che l'upload di un'immagine senza volto restituisca HTTP 422 NO_FACE_DETECTED."""
    from app.exceptions import FaceCountException
    def mock_run_pipeline(img, detector=None):
        raise FaceCountException("NO_FACE_DETECTED", "Nessun volto rilevato.")
        
    monkeypatch.setattr("app.routers.photos.align_face", mock_run_pipeline)
    
    # Crea un'immagine fittizia in memoria
    img = Image.new("RGB", (100, 100), color="red")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    
    response = client.post(
        "/api/v1/photos/upload?entry_date=2026-09-08",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    
    assert response.status_code == 422
    assert response.json()["detail"] == "NO_FACE_DETECTED"

def test_upload_photo_success_and_timeline_hydration(monkeypatch, tmp_path):
    """Verifica l'upload con successo, l'auto-creazione del log di default e l'idratazione della timeline."""
    # Imposta una cartella temporanea per le foto salvate nei test
    test_photos_dir = str(tmp_path / "photos")
    os.makedirs(test_photos_dir, exist_ok=True)
    monkeypatch.setattr("app.config.PHOTOS_DIR", test_photos_dir)
    monkeypatch.setattr("app.services.photo_analysis.PHOTOS_DIR", test_photos_dir)
    
    # Mock del pipeline di successo
    dummy_warped = np.zeros((2048, 2048, 3), dtype=np.uint8)
    dummy_ipd = 150.0
    
    monkeypatch.setattr(
        "app.routers.photos.align_face",
        lambda img, detector=None: (dummy_warped, dummy_ipd)
    )

    photo_date = "2026-09-15"
    pih_path = os.path.join(test_photos_dir, f"{photo_date}_pih.webp")
    tex_path = os.path.join(test_photos_dir, f"{photo_date}_tex.webp")
    analysis = {
        "analysis_version": "pih1-tex1",
        "active_spots_count": 2,
        "active_area_pct": 10.0,
        "affected_area_pct": 5.5,
        "residual_area_pct": 4.0,
        "discromia_score": 12.3,
        "peak_p98": 0.05,
        "pore_prominence_pct": 8.1,
        "roughness_index": 3.2,
        "smoothness_score": 87.0,
    }

    def mock_run_photo_analysis(warped, date_str):
        # Scrive overlay fittizi sul disco
        Image.new("RGB", (8, 8), color="purple").save(pih_path, format="WEBP")
        Image.new("RGB", (8, 8), color="orange").save(tex_path, format="WEBP")
        import json
        return {
            "pih_path": pih_path,
            "texture_path": tex_path,
            "analysis": analysis,
            "analysis_version": "pih1-tex1",
            "analysis_json": json.dumps(analysis),
            "pih_url": f"/static/photos/{date_str}_pih.webp",
            "texture_url": f"/static/photos/{date_str}_tex.webp",
        }

    monkeypatch.setattr("app.routers.photos.run_photo_analysis", mock_run_photo_analysis)
    
    # Crea un'immagine fittizia in memoria
    img = Image.new("RGB", (100, 100), color="blue")
    img_bytes = io.BytesIO()
    img.save(img_bytes, format="JPEG")
    img_bytes.seek(0)
    
    # Esegue l'upload per una data per cui NON esiste ancora un log giornaliero
    response = client.post(
        f"/api/v1/photos/upload?entry_date={photo_date}",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")}
    )
    
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "success"
    assert body["interpupillary_distance"] == 150.0
    assert body["pih_url"] == f"/static/photos/{photo_date}_pih.webp"
    assert body["texture_url"] == f"/static/photos/{photo_date}_tex.webp"
    assert body["analysis"]["discromia_score"] == 12.3
    
    # Verifica che il file WebP sia stato effettivamente scritto sul disco
    expected_file = os.path.join(test_photos_dir, f"{photo_date}.webp")
    assert os.path.exists(expected_file)
    assert os.path.exists(pih_path)
    assert os.path.exists(tex_path)
    
    # Verifica che sia stato auto-creato un log giornaliero di default per rispettare la FK
    response = client.get(f"/api/v1/logs/{photo_date}")
    assert response.status_code == 200
    log_data = response.json()
    assert log_data["am_cleanser"] is False
    assert log_data["am_azid"] is False
    assert log_data["pm_cleanser"] is False
    assert log_data["stinging_index"] == 0
    assert log_data["gym_workout"] == "NONE"
    assert log_data["has_photo"] is True
    assert log_data["photo_url"] == f"/static/photos/{photo_date}.webp"
    
    # Verifica che la timeline ora contenga questo checkpoint
    response = client.get("/api/v1/timeline")
    assert response.status_code == 200
    timeline = response.json()
    assert len(timeline) == 1
    item = timeline[0]
    assert item["entry_date"] == photo_date
    assert item["days_from_start"] == 0  # Essendo l'unica, è la baseline
    assert item["photo_url"] == f"/static/photos/{photo_date}.webp"
    assert item["pih_url"] == f"/static/photos/{photo_date}_pih.webp"
    assert item["texture_url"] == f"/static/photos/{photo_date}_tex.webp"
    assert item["analysis"]["smoothness_score"] == 87.0
    assert item["am_compliance_rate"] == 0.0  # Il log di default ha tutti i passaggi a False
    assert item["pm_compliance_rate"] == 0.0

    # Skin trend endpoint
    response = client.get("/api/v1/metrics/skin-trend")
    assert response.status_code == 200
    trend = response.json()
    assert len(trend) == 1
    assert trend[0]["entry_date"] == photo_date
    assert trend[0]["discromia_score"] == 12.3
    assert trend[0]["affected_area_pct"] == 5.5
    assert trend[0]["smoothness_score"] == 87.0

    # Test eliminazione foto (rimuove plain + overlay)
    response = client.delete(f"/api/v1/photos/delete?entry_date={photo_date}")
    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert not os.path.exists(expected_file)
    assert not os.path.exists(pih_path)
    assert not os.path.exists(tex_path)

    # Verifica che il log giornaliero esista ancora ma has_photo sia False
    response = client.get(f"/api/v1/logs/{photo_date}")
    assert response.status_code == 200
    assert response.json()["has_photo"] is False
    assert response.json()["photo_url"] is None

    # Timeline e trend vuoti dopo delete
    assert client.get("/api/v1/timeline").json() == []
    assert client.get("/api/v1/metrics/skin-trend").json() == []

def test_therapeutic_pressure_metrics():
    """Verifica il calcolo della pressione terapeutica su finestra mobile di 14 giorni."""
    # Con 0 log nel DB, l'endpoint restituisce indici a 0.0% senza sollevare errori di divisione per zero
    response = client.get("/api/v1/metrics/therapeutic-pressure?target_date=2026-09-09")
    assert response.status_code == 200
    data = response.json()
    assert data["days_recorded"] == 0
    assert data["window_days"] == 14
    axes = {a["id"]: a for a in data["axes"]}
    assert set(axes) >= {"adapalene", "azelaic", "rederma"}
    for axis in axes.values():
        assert axis["current_pct"] == 0.0
        assert axis["completed_slots"] == 0
        assert axis["is_compliant"] is False

    # Inseriamo un log con aderenza totale per la data target
    payload = {
        "entry_date": "2026-09-09",
        "am_cleanser": True,
        "am_azid": True,
        "am_rederma": True,
        "am_spf": True,
        "pm_cleanser": True,
        "pm_differin": True,
        "pm_rederma": True,
        "stinging_index": 0,
        "gym_workout": "NONE",
        "notes": "Test"
    }
    response = client.post("/api/v1/logs", json=payload)
    assert response.status_code == 200

    # Ora verifichiamo che le metriche riflettano questo log (1 giorno su 14 = 7.1%, 2/28 = 7.1% per rederma)
    response = client.get("/api/v1/metrics/therapeutic-pressure?target_date=2026-09-09")
    assert response.status_code == 200
    data = response.json()
    assert data["days_recorded"] == 1
    axes = {a["id"]: a for a in data["axes"]}
    assert axes["adapalene"]["completed_slots"] == 1
    assert axes["adapalene"]["current_pct"] == round((1 / 14.0) * 100, 1)
    assert axes["azelaic"]["completed_slots"] == 1
    assert axes["azelaic"]["current_pct"] == round((1 / 14.0) * 100, 1)
    assert axes["rederma"]["completed_slots"] == 2  # am_rederma + pm_rederma
    assert axes["rederma"]["current_pct"] == round((2 / 28.0) * 100, 1)
