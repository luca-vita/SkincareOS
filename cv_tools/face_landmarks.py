#!/usr/bin/env python3
"""
SkinCare OS - Face Landmarker MediaPipe (Tasks API)

Unica sede per download modello + rilevamento landmark.
Usato da extract_rois e da align_photo (allineamento biometrico).
"""

from __future__ import annotations

import gc
import os
import urllib.request

import numpy as np

from app.config import MODEL_PATH


def download_model_if_missing():
    """Scarica face_landmarker.task se assente in MODEL_PATH."""
    if os.path.exists(MODEL_PATH):
        return
    print(f"[+] Modello {MODEL_PATH} non trovato. Download in corso...")
    url = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/1/face_landmarker.task"
    )
    try:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        temp_model_path = MODEL_PATH + ".tmp"
        urllib.request.urlretrieve(url, temp_model_path)
        os.replace(temp_model_path, MODEL_PATH)
        print("[+] Download del modello completato con successo!")
    except Exception as e:
        if os.path.exists(MODEL_PATH + ".tmp"):
            os.remove(MODEL_PATH + ".tmp")
        raise RuntimeError(f"Impossibile scaricare il modello di MediaPipe da {url}: {str(e)}")


def create_face_landmarker(num_faces: int = 1):
    """Crea un FaceLandmarker Tasks. Caller deve .close()."""
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    download_model_if_missing()
    options = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
        num_faces=num_faces,
    )
    return vision.FaceLandmarker.create_from_options(options)


def detect_with_landmarker(landmarker, img_rgb: np.ndarray):
    """Esegue detect su immagine RGB uint8."""
    import mediapipe as mp

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    return landmarker.detect(mp_image)


def close_landmarker(landmarker) -> None:
    if landmarker is None:
        return
    try:
        landmarker.close()
    except Exception:
        pass
    gc.collect()


def landmarks_to_px(face_landmarks, width: int, height: int) -> np.ndarray:
    """Lista landmark MediaPipe → array (N, 2) float32 in pixel."""
    return np.array(
        [(lm.x * width, lm.y * height) for lm in face_landmarks],
        dtype=np.float32,
    )


def detect_face_landmarks_px(img_rgb: np.ndarray, num_faces: int = 1) -> np.ndarray:
    """
    Rileva i landmark del primo volto in pixel.
    Richiede esattamente ≥1 volto (alza RuntimeError se nessuno).
    """
    h, w = img_rgb.shape[:2]
    landmarker = None
    try:
        landmarker = create_face_landmarker(num_faces=num_faces)
        results = detect_with_landmarker(landmarker, img_rgb)
    finally:
        close_landmarker(landmarker)

    if not results.face_landmarks:
        raise RuntimeError("Nessun volto rilevato sull'immagine normalizzata.")

    return landmarks_to_px(results.face_landmarks[0], w, h)
