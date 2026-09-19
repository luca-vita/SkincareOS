#!/usr/bin/env python3
"""
SkinCare OS - Allineamento biometrico (computer vision)

Logica CV:
  - matrice affine di similitudine sugli iridi (landmarks 473 / 468)
  - quality gate (1 volto, tilt max)
  - warp Lanczos → canvas 2048×2048

Dipende da tools.face_landmarks per MediaPipe.
CLI: allinea un file e opzionalmente salva preview ROI (extract_rois).

  PYTHONPATH=. python cv_tools/align_photo.py input.jpg [output.webp]
"""

from __future__ import annotations

import io
import math
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps
from pillow_heif import register_heif_opener

register_heif_opener()

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import CANVAS_SIZE, D_TARGET, T_MID_X, T_MID_Y, MAX_TILT_DEG  # noqa: E402
from app.exceptions import FaceCountException, PoseAngleException, InvalidImageException  # noqa: E402
from cv_tools.face_landmarks import (  # noqa: E402
    close_landmarker,
    create_face_landmarker,
    detect_with_landmarker,
)

# =====================================================================
# Computer vision
# =====================================================================


def load_rgb_from_bytes(image_bytes: bytes) -> np.ndarray:
    """Byte immagine → RGB NumPy con EXIF transpose."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return np.array(img)
    except Exception as e:
        raise InvalidImageException(f"Impossibile decodificare l'immagine: {str(e)}")


def calculate_affine_matrix(
    p_l: tuple[float, float], p_r: tuple[float, float]
) -> tuple[np.ndarray, float, float]:
    """
    Matrice affine di similitudine 2×3 che mappa il segmento interpupillare
    sul target (IPD = D_TARGET, centro = T_MID).

    p_l / p_r: pupille in pixel (landmark 473 / 468).
    Ritorna (M, d_current, theta_rad).
    """
    x_l, y_l = p_l
    x_r, y_r = p_r

    x_mid = (x_l + x_r) / 2.0
    y_mid = (y_l + y_r) / 2.0
    dx = x_r - x_l
    dy = y_r - y_l

    theta = math.atan2(dy, dx)
    while theta > math.pi / 2:
        theta -= math.pi
    while theta < -math.pi / 2:
        theta += math.pi

    d_current = math.hypot(dx, dy)
    if d_current == 0:
        raise ValueError("La distanza interpupillare corrente non può essere zero.")

    s = D_TARGET / d_current
    alpha = s * math.cos(theta)
    beta = s * math.sin(theta)
    tx = (1.0 - alpha) * x_mid - beta * y_mid + (T_MID_X - x_mid)
    ty = beta * x_mid + (1.0 - alpha) * y_mid + (T_MID_Y - y_mid)

    M = np.array([[alpha, beta, tx], [-beta, alpha, ty]], dtype=np.float32)
    return M, d_current, theta


def align_face(
    img_rgb: np.ndarray, face_mesh_detector=None
) -> tuple[np.ndarray, float]:
    """
    Allinea un volto RGB al canvas biometrico.

    1. Face Landmarker (tools.face_landmarks)
    2. Quality gate: esattamente 1 volto, |tilt| ≤ MAX_TILT_DEG
    3. Affine + warp Lanczos → CANVAS_SIZE×CANVAS_SIZE

    face_mesh_detector: opzionale (mock nei test con .process()).
    Ritorna (warped_rgb, ipd_originale_px).
    """
    h, w, _ = img_rgb.shape
    owns_detector = face_mesh_detector is None

    try:
        if owns_detector:
            # num_faces=2: serve al quality gate (se >1 → rifiuto)
            face_mesh_detector = create_face_landmarker(num_faces=2)
            results = detect_with_landmarker(face_mesh_detector, img_rgb)
        else:
            results = face_mesh_detector.process(img_rgb)

        if not results.face_landmarks:
            raise FaceCountException("NO_FACE_DETECTED", "Nessun volto rilevato nell'immagine.")

        num_faces = len(results.face_landmarks)
        if num_faces > 1:
            raise FaceCountException(
                "MULTIPLE_FACES_DETECTED",
                f"Rilevati {num_faces} volti. È richiesto esattamente 1 volto.",
            )

        landmarks = results.face_landmarks[0]
        # 473 = iride dx soggetto (sinistra in foto → P_L)
        # 468 = iride sx soggetto (destra in foto → P_R)
        lm_l = landmarks[473]
        lm_r = landmarks[468]
        p_l = (lm_l.x * w, lm_l.y * h)
        p_r = (lm_r.x * w, lm_r.y * h)

        print(f"[DEBUG] Image dimensions: {w}x{h}")
        print(f"[DEBUG] Left Iris (473): {p_l}")
        print(f"[DEBUG] Right Iris (468): {p_r}")

        M, d_current, theta = calculate_affine_matrix(p_l, p_r)
        theta_deg = math.degrees(theta)
        print(f"[DEBUG] Calculated IPD (d_current): {d_current:.2f}px")
        print(f"[DEBUG] Calculated Theta (radians): {theta:.4f}, (degrees): {theta_deg:.2f}°")

        if abs(theta_deg) > MAX_TILT_DEG:
            raise PoseAngleException(
                f"Inclinazione della testa eccessiva ({abs(theta_deg):.1f}°). "
                f"Il limite massimo è {MAX_TILT_DEG}°."
            )

        warped = cv2.warpAffine(
            img_rgb,
            M,
            (CANVAS_SIZE, CANVAS_SIZE),
            flags=cv2.INTER_LANCZOS4,
        )
        return warped, d_current
    finally:
        if owns_detector and face_mesh_detector is not None:
            close_landmarker(face_mesh_detector)


# =====================================================================
# CLI / file I/O
# =====================================================================


def draw_patch_rois(img_rgb):
    """Preview RGB con poligoni ROI (fronte + guance)."""
    from cv_tools.extract_rois import draw_region_polygons

    return draw_region_polygons(img_rgb)


def _preview_path_from(aligned_path: str) -> str:
    base, _ = os.path.splitext(aligned_path)
    return f"{base}_rois.png"


def align_image(input_path, output_path=None, draw_rois=True):
    """
    Allinea un file immagine e salva WebP 2048.
    Se draw_rois=True, salva anche la preview PNG con i poligoni.
    Ritorna il path dell'immagine allineata, o None in caso di errore.
    """
    if not os.path.exists(input_path):
        print(f"[ERRORE] Il file specificato non esiste: {input_path}")
        return None

    print(f"[*] Caricamento e decodifica dell'immagine: {input_path}")
    try:
        with open(input_path, "rb") as f:
            img_rgb = load_rgb_from_bytes(f.read())
    except Exception as e:
        print(f"[ERRORE] Impossibile caricare l'immagine: {str(e)}")
        return None

    print("[*] Allineamento biometrico (Face Landmarker + warp)...")
    try:
        warped_img, ipd = align_face(img_rgb)
    except Exception as e:
        print(f"[ERRORE] Allineamento biometrico fallito: {str(e)}")
        return None

    if output_path is None:
        from app.config import BASE_DIR

        os.makedirs(BASE_DIR, exist_ok=True)
        output_path = os.path.join(BASE_DIR, "debug_aligned.webp")

    print(f"[*] Salvataggio (2048x2048 WebP): {output_path}")
    try:
        Image.fromarray(warped_img).save(output_path, format="WEBP", quality=90, lossless=False)
        print(f"[+] Allineamento completato. IPD originale: {ipd:.2f}px")
    except Exception as e:
        print(f"[ERRORE] Impossibile salvare l'immagine di output: {str(e)}")
        return None

    if draw_rois:
        preview_path = _preview_path_from(output_path)
        print("[*] Disegno dei poligoni Face Mesh (fronte + guance)...")
        try:
            annotated = draw_patch_rois(warped_img)
            Image.fromarray(annotated).save(preview_path, format="PNG")
            print(f"[+] Preview con poligoni salvata in: {preview_path}")
        except Exception as e:
            print(f"[AVVISO] Preview poligoni non salvata: {e}")

    return output_path


def main():
    import argparse

    parser = argparse.ArgumentParser(description="SkinCare OS - Biometric Alignment Tool")
    parser.add_argument("input_path", help="Percorso dell'immagine di input da allineare")
    parser.add_argument("output_path", nargs="?", help="Percorso output WebP (opzionale)")
    parser.add_argument(
        "--no-rois",
        action="store_true",
        help="Non salvare la preview con i poligoni ROI",
    )
    args = parser.parse_args()
    align_image(args.input_path, args.output_path, draw_rois=not args.no_rois)


if __name__ == "__main__":
    main()
