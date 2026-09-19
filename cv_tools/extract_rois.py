#!/usr/bin/env python3
"""
SkinCare OS - Estrazione ROI anatomiche (Face Mesh)

Geometria calibrata fronte + guance da landmark MediaPipe.
Usato da align_photo (preview), debug_pih_spectral, debug_texture.
Landmark detection: cv_tools.face_landmarks (usato anche da align_photo).

CLI:
  PYTHONPATH=. python cv_tools/extract_rois.py debug_aligned.webp
  → salva debug_aligned_rois.png (preview poligoni)
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =====================================================================
# CONFIGURAZIONE ROI
# =====================================================================
# Fronte: convex hull. Guance: poligono in ordine di bordo.
REGION_LANDMARK_IDS = {
    "Fronte Neutra": [
        10, 67, 109, 103, 9, 107, 66, 105, 63, 70,
        336, 296, 334, 293, 300, 332, 338, 297,
    ],
    "Guancia Destra": [
        111, 117, 118, 119, 120, 121,
        47, 126, 142, 203, 206, 216,
        135, 192, 213, 147, 137, 116,
    ],
    "Guancia Sinistra": [
        350, 349, 348, 347, 346, 345,
        352, 376, 433, 416, 364, 436,
        426, 423, 371, 277,
    ],
}

REGION_USE_CONVEX_HULL = {
    "Fronte Neutra": True,
    "Guancia Destra": False,
    "Guancia Sinistra": False,
}

# Attaccatura: lift + tetto Y sul canvas 2048 allineato.
FOREHEAD_TOP_IDS = {10, 67, 109, 338, 297}
FOREHEAD_TOP_HALF_LIFT_IDS = {103, 332}
FOREHEAD_TOP_HALF_LIFT_FRAC = 0.5
FOREHEAD_GLABELLA_ID = 9
FOREHEAD_TOP_LIFT_FRAC = 0.40
FOREHEAD_TOP_LIFT_MIN_PX = 110
FOREHEAD_TOP_TARGET_Y = 190.0
FOREHEAD_TOP_HALF_TARGET_Y = 260.0
FOREHEAD_BOTTOM_IDS = {9, 107, 66, 105, 63, 70, 336, 296, 334, 293, 300}
FOREHEAD_BOTTOM_INSET_PX = 48
FOREHEAD_MAX_BELOW_GLABELLA_PX = 10

EYEBROW_LANDMARK_IDS = (
    (70, 63, 105, 66, 107, 55, 65, 52, 53, 46),
    (300, 293, 334, 296, 336, 285, 295, 282, 283, 276),
)
EYEBROW_DILATE_KERNEL = (49, 130)

POLYGON_COLOR = (0, 220, 80)

# =====================================================================


def detect_face_landmarks_px(img_rgb):
    """Landmark Face Mesh in pixel — delega a cv_tools.face_landmarks."""
    from cv_tools.face_landmarks import detect_face_landmarks_px as _detect

    return _detect(img_rgb)


def _clip_xy(pts, image_shape):
    if image_shape is None or pts.size == 0:
        return pts
    h, w = image_shape[:2]
    pts = pts.copy()
    pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
    return pts


def region_hull_points(landmarks_px, name, landmark_ids, image_shape=None):
    """
    Poligono regione (N, 1, 2) int32.
    Fronte: lift attaccatura, inset sopracciglia, scarta punti sotto glabella.
    Guance: bordo ordinato dei landmark.
    """
    pts = []
    n = len(landmarks_px)
    glabella_y = float(landmarks_px[FOREHEAD_GLABELLA_ID][1]) if n > FOREHEAD_GLABELLA_ID else None
    lift_px = 0.0
    y_limit = None
    if name == "Fronte Neutra" and n > 10 and glabella_y is not None:
        lift_px = max(
            FOREHEAD_TOP_LIFT_MIN_PX,
            FOREHEAD_TOP_LIFT_FRAC * (glabella_y - float(landmarks_px[10][1])),
        )
        y_limit = glabella_y + FOREHEAD_MAX_BELOW_GLABELLA_PX

    for i in landmark_ids:
        if i >= n:
            continue
        p = np.array(landmarks_px[i], dtype=np.float32)
        if name == "Fronte Neutra":
            if y_limit is not None and p[1] > y_limit:
                continue
            if i in FOREHEAD_TOP_IDS:
                p[1] -= lift_px
                p[1] = min(p[1], FOREHEAD_TOP_TARGET_Y)
            elif i in FOREHEAD_TOP_HALF_LIFT_IDS:
                p[1] -= lift_px * FOREHEAD_TOP_HALF_LIFT_FRAC
                p[1] = min(p[1], FOREHEAD_TOP_HALF_TARGET_Y)
            if i in FOREHEAD_BOTTOM_IDS:
                p[1] -= FOREHEAD_BOTTOM_INSET_PX
        pts.append(p)

    if len(pts) < 3:
        raise RuntimeError(f"Troppi pochi landmark validi per il poligono di '{name}'.")

    pts = _clip_xy(np.stack(pts, axis=0), image_shape)
    pts_i32 = np.round(pts).astype(np.int32).reshape((-1, 1, 2))
    if REGION_USE_CONVEX_HULL.get(name, True):
        return cv2.convexHull(pts_i32)
    return pts_i32


def build_region_polygons(landmarks_px, image_shape=None):
    """Mappa nome regione → poligono."""
    return {
        name: region_hull_points(landmarks_px, name, ids, image_shape=image_shape)
        for name, ids in REGION_LANDMARK_IDS.items()
    }


def fill_polygon_mask(shape_hw, points):
    mask = np.zeros(shape_hw[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    return mask


def eyebrow_exclusion_mask(shape_hw, landmarks_px):
    """Peluria sopraccigliare + ombra sotto l'arcata (da sottrarre alla fronte)."""
    mask = np.zeros(shape_hw[:2], dtype=np.uint8)
    n = len(landmarks_px)
    for ids in EYEBROW_LANDMARK_IDS:
        pts = []
        for i in ids:
            if i < n:
                pts.append(landmarks_px[i])
        if len(pts) < 3:
            continue
        pts_i32 = np.round(np.array(pts, dtype=np.float32)).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [cv2.convexHull(pts_i32)], 255)
        cv2.polylines(mask, [pts_i32], isClosed=False, color=255, thickness=18)
    kw, kh = EYEBROW_DILATE_KERNEL
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kw, kh))
    return cv2.dilate(mask, kernel)


def extract_rois(img_rgb):
    """
    Estrae landmark, poligoni e maschere ROI (fronte senza sopracciglia).

    Returns dict:
      landmarks_px, polygons, masks (nome → uint8), combined_mask, eyebrow_mask
    """
    landmarks_px = detect_face_landmarks_px(img_rgb)
    polygons = build_region_polygons(landmarks_px, image_shape=img_rgb.shape)
    brow_excl = eyebrow_exclusion_mask(img_rgb.shape, landmarks_px)

    masks = {}
    combined = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    for name, points in polygons.items():
        mask = fill_polygon_mask(img_rgb.shape, points)
        if name == "Fronte Neutra":
            mask[brow_excl > 0] = 0
        masks[name] = mask
        combined = np.maximum(combined, mask)

    return {
        "landmarks_px": landmarks_px,
        "polygons": polygons,
        "masks": masks,
        "combined_mask": combined,
        "eyebrow_mask": brow_excl,
    }


def draw_region_polygons(img_rgb, polygons=None, landmarks_px=None):
    """Copia RGB con perimetri verdi (preview align_photo / CLI)."""
    annotated = img_rgb.copy()
    if polygons is None:
        if landmarks_px is None:
            landmarks_px = detect_face_landmarks_px(img_rgb)
        polygons = build_region_polygons(landmarks_px, image_shape=img_rgb.shape)

    overlay = annotated.copy()
    for points in polygons.values():
        cv2.fillPoly(overlay, [points], POLYGON_COLOR)
        cv2.polylines(
            annotated, [points], isClosed=True, color=POLYGON_COLOR, thickness=6, lineType=cv2.LINE_AA
        )
    cv2.addWeighted(overlay, 0.18, annotated, 0.82, 0, annotated)
    return annotated


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="SkinCare OS - Estrae ROI Face Mesh (fronte + guance)"
    )
    parser.add_argument("img_path", help="Immagine allineata (idealmente 2048x2048)")
    parser.add_argument(
        "--out",
        default=None,
        help="Path PNG preview (default: <img>_rois.png)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.img_path):
        print(f"[ERRORE] File non trovato: {args.img_path}")
        sys.exit(1)

    img_bgr = cv2.imread(args.img_path)
    if img_bgr is None:
        print("[ERRORE] Impossibile caricare l'immagine.")
        sys.exit(1)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    print("[*] Estrazione ROI Face Mesh...")
    rois = extract_rois(img_rgb)
    for name, mask in rois["masks"].items():
        print(f"  - {name}: {int(np.count_nonzero(mask))} px")

    preview = draw_region_polygons(img_rgb, polygons=rois["polygons"])
    out_path = args.out
    if not out_path:
        base, _ = os.path.splitext(args.img_path)
        out_path = f"{base}_rois.png"
    cv2.imwrite(out_path, cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
    print(f"[+] Preview ROI: {out_path}")


if __name__ == "__main__":
    main()
