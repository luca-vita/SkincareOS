#!/usr/bin/env python3
"""
SkinCare OS - PIH spectral playground (maschere poligonali Face Mesh)

Copia di debug_pih.py: stessa ROI/overlay/layout, ma la grandezza di misura
è lo sbilanciamento spettrale OD_G − OD_R (anti-ombra) al posto di L*.

Output in root del progetto (default):
  - debug_pih_spectral_result.png  — ROI + delta spettrale + heatmap
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# =====================================================================
# CONFIGURAZIONE E PARAMETRI DI CALIBRAZIONE (REGOLABILI)
# =====================================================================
# Fronte: convex hull (geometria già calibrata).
# Guance: poligono in ordine di bordo (curva sotto-occhiaia → naso → baffi → zigomo).
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
        350, 349,  348, 347, 346, 345,
        352, 376, 433, 416, 364, 436,
        426, 423, 371, 277,
    ],
}

# True = convex hull; False = collega i landmark nell'ordine della lista.
REGION_USE_CONVEX_HULL = {
    "Fronte Neutra": True,
    "Guancia Destra": False,
    "Guancia Sinistra": False,
}

# Attaccatura: la mesh 10 sta troppo in basso rispetto alla fronte visibile.
# Il lift percentuale collassa se 10 è già vicino alla glabella (ROI a striscia):
# quindi c'è un minimo in px e un tetto Y sul canvas allineato 2048.
FOREHEAD_TOP_IDS = {10, 67, 109, 338, 297}
FOREHEAD_GLABELLA_ID = 9
FOREHEAD_TOP_LIFT_FRAC = 0.40
FOREHEAD_TOP_LIFT_MIN_PX = 110
FOREHEAD_TOP_TARGET_Y = 190.0
# Bordo basso sopra le sopracciglia (più inset = meno artefatti da ombra del sopracciglio).
FOREHEAD_BOTTOM_IDS = {9, 107, 66, 105, 63, 70, 336, 296, 334, 293, 300}
FOREHEAD_BOTTOM_INSET_PX = 48
FOREHEAD_MAX_BELOW_GLABELLA_PX = 10
# Tempia / zigomo: allargano la fronte sulle zone in ombra.
FOREHEAD_SKIP_IDS = {21, 54, 162, 127, 234, 8, 251, 284, 389, 356, 454}

# Fascia sopraccigliare da sottrarre alla maschera (pelo + ombra sotto l'arcata).
EYEBROW_LANDMARK_IDS = (
    (70, 63, 105, 66, 107, 55, 65, 52, 53, 46),
    (300, 293, 334, 296, 336, 285, 295, 282, 283, 276),
)
EYEBROW_DILATE_KERNEL = (49, 130)

# PIH spectral: OD_G−OD_R locale + gating a* (eritema) → Δ segnale vs fondo
ERYTHEMA_A_OFFSET = 100.0      # Δa* rispetto alla mediana ROI (brufoli / infiammazione)
BG_BLUR_SIGMA = 22.0         # fondo Gaussiano normalizzato (px)
OD_EPSILON = 1e-4
# Soglie solo per metriche continue / overlay (non alterano il calcolo del delta).
# Soglie ricalibrate sulla scala OD (p98≈0.017 vs Weber≈0.051).
# Target: leggermente sotto il PIH classico (affected ~4.2% vs ~4.8%).
# Soglia unica score + overlay: così mappa e PIH_score combaciano.
TAU_NOISE = 0.004            # sotto questa soglia: né score né overlay
AFFECTED_AREA_THRESHOLD = 0.0140  # soglia percettiva per Affected Area Ratio (%)
HEATMAP_COLORMAP = cv2.COLORMAP_MAGMA  # legacy (se USE_FLAT_MARKS_OVERLAY=False)
HEATMAP_NORM_PERCENTILE = 90.0  # satura i colori prima (intensità più alta)
# Niente opening aggressivo: spezza le macchie a Y (rami sottili).
OVERLAY_CLOSE_KERNEL = 3     # solo micro-closing dopo il filtro area
OVERLAY_MIN_AREA = 200       # scarta peletti (px)
OVERLAY_COLOR_SIGMA = 4.0
OVERLAY_EDGE_SIGMA = 2.5

# --- Rendering sperimentale (False = torna a Magma + fill cyan) ---
USE_FLAT_MARKS_OVERLAY = True   # viola flat + alpha ∝ intensità
USE_ACTIVE_CIRCLES = True       # cerchi rossi sui centroidi active
COLOR_MARKS_RGB = (168, 70, 210)   # viola marks
MARKS_ALPHA_MIN = 0.15             # floor visibile sulle macchie deboli
MARKS_ALPHA_MAX = 1.0              # picchi pienamente opachi (prima 0.72, troppo piatto)
COLOR_ACTIVE_RING_RGB = (230, 35, 35)  # rosso cerchi active
ACTIVE_RING_THICKNESS = 3
ACTIVE_RING_MIN_RADIUS = 14
ACTIVE_RING_RADIUS_SCALE = 1.35    # r ≈ scale * sqrt(area/π)
ACTIVE_RING_PADDING_PX = 8         # aria extra intorno al brufolo

# Active
ACTIVE_THRESHOLD = 0.040     # brufoli attivi / infiammazione acuta
ACTIVE_MIN_AREA = 15         # CC min area (px) per contare uno spot
OVERLAY_ALPHA_ACTIVE = 0.85  # solo se USE_ACTIVE_CIRCLES=False (fill legacy)
COLOR_ACTIVE_RGB = (0, 255, 255)  # cyan fill legacy

POLYGON_COLOR = (0, 220, 80)

# =====================================================================


def detect_face_landmarks_px(img_rgb):
    """
    Rileva i landmark Face Mesh sull'immagine RGB e li restituisce in pixel.

    MediaPipe 1.x non espone più mp.solutions.face_mesh: si usa FaceLandmarker
    (Tasks API), che restituisce gli stessi 468 indici della mesh classica.
    """
    import gc
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    from app.config import MODEL_PATH
    from app.cv_pipeline import download_model_if_missing

    h, w = img_rgb.shape[:2]
    download_model_if_missing()

    landmarker = None
    try:
        options = vision.FaceLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            num_faces=1,
        )
        landmarker = vision.FaceLandmarker.create_from_options(options)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        results = landmarker.detect(mp_image)
    finally:
        if landmarker is not None:
            try:
                landmarker.close()
            except Exception:
                pass
            landmarker = None
            gc.collect()

    if not results.face_landmarks:
        raise RuntimeError("Nessun volto rilevato sull'immagine normalizzata.")

    mesh = results.face_landmarks[0]
    pts = np.array(
        [(lm.x * w, lm.y * h) for lm in mesh],
        dtype=np.float32,
    )
    return pts


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
    Raccoglie i landmark candidati e restituisce il convex hull (N, 1, 2) int32.
    Sulla fronte: alza l'attaccatura, insetta il bordo sopraccigliare, scarta i
    punti sotto la glabella (altrimenti 234/454/8 coprono gli occhi).
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
            if i in FOREHEAD_SKIP_IDS:
                continue
            if y_limit is not None and p[1] > y_limit:
                continue
            if i in FOREHEAD_TOP_IDS:
                p[1] -= lift_px
                p[1] = min(p[1], FOREHEAD_TOP_TARGET_Y)
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
    """Mappa nome regione -> poligono (hull sulla fronte, bordo ordinato sulle guance)."""
    return {
        name: region_hull_points(landmarks_px, name, ids, image_shape=image_shape)
        for name, ids in REGION_LANDMARK_IDS.items()
    }


def fill_polygon_mask(shape_hw, points):
    mask = np.zeros(shape_hw[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [points], 255)
    return mask


def eyebrow_exclusion_mask(shape_hw, landmarks_px):
    """Maschera della peluria sopraccigliare e dell'ombra subito sotto l'arcata."""
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


def _empty_pih_metrics():
    return {
        "pih_score": 0.0,
        "peak_p98": 0.0,
        "affected_area_pct": 0.0,
        "valid_pixels": 0,
    }


def _empty_dual_metrics():
    return {
        "active_spots_count": 0,
        "active_area_pct": 0.0,
        "residual_area_pct": 0.0,
        "residual_severity_score": 0.0,
        "valid_pixels": 0,
    }


def segment_dual_bands(delta_signal, roi_mask):
    """
    Active: Δ ≥ ACTIVE_THRESHOLD
    Residual marks: TAU_NOISE ≤ Δ < ACTIVE_THRESHOLD  (stessa soglia bassa dell'overlay Magma)
    """
    valid_roi = roi_mask == 255
    active_mask = (delta_signal >= ACTIVE_THRESHOLD) & valid_roi
    residual_mask = (
        (delta_signal >= TAU_NOISE)
        & (delta_signal < ACTIVE_THRESHOLD)
        & valid_roi
    )
    return active_mask, residual_mask


def count_active_spots(active_mask, min_area=ACTIVE_MIN_AREA):
    mask_u8 = active_mask.astype(np.uint8)
    if not np.any(mask_u8):
        return 0
    num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    return sum(1 for i in range(1, num_labels) if int(stats[i, cv2.CC_STAT_AREA]) >= min_area)


def compute_dual_clinical_metrics(delta_signal, roi_mask):
    """Metriche Active / Residual Marks sul delta spettrale."""
    valid_roi = roi_mask == 255
    valid_pixels = int(np.sum(valid_roi))
    if valid_pixels == 0:
        return _empty_dual_metrics(), np.zeros_like(valid_roi), np.zeros_like(valid_roi)

    active_mask, residual_mask = segment_dual_bands(delta_signal, roi_mask)
    metrics = {
        "active_spots_count": int(count_active_spots(active_mask)),
        "active_area_pct": round(float(np.sum(active_mask) / valid_pixels * 100.0), 2),
        "residual_area_pct": round(float(np.sum(residual_mask) / valid_pixels * 100.0), 2),
        "residual_severity_score": round(
            float(np.sum(delta_signal[residual_mask]) / valid_pixels * 10000.0)
            if np.any(residual_mask)
            else 0.0,
            1,
        ),
        "valid_pixels": valid_pixels,
    }
    return metrics, active_mask, residual_mask


def compute_spectral_signal(img_rgb):
    """
    Sbilanciamento cromatico OD_G − OD_R (sostituto di L*).
    Ombra acromatica → segnale ~0; macchia/lesione → segnale positivo.
    """
    rgb_f = img_rgb.astype(np.float32) / 255.0
    R = np.clip(rgb_f[:, :, 0], OD_EPSILON, 1.0)
    G = np.clip(rgb_f[:, :, 1], OD_EPSILON, 1.0)
    OD_R = -np.log10(R)
    OD_G = -np.log10(G)
    return np.maximum(0.0, OD_G - OD_R).astype(np.float32)


def compute_normalized_background(signal, mask):
    """Fondo locale via convoluzione gaussiana normalizzata sulla maschera valida."""
    mask_f = (mask == 255).astype(np.float32)
    sigma = BG_BLUR_SIGMA
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    num = cv2.GaussianBlur(signal * mask_f, (ksize, ksize), sigma)
    den = cv2.GaussianBlur(mask_f, (ksize, ksize), sigma)
    return num / (den + 1e-5)


def compute_continuous_pih_metrics(local_contrast, roi_bool):
    """
    Metriche continue (niente blob):
      PIH_score = mean(max(0, Δ - τ_noise)) sulla ROI
      peak_p98  = 98° percentile di Δ
      affected  = % pixel con Δ > soglia percettiva
    """
    if not np.any(roi_bool):
        return _empty_pih_metrics()
    vals = local_contrast[roi_bool].astype(np.float64)
    n = float(vals.size)
    excess = np.maximum(0.0, vals - TAU_NOISE)
    return {
        "pih_score": float(np.sum(excess) / n),
        "peak_p98": float(np.percentile(vals, 98)),
        "affected_area_pct": float(np.mean(vals > AFFECTED_AREA_THRESHOLD) * 100.0),
        "valid_pixels": int(vals.size),
    }


def analyze_pih_on_mask(mask, spectral_signal, a_cie):
    """
    Delta spettrale: gating Δa* + fondo Gaussiano normalizzato su OD_G−OD_R
    + Δ = max(0, segnale − fondo). Metriche continue sul delta.
    """
    h, w = mask.shape
    zeros = np.zeros((h, w), dtype=np.float32)
    empty = _empty_pih_metrics()

    if not np.any(mask == 255):
        return {
            "metrics": empty,
            "dual_metrics": _empty_dual_metrics(),
            "local_contrast": zeros,
            "erythema_mask": np.zeros((h, w), dtype=np.uint8),
            "active_mask": np.zeros((h, w), dtype=bool),
            "residual_mask": np.zeros((h, w), dtype=bool),
        }

    roi = mask == 255
    median_a = float(np.median(a_cie[roi]))
    erythema = (a_cie - median_a) >= ERYTHEMA_A_OFFSET
    erythema_mask = (erythema & roi).astype(np.uint8) * 255
    bg_mask = mask.copy()
    bg_mask[erythema] = 0

    if not np.any(bg_mask == 255):
        return {
            "metrics": empty,
            "dual_metrics": _empty_dual_metrics(),
            "local_contrast": zeros,
            "erythema_mask": erythema_mask,
            "active_mask": np.zeros((h, w), dtype=bool),
            "residual_mask": np.zeros((h, w), dtype=bool),
        }

    signal_bg = compute_normalized_background(spectral_signal, bg_mask)
    # Macchie: spectral_signal sale rispetto al fondo (ombra acromatica si annulla).
    local_contrast = np.maximum(0.0, spectral_signal - signal_bg).astype(np.float32)

    # Metriche PIH continue sul tessuto non eritematoso; la mappa delta resta intatta.
    metrics = compute_continuous_pih_metrics(local_contrast, roi & ~erythema)
    dual, active_mask, residual_mask = compute_dual_clinical_metrics(local_contrast, mask)
    return {
        "metrics": metrics,
        "dual_metrics": dual,
        "local_contrast": local_contrast,
        "erythema_mask": erythema_mask,
        "active_mask": active_mask,
        "residual_mask": residual_mask,
    }


def _caption(img, text, origin=(24, 56)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), baseline = cv2.getTextSize(text, font, 1.35, 3)
    x, y = origin
    pad = 12
    cv2.rectangle(
        img,
        (x - pad, y - th - pad),
        (x + tw + pad, y + baseline + 6),
        (0, 0, 0),
        -1,
    )
    cv2.putText(img, text, (x, y), font, 1.35, (255, 255, 255), 3, cv2.LINE_AA)
    return img


def clean_overlay_mask(signal_bool):
    """
    Solo rendering. Niente opening (distrugge macchie a Y).
    Drop isole sotto OVERLAY_MIN_AREA, poi micro-closing.
    """
    mask = signal_bool.astype(np.uint8) * 255
    if not np.any(mask):
        return mask

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    kept = np.zeros_like(mask)
    for i in range(1, num):
        if int(stats[i, cv2.CC_STAT_AREA]) >= OVERLAY_MIN_AREA:
            kept[labels == i] = 255

    if not np.any(kept) or OVERLAY_CLOSE_KERNEL < 2:
        return kept
    close_k = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (OVERLAY_CLOSE_KERNEL, OVERLAY_CLOSE_KERNEL)
    )
    return cv2.morphologyEx(kept, cv2.MORPH_CLOSE, close_k)


def _prepare_marks_overlay_maps(local_contrast, roi_mask, erythema_mask=None):
    """
    Maschera anti-peletti + excess normalizzato (p90) + alpha edge.
    Condiviso tra Magma legacy e flat viola.
    Restituisce (clean_u8, norm_01, edge_alpha) oppure None se vuoto.
    """
    roi = roi_mask == 255
    if erythema_mask is not None:
        roi = roi & (erythema_mask == 0)
    raw = roi & (local_contrast > TAU_NOISE)
    if not np.any(raw):
        return None

    clean = clean_overlay_mask(raw)
    if not np.any(clean):
        return None

    excess = np.maximum(0.0, local_contrast.astype(np.float32) - TAU_NOISE)
    k_color = int(2 * np.ceil(3 * OVERLAY_COLOR_SIGMA) + 1)
    mask_f = (clean > 0).astype(np.float32)
    dilate_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_color, k_color))
    support = cv2.dilate(clean, dilate_k)
    support_f = (support > 0).astype(np.float32)
    num = cv2.GaussianBlur(excess * support_f, (k_color, k_color), OVERLAY_COLOR_SIGMA)
    den = cv2.GaussianBlur(support_f, (k_color, k_color), OVERLAY_COLOR_SIGMA)
    excess_smooth = np.zeros_like(excess)
    valid = den > 1e-5
    excess_smooth[valid] = num[valid] / den[valid]

    signal = clean > 0
    scale = float(np.percentile(excess_smooth[signal], HEATMAP_NORM_PERCENTILE))
    if scale <= 1e-8:
        scale = float(excess_smooth[signal].max())
    if scale <= 1e-8:
        return None
    norm = np.clip(excess_smooth / scale, 0.0, 1.0)

    k_edge = int(2 * np.ceil(3 * OVERLAY_EDGE_SIGMA) + 1)
    edge_alpha = cv2.GaussianBlur(mask_f, (k_edge, k_edge), OVERLAY_EDGE_SIGMA)
    edge_alpha = np.clip(edge_alpha, 0.0, 1.0)
    edge_alpha = np.maximum(edge_alpha, mask_f * 0.92)
    return clean, norm, edge_alpha


def blend_delta_heatmap(img_rgb, local_contrast, roi_mask, erythema_mask=None):
    """Overlay Magma legacy (TAU_NOISE + anti-peletti)."""
    out = img_rgb.copy()
    prep = _prepare_marks_overlay_maps(local_contrast, roi_mask, erythema_mask)
    if prep is None:
        return out
    _clean, norm, edge_alpha = prep

    gray_u8 = np.clip(norm * 255.0, 0, 255).astype(np.uint8)
    heat_bgr = cv2.applyColorMap(gray_u8, HEATMAP_COLORMAP)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

    base = out.astype(np.float32)
    a3 = edge_alpha[:, :, None]
    blended = base * (1.0 - a3) + heat_rgb * a3
    paint = edge_alpha > 0.02
    out = base.copy()
    out[paint] = blended[paint]
    return np.clip(out, 0, 255).astype(np.uint8)


def blend_flat_marks_overlay(img_rgb, local_contrast, roi_mask, erythema_mask=None):
    """Viola flat: alpha ∝ intensità (macchie sbiadite più trasparenti)."""
    out = img_rgb.copy()
    prep = _prepare_marks_overlay_maps(local_contrast, roi_mask, erythema_mask)
    if prep is None:
        return out
    _clean, norm, edge_alpha = prep

    # Intensità tra MIN e MAX; edge_alpha smussa i bordi.
    alpha_core = MARKS_ALPHA_MIN + norm * (MARKS_ALPHA_MAX - MARKS_ALPHA_MIN)
    alpha = np.clip(alpha_core * edge_alpha, 0.0, MARKS_ALPHA_MAX)
    marks_rgb = np.array(COLOR_MARKS_RGB, dtype=np.float32)
    base = out.astype(np.float32)
    a3 = alpha[:, :, None]
    blended = base * (1.0 - a3) + marks_rgb * a3
    paint = alpha > 0.05
    out = base.copy()
    out[paint] = blended[paint]
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_active_spot_circles(img_rgb, active_mask):
    """Cerchi rossi sui centroidi delle componenti active (area ≥ ACTIVE_MIN_AREA)."""
    out = img_rgb.copy()
    mask_u8 = active_mask.astype(np.uint8)
    if not np.any(mask_u8):
        return out
    num, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    color = COLOR_ACTIVE_RING_RGB
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < ACTIVE_MIN_AREA:
            continue
        cx = int(round(float(centroids[i, 0])))
        cy = int(round(float(centroids[i, 1])))
        radius = max(
            ACTIVE_RING_MIN_RADIUS,
            int(round(ACTIVE_RING_RADIUS_SCALE * np.sqrt(area / np.pi))) + ACTIVE_RING_PADDING_PX,
        )
        cv2.circle(out, (cx, cy), radius, color, ACTIVE_RING_THICKNESS, lineType=cv2.LINE_AA)
    return out


def _draw_hud(img, lines, origin=(24, 48), line_h=52):
    """HUD scuro semi-opaco multilinea."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness = 1.15, 3
    sizes = [cv2.getTextSize(t, font, scale, thickness) for t in lines]
    max_w = max(s[0][0] for s in sizes)
    total_h = line_h * len(lines) + 16
    x, y0 = origin
    pad = 14
    overlay = img.copy()
    cv2.rectangle(
        overlay,
        (x - pad, y0 - sizes[0][0][1] - pad),
        (x + max_w + pad, y0 - sizes[0][0][1] + total_h),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.72, img, 0.28, 0, img)
    for i, text in enumerate(lines):
        y = y0 + i * line_h
        cv2.putText(img, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return img


def compose_box3_overlay(img_rgb, local_contrast, roi_mask, active_mask, erythema_mask=None):
    """
    Marks (flat viola o Magma legacy) + active (cerchi rossi o fill cyan legacy).
    Flag in testa file per rollback immediato.
    """
    if USE_FLAT_MARKS_OVERLAY:
        out = blend_flat_marks_overlay(img_rgb, local_contrast, roi_mask, erythema_mask)
    else:
        out = blend_delta_heatmap(img_rgb, local_contrast, roi_mask, erythema_mask)

    if USE_ACTIVE_CIRCLES:
        out = draw_active_spot_circles(out, active_mask)
    elif np.any(active_mask):
        base = out.astype(np.float32)
        active_rgb = np.array(COLOR_ACTIVE_RGB, dtype=np.float32)
        a = OVERLAY_ALPHA_ACTIVE
        base[active_mask] = base[active_mask] * (1.0 - a) + active_rgb * a
        out = np.clip(base, 0, 255).astype(np.uint8)
    return out


def build_diagnostic_visualization(
    img_rgb, original_mask, local_contrast, erythema_mask, active_mask, metrics_summary
):
    """
    Tre pannelli:
      1. originale + perimetro verde ROI
      2. mappa delta continua (grigio)
      3. marks + active markers + HUD
    """
    box1 = img_rgb.copy()
    orig_contours, _ = cv2.findContours(original_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(box1, orig_contours, -1, POLYGON_COLOR, 5, lineType=cv2.LINE_AA)

    contrast_vis = np.zeros(original_mask.shape, dtype=np.float32)
    roi = original_mask == 255
    contrast_vis[roi] = local_contrast[roi]
    peak = float(contrast_vis.max()) if np.any(roi) else 0.0
    if peak > 0:
        box2_gray = np.clip(contrast_vis / peak * 255.0, 0, 255).astype(np.uint8)
    else:
        box2_gray = np.zeros_like(original_mask)
    box2 = cv2.cvtColor(box2_gray, cv2.COLOR_GRAY2RGB)

    box3 = compose_box3_overlay(
        img_rgb, local_contrast, original_mask, active_mask, erythema_mask
    )

    spots = metrics_summary.get("active_spots_count", 0)
    aarea = metrics_summary.get("active_area_pct", 0.0)
    rarea = metrics_summary.get("residual_area_pct", 0.0)
    rscore = metrics_summary.get("residual_severity_score", 0.0)

    _caption(box1, "1. ROI anatomica")
    _caption(box2, "2. Delta spettrale")
    mode = "flat+rings" if (USE_FLAT_MARKS_OVERLAY and USE_ACTIVE_CIRCLES) else "legacy"
    _draw_hud(
        box3,
        [
            f"3. Overlay ({mode})  Active Spots: {spots} | Active Area: {aarea:.2f}%",
            f"Marks Area: {rarea:.2f}% | Marks Score: {rscore:.1f}",
        ],
    )

    return np.hstack((box1, box2, box3))


def draw_region_polygons(img_rgb, polygons=None, landmarks_px=None):
    """Copia RGB con perimetri verdi (usata anche da tools/align_photo.py)."""
    annotated = img_rgb.copy()
    if polygons is None:
        if landmarks_px is None:
            landmarks_px = detect_face_landmarks_px(img_rgb)
        polygons = build_region_polygons(landmarks_px, image_shape=img_rgb.shape)

    overlay = annotated.copy()
    for points in polygons.values():
        cv2.fillPoly(overlay, [points], POLYGON_COLOR)
        cv2.polylines(annotated, [points], isClosed=True, color=POLYGON_COLOR, thickness=6, lineType=cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.18, annotated, 0.82, 0, annotated)
    return annotated


def analyze_image(img_rgb):
    """Rileva i poligoni, esclude solo le sopracciglia anatomiche, estrae la PIH spettrale."""
    print("[*] Rilevamento landmark Face Mesh sulla foto normalizzata...")
    landmarks_px = detect_face_landmarks_px(img_rgb)
    polygons = build_region_polygons(landmarks_px, image_shape=img_rgb.shape)
    brow_excl = eyebrow_exclusion_mask(img_rgb.shape, landmarks_px)

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    a_cie = lab[:, :, 1].astype(np.float32) - 128.0
    spectral_signal = compute_spectral_signal(img_rgb)

    region_results = {}
    combined_final_mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    combined_erythema = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    combined_contrast = np.zeros(img_rgb.shape[:2], dtype=np.float32)
    combined_active = np.zeros(img_rgb.shape[:2], dtype=bool)
    combined_residual = np.zeros(img_rgb.shape[:2], dtype=bool)

    for name, points in polygons.items():
        poly_mask = fill_polygon_mask(img_rgb.shape, points)
        if name == "Fronte Neutra":
            poly_mask[brow_excl > 0] = 0
        # ROI integrale: nessun filtro statistico peli/ombre.
        final_mask = poly_mask
        analysis = analyze_pih_on_mask(final_mask, spectral_signal, a_cie)
        region_results[name] = {
            "polygon": points,
            "poly_mask": poly_mask,
            "final_mask": final_mask,
            "roi_pixels": int(np.count_nonzero(final_mask == 255)),
            **analysis,
        }
        combined_final_mask = np.maximum(combined_final_mask, final_mask)
        combined_erythema = np.maximum(combined_erythema, analysis["erythema_mask"])
        sm = final_mask == 255
        combined_contrast[sm] = np.maximum(combined_contrast[sm], analysis["local_contrast"][sm])
        combined_active |= analysis["active_mask"]
        combined_residual |= analysis["residual_mask"]

    combined_residual &= ~combined_active

    # Riepilogo PIH continuo (TAU_NOISE) pesato sui pixel ROI validi (non eritema).
    tot_n = 0
    tot_excess = 0.0
    all_vals = []
    for r in region_results.values():
        roi = r["final_mask"] == 255
        ery = r["erythema_mask"] == 255
        valid = roi & ~ery
        if not np.any(valid):
            continue
        vals = r["local_contrast"][valid]
        tot_n += int(vals.size)
        tot_excess += float(np.sum(np.maximum(0.0, vals - TAU_NOISE)))
        all_vals.append(vals)
    if tot_n and all_vals:
        cat = np.concatenate(all_vals)
        summary = {
            "pih_score": tot_excess / tot_n,
            "peak_p98": float(np.percentile(cat, 98)),
            "affected_area_pct": float(np.mean(cat > AFFECTED_AREA_THRESHOLD) * 100.0),
            "valid_pixels": tot_n,
        }
    else:
        summary = _empty_pih_metrics()

    dual_summary, _, _ = compute_dual_clinical_metrics(combined_contrast, combined_final_mask)
    dual_summary["active_spots_count"] = count_active_spots(combined_active)
    summary.update(dual_summary)

    visualization = build_diagnostic_visualization(
        img_rgb,
        combined_final_mask,
        combined_contrast,
        combined_erythema,
        combined_active,
        summary,
    )
    return {
        "landmarks_px": landmarks_px,
        "polygons": polygons,
        "regions": region_results,
        "final_mask": combined_final_mask,
        "erythema_mask": combined_erythema,
        "local_contrast": combined_contrast,
        "active_mask": combined_active,
        "residual_mask": combined_residual,
        "summary": summary,
        "visualization": visualization,
    }


def _print_region_metrics(name, region):
    m = region["metrics"]
    d = region["dual_metrics"]
    print(f"[*] {name}...")
    print(f"  - Pixel ROI:              {region['roi_pixels']}")
    print(f"  - valid (no eritema):     {m['valid_pixels']}")
    print(f"  - PIH_score:              {m['pih_score']:.5f}")
    print(f"  - Peak Δ (p98):           {m['peak_p98']:.4f}")
    print(f"  - Affected area:          {m['affected_area_pct']:.2f}%")
    print(f"  - Active spots:           {d['active_spots_count']}")
    print(f"  - Active area:            {d['active_area_pct']:.2f}%")
    print(f"  - Marks area:             {d['residual_area_pct']:.2f}%")
    print(f"  - Marks severity:         {d['residual_severity_score']:.1f}")
    print("-" * 60)

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="SkinCare OS - PIH spectral playground (ROI + OD_G−OD_R anti-ombra)"
    )
    parser.add_argument("img_path", help="Percorso dell'immagine da analizzare (2048x2048 o da allineare)")
    parser.add_argument("--align", action="store_true", help="Esegue l'allineamento biometrico prima delle ROI")
    parser.add_argument("--output-aligned", help="Percorso in cui salvare l'immagine allineata pulita (se usato --align)")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Cartella di output (default: root del progetto)",
    )
    args = parser.parse_args()

    from app.config import BASE_DIR

    out_dir = args.out_dir or BASE_DIR
    os.makedirs(out_dir, exist_ok=True)

    img_path = args.img_path
    if not os.path.exists(img_path):
        print(f"[ERRORE] Il file specificato non esiste: {img_path}")
        sys.exit(1)

    aligned_path = None
    if args.align:
        print("[*] Rilevato flag --align. Avvio allineamento biometrico preliminare...")
        from tools.align_photo import align_image

        aligned_path = args.output_aligned
        if not aligned_path:
            aligned_path = os.path.join(out_dir, "debug_aligned.webp")

        res_path = align_image(img_path, aligned_path)
        if res_path:
            img_path = res_path
        else:
            print("[ERRORE] Allineamento biometrico fallito. Impossibile procedere.")
            sys.exit(1)

    print(f"[*] Caricamento immagine: {img_path}")
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("[ERRORE] Impossibile caricare o decodificare l'immagine.")
        sys.exit(1)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    if h != 2048 or w != 2048:
        print(f"[AVVISO] L'immagine caricata ha dimensioni {w}x{h}. Calibrato per 2048x2048.")
        print("[!] Tip: Usa il flag --align per allineare automaticamente l'immagine.")

    print("\n" + "=" * 60)
    print(" ANALISI PIH SPECTRAL (OD_G−OD_R + dual Active/Marks)")
    print("=" * 60)
    print("Segnale: max(0, OD_G − OD_R)  |  Δ = max(0, segnale − fondo)")
    print(f"Gating eritema Δa* >= {ERYTHEMA_A_OFFSET}")
    print(f"Fondo Gaussiano normalizzato σ={BG_BLUR_SIGMA:g}")
    print(f"τ_noise=overlay Magma={TAU_NOISE:.3f}  |  affected Δ>{AFFECTED_AREA_THRESHOLD:.3f}")
    print(f"Active only: Δ≥{ACTIVE_THRESHOLD:.3f} (cyan)  min spot {ACTIVE_MIN_AREA} px")
    print("Filtro peli: disattivato (ROI MediaPipe integrale)")
    print("-" * 60)

    try:
        result = analyze_image(img_rgb)
    except Exception as e:
        print(f"[ERRORE] Analisi PIH spectral fallita: {e}")
        sys.exit(1)

    for name in ("Fronte Neutra", "Guancia Destra", "Guancia Sinistra"):
        _print_region_metrics(name, result["regions"][name])
    s = result["summary"]
    print(" RIEPILOGO PIH SPECTRAL")
    print(f"  - PIH_score:              {s['pih_score']:.5f}")
    print(f"  - Peak Δ (p98):           {s['peak_p98']:.4f}")
    print(f"  - Affected area:          {s['affected_area_pct']:.2f}%")
    print(f"  - Active Spots:           {s['active_spots_count']}")
    print(f"  - Active Area:            {s['active_area_pct']:.2f}%")
    print(f"  - Marks Area:             {s['residual_area_pct']:.2f}%")
    print(f"  - Marks Severity Score:   {s['residual_severity_score']:.1f}")
    print("=" * 60)

    vis_bgr = cv2.cvtColor(result["visualization"], cv2.COLOR_RGB2BGR)
    target_width = 2400
    h_vis, w_vis = vis_bgr.shape[:2]
    target_height = int(target_width * h_vis / w_vis)
    vis_bgr = cv2.resize(vis_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)

    combined_path = os.path.join(out_dir, "debug_pih_spectral_result.png")
    cv2.imwrite(combined_path, vis_bgr)
    print(f"\n[+] Output salvati in: {out_dir}")
    if args.align and aligned_path:
        rois_path = os.path.splitext(aligned_path)[0] + "_rois.png"
        print(f"    - allineamento + poligoni: {rois_path}")
    print(f"    - ROI + delta + heatmap PIH spectral: {combined_path}")


if __name__ == "__main__":
    main()
