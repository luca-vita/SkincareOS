#!/usr/bin/env python3
"""
SkinCare OS - Analisi PIH spectral (OD_G − OD_R)

API libreria: analyze_pih(img_rgb, rois=None) → summary + overlay_rgb full-frame.
CLI: pannello diagnostico 3 box (debug).
"""

import os
import sys
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cv_tools.extract_rois import (  # noqa: E402
    POLYGON_COLOR,
    extract_rois,
)

# =====================================================================
# CALIBRAZIONE ANALISI
# =====================================================================
BG_BLUR_SIGMA = 22.0
OD_EPSILON = 1e-4
# Esclude brufoli/eritema dalla stima del fondo (Δa* vs mediana ROI).
ERYTHEMA_A_OFFSET = 100.0

TAU_NOISE = 0.004            # Δ ≥ TAU → affected (marks + active)
ACTIVE_THRESHOLD = 0.040     # brufoli attivi / infiammazione acuta
ACTIVE_MIN_AREA = 15         # CC min area (px) per contare uno spot
DISCROMIA_SCORE_SCALE = 10000.0

# Specular: su highlight bianchi OD_G≈OD_R → Δ≈0 (valle artefattuale).
# I pixel non sono misura di cromoforo (riflettanza speculare ≠ assorbanza).
SPECULAR_L_OFFSET = 18.0     # L* OpenCV: sopra la media locale
SPECULAR_CHROMA_MAX = 22.0   # basso croma = highlight acromatico
SPECULAR_INPAINT_RADIUS = 9  # raggio (px) max-filter dai vicini diffusi

# Ricomposizione frammenti: unisci due CC active solo se il gap è artefatto
# (specular e/o valle di Δ), non se sono due lesioni separate su cute sana.
SPLIT_MAX_GAP_PX = 30        # gap max tra bordi equivalenti (px)
SPLIT_SPECULAR_FRAC = 0.30   # frazione specular sul segmento tra i due pezzi
SPLIT_VALLEY_RATIO = 0.60    # mean(Δ_gap) < ratio × min(mean Δ dei due pezzi)

# Overlay marks (viola flat, alpha ∝ intensità)
HEATMAP_NORM_PERCENTILE = 90.0
OVERLAY_CLOSE_KERNEL = 3
OVERLAY_MIN_AREA = 200
OVERLAY_COLOR_SIGMA = 4.0
OVERLAY_EDGE_SIGMA = 2.5
COLOR_MARKS_RGB = (168, 70, 210)
MARKS_ALPHA_MIN = 0.15
MARKS_ALPHA_MAX = 1.0

# Cerchi active
COLOR_ACTIVE_RING_RGB = (230, 35, 35)
ACTIVE_RING_THICKNESS = 3
ACTIVE_RING_MIN_RADIUS = 14
ACTIVE_RING_RADIUS_SCALE = 1.35
ACTIVE_RING_PADDING_PX = 8

# =====================================================================


def _empty_clinical_metrics():
    return {
        "active_spots_count": 0,
        "active_area_pct": 0.0,
        "affected_area_pct": 0.0,
        "residual_area_pct": 0.0,
        "discromia_score": 0.0,
        "peak_p98": 0.0,
        "valid_pixels": 0,
    }


def segment_clinical_bands(delta_signal, roi_mask):
    """
    Affected:  Δ ≥ TAU_NOISE
    Residual:  TAU_NOISE ≤ Δ < ACTIVE_THRESHOLD
    Active:    Δ ≥ ACTIVE_THRESHOLD
    """
    valid_roi = roi_mask == 255
    active_mask = (delta_signal >= ACTIVE_THRESHOLD) & valid_roi
    residual_mask = (
        (delta_signal >= TAU_NOISE)
        & (delta_signal < ACTIVE_THRESHOLD)
        & valid_roi
    )
    affected_mask = active_mask | residual_mask
    return active_mask, residual_mask, affected_mask


def count_active_spots(active_mask, min_area=ACTIVE_MIN_AREA):
    mask_u8 = active_mask.astype(np.uint8)
    if not np.any(mask_u8):
        return 0
    num_labels, _labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    return sum(1 for i in range(1, num_labels) if int(stats[i, cv2.CC_STAT_AREA]) >= min_area)


def detect_specular_mask(img_rgb, roi_mask=None):
    """
    Highlight specular locali: L* ≫ media locale e croma basso.
    Relativo alla cute circostante (non una soglia assoluta di luminosità),
    così pelle chiara uniforme non viene marcata intera.
    """
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    L = lab[:, :, 0]
    a = lab[:, :, 1] - 128.0
    b = lab[:, :, 2] - 128.0
    chroma = np.sqrt(a * a + b * b)
    k = int(2 * np.ceil(3 * 8.0) + 1)
    L_local = cv2.GaussianBlur(L, (k, k), 8.0)
    specular = (L - L_local >= SPECULAR_L_OFFSET) & (chroma <= SPECULAR_CHROMA_MAX)
    if roi_mask is not None:
        specular &= roi_mask == 255
    return specular


def inpaint_delta_on_specular(delta_signal, specular_mask, radius=None):
    """
    Sui pixel specular Δ misurato non è cromoforo: propaga il max locale
    dai soli pixel a riflettanza diffusa (ricostruisce la mappa latente).
    """
    if radius is None:
        radius = SPECULAR_INPAINT_RADIUS
    if radius < 1 or not np.any(specular_mask):
        return delta_signal
    valid = ~specular_mask
    k = int(2 * radius + 1) | 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # Max solo da pixel validi: i specular partono da -1 così non propagano sé stessi.
    seed = np.where(valid, delta_signal, -1.0).astype(np.float32)
    local_max = cv2.dilate(seed, se)
    out = delta_signal.copy()
    replace = specular_mask & (local_max >= 0.0)
    out[replace] = local_max[replace]
    return out


def _sample_segment(y0, x0, y1, x1, n_samples=24):
    """Pixel lungo il segmento tra due punti (y,x)."""
    ts = np.linspace(0.0, 1.0, n_samples)
    ys = np.round(y0 + ts * (y1 - y0)).astype(np.int32)
    xs = np.round(x0 + ts * (x1 - x0)).astype(np.int32)
    return ys, xs


def reconnect_specular_splits(active_mask, specular_mask, delta_signal):
    """
    Unisce CC active separate da un gap artefattuale (riflesso / valle di Δ).

    Criterio (entrambi i pezzi ≥ ACTIVE_MIN_AREA, gap tra bordi ≤ SPLIT_MAX_GAP_PX):
      - frazione specular sul segmento ≥ SPLIT_SPECULAR_FRAC, oppure
      - mean(Δ) sul gap < SPLIT_VALLEY_RATIO × min(mean Δ dei due pezzi).

    Due brufoli distinti su cute sana non soddisfano il criterio e restano separati.
    """
    if not np.any(active_mask):
        return active_mask
    u8 = (active_mask.astype(np.uint8)) * 255
    n, labels, stats, cents = cv2.connectedComponentsWithStats(u8, connectivity=8)
    if n <= 2:
        return active_mask

    h, w = u8.shape
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    mean_delta = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        m = labels == i
        mean_delta[i] = float(delta_signal[m].mean()) if np.any(m) else 0.0

    kept = [
        i for i in range(1, n)
        if int(stats[i, cv2.CC_STAT_AREA]) >= ACTIVE_MIN_AREA
    ]
    for a_idx, i in enumerate(kept):
        ri = float(np.sqrt(stats[i, cv2.CC_STAT_AREA] / np.pi))
        cxi, cyi = float(cents[i, 0]), float(cents[i, 1])
        for j in kept[a_idx + 1:]:
            rj = float(np.sqrt(stats[j, cv2.CC_STAT_AREA] / np.pi))
            cxj, cyj = float(cents[j, 0]), float(cents[j, 1])
            dist = float(np.hypot(cxi - cxj, cyi - cyj))
            gap = dist - ri - rj
            if gap > SPLIT_MAX_GAP_PX:
                continue

            ys, xs = _sample_segment(cyi, cxi, cyj, cxj)
            inside = (ys >= 0) & (ys < h) & (xs >= 0) & (xs < w)
            ys, xs = ys[inside], xs[inside]
            if ys.size == 0:
                continue
            # Solo il tratto tra i due pezzi (né i né j).
            lab = labels[ys, xs]
            gap_pix = (lab != i) & (lab != j)
            if not np.any(gap_pix):
                # Si toccano già o si sovrappongono: unisci.
                union(i, j)
                continue
            g_ys, g_xs = ys[gap_pix], xs[gap_pix]
            spec_frac = float(np.mean(specular_mask[g_ys, g_xs]))
            gap_d = float(np.mean(delta_signal[g_ys, g_xs]))
            floor = SPLIT_VALLEY_RATIO * min(mean_delta[i], mean_delta[j])
            if spec_frac >= SPLIT_SPECULAR_FRAC or gap_d < floor:
                union(i, j)

    # Maschera ricomposta: hull per gruppo unito (riempie il bianco centrale).
    out = np.zeros_like(active_mask)
    roots = {}
    for i in range(1, n):
        roots.setdefault(find(i), []).append(i)

    for members in roots.values():
        member_mask = np.isin(labels, members)
        if len(members) == 1:
            out[member_mask] = True
            continue
        ys, xs = np.where(member_mask)
        if xs.size < 3:
            out[member_mask] = True
            continue
        pts = np.column_stack([xs, ys]).astype(np.int32).reshape(-1, 1, 2)
        hull = cv2.convexHull(pts)
        layer = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(layer, hull, 255)
        out[layer > 0] = True

    return out


def compute_clinical_metrics(delta_signal, roi_mask, specular_mask=None):
    """
    active_area_pct   = % di pixel active sul solo affected
    affected_area_pct = % ROI con Δ ≥ TAU
    residual_area_pct = % ROI con TAU ≤ Δ < ACTIVE
    discromia_score   = sum(Δ[affected]) / N_ROI × scale
    """
    valid_roi = roi_mask == 255
    valid_pixels = int(np.sum(valid_roi))
    if valid_pixels == 0:
        false = np.zeros_like(valid_roi)
        return _empty_clinical_metrics(), false, false, false

    active_mask, residual_mask, affected_mask = segment_clinical_bands(delta_signal, roi_mask)
    if specular_mask is not None:
        active_mask = reconnect_specular_splits(
            active_mask, specular_mask & valid_roi, delta_signal
        )
        # Residual non deve restare sotto i pezzi ricomposti.
        residual_mask = residual_mask & ~active_mask
        affected_mask = active_mask | residual_mask
    vals = delta_signal[valid_roi]
    affected_pixels = int(np.sum(affected_mask))
    active_area_pct = (
        round(float(np.sum(active_mask) / affected_pixels * 100.0), 2)
        if affected_pixels > 0
        else 0.0
    )
    metrics = {
        "active_spots_count": int(count_active_spots(active_mask)),
        "active_area_pct": active_area_pct,
        "affected_area_pct": round(float(affected_pixels / valid_pixels * 100.0), 2),
        "residual_area_pct": round(float(np.sum(residual_mask) / valid_pixels * 100.0), 2),
        "discromia_score": round(
            float(np.sum(delta_signal[affected_mask]) / valid_pixels * DISCROMIA_SCORE_SCALE)
            if affected_pixels > 0
            else 0.0,
            1,
        ),
        "peak_p98": float(np.percentile(vals, 98)) if vals.size else 0.0,
        "valid_pixels": valid_pixels,
    }
    return metrics, active_mask, residual_mask, affected_mask


def compute_spectral_signal(img_rgb):
    """OD_G − OD_R (ombra acromatica → ~0)."""
    rgb_f = img_rgb.astype(np.float32) / 255.0
    R = np.clip(rgb_f[:, :, 0], OD_EPSILON, 1.0)
    G = np.clip(rgb_f[:, :, 1], OD_EPSILON, 1.0)
    return np.maximum(0.0, (-np.log10(G)) - (-np.log10(R))).astype(np.float32)


def compute_normalized_background(signal, mask):
    mask_f = (mask == 255).astype(np.float32)
    sigma = BG_BLUR_SIGMA
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    num = cv2.GaussianBlur(signal * mask_f, (ksize, ksize), sigma)
    den = cv2.GaussianBlur(mask_f, (ksize, ksize), sigma)
    return num / (den + 1e-5)


def analyze_pih_on_mask(mask, spectral_signal, a_cie=None, specular_mask=None):
    """
    Δ = max(0, segnale − fondo) + metriche cliniche.

    Se a_cie è fornito, i pixel eritematosi (Δa* ≥ ERYTHEMA_A_OFFSET rispetto
    alla mediana ROI) sono esclusi dalla stima del fondo Gaussiano, così i
    brufoli attivi non alzano il background e non comprimono le macchie PIH.

    Se specular_mask è fornito, quegli pixel sono esclusi dal fondo e il Δ
    viene inpaintato dai vicini diffusi prima della segmentazione.
    """
    h, w = mask.shape
    zeros = np.zeros((h, w), dtype=np.float32)
    empty = _empty_clinical_metrics()
    false = np.zeros((h, w), dtype=bool)

    if not np.any(mask == 255):
        return {
            "metrics": empty,
            "local_contrast": zeros,
            "active_mask": false,
            "residual_mask": false,
            "erythema_mask": np.zeros((h, w), dtype=np.uint8),
        }

    roi = mask == 255
    erythema_mask = np.zeros((h, w), dtype=np.uint8)
    bg_mask = mask.copy()

    if a_cie is not None:
        median_a = float(np.median(a_cie[roi]))
        erythema = (a_cie - median_a) >= ERYTHEMA_A_OFFSET
        erythema_mask = (erythema & roi).astype(np.uint8) * 255
        bg_mask[erythema] = 0

    if specular_mask is not None:
        bg_mask[specular_mask & roi] = 0

    if not np.any(bg_mask == 255):
        return {
            "metrics": empty,
            "local_contrast": zeros,
            "active_mask": false,
            "residual_mask": false,
            "erythema_mask": erythema_mask,
        }

    signal_bg = compute_normalized_background(spectral_signal, bg_mask)
    local_contrast = np.maximum(0.0, spectral_signal - signal_bg).astype(np.float32)
    if specular_mask is not None:
        local_contrast = inpaint_delta_on_specular(
            local_contrast, specular_mask & roi
        )
    metrics, active_mask, residual_mask, _affected = compute_clinical_metrics(
        local_contrast, mask, specular_mask=specular_mask
    )
    return {
        "metrics": metrics,
        "local_contrast": local_contrast,
        "active_mask": active_mask,
        "residual_mask": residual_mask,
        "erythema_mask": erythema_mask,
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


def _draw_hud(img, lines, origin=(24, 48), line_h=52):
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


def clean_overlay_mask(signal_bool):
    """Drop isole < OVERLAY_MIN_AREA, poi micro-closing (no opening)."""
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


def _prepare_marks_overlay_maps(local_contrast, roi_mask):
    """Anti-peletti + excess normalizzato (p90) + alpha edge. None se vuoto."""
    roi = roi_mask == 255
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


def blend_flat_marks_overlay(img_rgb, local_contrast, roi_mask):
    """Viola flat: alpha ∝ intensità."""
    out = img_rgb.copy()
    prep = _prepare_marks_overlay_maps(local_contrast, roi_mask)
    if prep is None:
        return out
    _clean, norm, edge_alpha = prep

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
    """Cerchi rossi sui centroidi active (area ≥ ACTIVE_MIN_AREA)."""
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


def render_pih_overlay(img_rgb, local_contrast, roi_mask, active_mask):
    """Overlay full-frame per la web: viola flat + cerchi active (niente HUD)."""
    out = blend_flat_marks_overlay(img_rgb, local_contrast, roi_mask)
    return draw_active_spot_circles(out, active_mask)


def build_diagnostic_visualization(img_rgb, roi_mask, local_contrast, active_mask, metrics_summary):
    """ROI | delta grigio | overlay + HUD (solo CLI)."""
    box1 = img_rgb.copy()
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(box1, contours, -1, POLYGON_COLOR, 5, lineType=cv2.LINE_AA)

    contrast_vis = np.zeros(roi_mask.shape, dtype=np.float32)
    roi = roi_mask == 255
    contrast_vis[roi] = local_contrast[roi]
    peak = float(contrast_vis.max()) if np.any(roi) else 0.0
    if peak > 0:
        box2_gray = np.clip(contrast_vis / peak * 255.0, 0, 255).astype(np.uint8)
    else:
        box2_gray = np.zeros_like(roi_mask)
    box2 = cv2.cvtColor(box2_gray, cv2.COLOR_GRAY2RGB)

    box3 = render_pih_overlay(img_rgb, local_contrast, roi_mask, active_mask)

    spots = metrics_summary.get("active_spots_count", 0)
    aarea = metrics_summary.get("active_area_pct", 0.0)
    aff = metrics_summary.get("affected_area_pct", 0.0)
    rarea = metrics_summary.get("residual_area_pct", 0.0)
    dscore = metrics_summary.get("discromia_score", 0.0)

    _caption(box1, "1. ROI anatomica")
    _caption(box2, "2. Delta spettrale")
    _draw_hud(
        box3,
        [
            f"3. Overlay  Active Spots: {spots} | Active/Aff: {aarea:.2f}%",
            f"Affected/ROI: {aff:.2f}% | Residual/ROI: {rarea:.2f}% | Discromia: {dscore:.1f}",
        ],
    )
    return np.hstack((box1, box2, box3))


def analyze_pih(img_rgb, rois=None):
    """
    Analisi PIH spettrale.

    rois: risultato di extract_rois (opzionale). Se None, estrae ROI internamente.
    Ritorna summary, overlay_rgb full-frame, mappe interne e regions.
    """
    if rois is None:
        rois = extract_rois(img_rgb)

    spectral_signal = compute_spectral_signal(img_rgb)
    # Canale a* CIELAB per gating eritema (stessa logica del playground debug).
    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)
    a_cie = lab[:, :, 1].astype(np.float32)
    specular = detect_specular_mask(img_rgb)

    region_results = {}
    combined_mask = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    combined_contrast = np.zeros(img_rgb.shape[:2], dtype=np.float32)
    combined_active = np.zeros(img_rgb.shape[:2], dtype=bool)

    for name, final_mask in rois["masks"].items():
        analysis = analyze_pih_on_mask(
            final_mask, spectral_signal, a_cie=a_cie, specular_mask=specular
        )
        region_results[name] = {
            "polygon": rois["polygons"][name],
            "final_mask": final_mask,
            "roi_pixels": int(np.count_nonzero(final_mask == 255)),
            **analysis,
        }
        combined_mask = np.maximum(combined_mask, final_mask)
        sm = final_mask == 255
        combined_contrast[sm] = np.maximum(combined_contrast[sm], analysis["local_contrast"][sm])
        combined_active |= analysis["active_mask"]

    # Frame combinato: stessa ricomposizione specular-aware sul Δ già inpaintato.
    summary, combined_active, _, _ = compute_clinical_metrics(
        combined_contrast, combined_mask, specular_mask=specular
    )

    overlay_rgb = render_pih_overlay(
        img_rgb, combined_contrast, combined_mask, combined_active
    )
    return {
        "landmarks_px": rois["landmarks_px"],
        "polygons": rois["polygons"],
        "regions": region_results,
        "final_mask": combined_mask,
        "local_contrast": combined_contrast,
        "active_mask": combined_active,
        "summary": summary,
        "overlay_rgb": overlay_rgb,
    }


def _print_region_metrics(name, region):
    m = region["metrics"]
    print(f"[*] {name}...")
    print(f"  - Pixel ROI:              {region['roi_pixels']}")
    print(f"  - valid pixels:           {m['valid_pixels']}")
    print(f"  - Active spots:           {m['active_spots_count']}")
    print(f"  - Active area (/aff.):    {m['active_area_pct']:.2f}%")
    print(f"  - Affected area (/ROI):   {m['affected_area_pct']:.2f}%")
    print(f"  - Residual area (/ROI):   {m['residual_area_pct']:.2f}%")
    print(f"  - Discromia score:        {m['discromia_score']:.1f}")
    print(f"  - Peak Δ (p98):           {m['peak_p98']:.4f}")
    print("-" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="SkinCare OS - PIH spectral (ROI + OD_G−OD_R)"
    )
    parser.add_argument("img_path", help="Immagine da analizzare (2048x2048 o da allineare)")
    parser.add_argument("--align", action="store_true", help="Allineamento biometrico preliminare")
    parser.add_argument("--output-aligned", help="Path immagine allineata (con --align)")
    parser.add_argument("--out-dir", default=None, help="Cartella output (default: root progetto)")
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
        print("[*] Flag --align: allineamento biometrico...")
        from cv_tools.align_photo import align_image

        aligned_path = args.output_aligned or os.path.join(out_dir, "debug_aligned.webp")
        res_path = align_image(img_path, aligned_path)
        if not res_path:
            print("[ERRORE] Allineamento biometrico fallito.")
            sys.exit(1)
        img_path = res_path

    print(f"[*] Caricamento immagine: {img_path}")
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        print("[ERRORE] Impossibile caricare o decodificare l'immagine.")
        sys.exit(1)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    if h != 2048 or w != 2048:
        print(f"[AVVISO] Dimensioni {w}x{h} (calibrato per 2048x2048). Usa --align se serve.")

    print("\n" + "=" * 60)
    print(" ANALISI PIH SPECTRAL (OD_G−OD_R + Active/Marks)")
    print("=" * 60)
    print("Segnale: max(0, OD_G − OD_R)  |  Δ = max(0, segnale − fondo)")
    print(f"Fondo Gaussiano σ={BG_BLUR_SIGMA:g}")
    print(f"Affected: Δ≥{TAU_NOISE:.3f}  |  Residual: τ≤Δ<active  |  Active: Δ≥{ACTIVE_THRESHOLD:.3f}")
    print(f"Discromia = sum(Δ[affected])/N_ROI × {DISCROMIA_SCORE_SCALE:g}")
    print(f"Active min spot {ACTIVE_MIN_AREA} px  |  Overlay: flat viola + cerchi rossi")
    print("-" * 60)

    try:
        result = analyze_pih(img_rgb)
    except Exception as e:
        print(f"[ERRORE] Analisi PIH spectral fallita: {e}")
        sys.exit(1)

    for name in ("Fronte Neutra", "Guancia Destra", "Guancia Sinistra"):
        _print_region_metrics(name, result["regions"][name])
    s = result["summary"]
    print(" RIEPILOGO PIH SPECTRAL")
    print(f"  - Active Spots:           {s['active_spots_count']}")
    print(f"  - Active Area (/aff.):    {s['active_area_pct']:.2f}%")
    print(f"  - Affected Area (/ROI):   {s['affected_area_pct']:.2f}%")
    print(f"  - Residual Area (/ROI):   {s['residual_area_pct']:.2f}%")
    print(f"  - Discromia Score:        {s['discromia_score']:.1f}")
    print(f"  - Peak Δ (p98):           {s['peak_p98']:.4f}")
    print("=" * 60)

    visualization = build_diagnostic_visualization(
        img_rgb,
        result["final_mask"],
        result["local_contrast"],
        result["active_mask"],
        result["summary"],
    )
    vis_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    target_width = 2400
    h_vis, w_vis = vis_bgr.shape[:2]
    vis_bgr = cv2.resize(
        vis_bgr,
        (target_width, int(target_width * h_vis / w_vis)),
        interpolation=cv2.INTER_AREA,
    )
    combined_path = os.path.join(out_dir, "debug_pih_spectral_result.png")
    cv2.imwrite(combined_path, vis_bgr)
    print(f"\n[+] Output: {combined_path}")
    if args.align and aligned_path:
        print(f"    - allineamento: {aligned_path}")


if __name__ == "__main__":
    main()
