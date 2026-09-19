#!/usr/bin/env python3
"""
SkinCare OS - Analisi texture / pori / microrilievo

API libreria: analyze_texture(img_rgb, rois=None) → summary + overlay_rgb full-frame.
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
# CONFIGURAZIONE ANALISI TEXTURE
# =====================================================================
# Gating infiammatorio: OD_G−OD_R meno fondo locale (stessa logica del tool spettrale).
# Sul segnale grezzo la cute sana sta già a ~0.15–0.25 → soglia 0.030 azzererebbe la ROI.
OD_EPSILON = 1e-4
INFLAMMATION_DELTA_E = 0.030  # esclude punte di brufolo / eritema acuto sul Δ locale
INFLAMMATION_BG_SIGMA = 22.0  # fondo Gaussiano normalizzato (px)

# Bandpass DoG su log10(G) — contrasto % invariante all'illuminazione locale
G_CLEAN_KSIZE = (3, 3)
G_CLEAN_SIGMA = 0.7
G_BASE_KSIZE = (13, 13)
G_BASE_SIGMA = 2.8
MICRO_WEIGHT = 0.6
LAPLACIAN_WEIGHT = 0.4

# Metriche (scala logaritmica)
PORE_THRESHOLD = 0.018
ROUGHNESS_SCALE = 1000.0
SMOOTHNESS_ROUGHNESS_GAIN = 4.0

# Rendering
HEATMAP_COLORMAP = cv2.COLORMAP_INFERNO
HEATMAP_NORM_PERCENTILE = 95.0
HEATMAP_ALPHA = 0.55

# =====================================================================


def compute_inflammation_delta_e(img_rgb):
    """
    OD_G − OD_R grezzo (baseline cutanea tipicamente ≫ 0.03).
    Usare compute_inflammation_excess() per il gating lesioni.
    """
    rgb_f = img_rgb.astype(np.float32) / 255.0
    R = np.clip(rgb_f[:, :, 0], OD_EPSILON, 1.0)
    G = np.clip(rgb_f[:, :, 1], OD_EPSILON, 1.0)
    return np.maximum(0.0, (-np.log10(G)) - (-np.log10(R))).astype(np.float32)


def compute_normalized_background(signal, mask):
    """Fondo locale via convoluzione gaussiana normalizzata sulla maschera valida."""
    mask_f = (mask == 255).astype(np.float32)
    sigma = INFLAMMATION_BG_SIGMA
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    num = cv2.GaussianBlur(signal * mask_f, (ksize, ksize), sigma)
    den = cv2.GaussianBlur(mask_f, (ksize, ksize), sigma)
    return num / (den + 1e-5)


def compute_inflammation_excess(img_rgb, roi_mask):
    """
    Δ locale = max(0, (OD_G−OD_R) − fondo_ROI).
    Allineato al tool spettrale: soglia 0.030 ≈ lesioni acute, non baseline cute.
    """
    signal = compute_inflammation_delta_e(img_rgb)
    if not np.any(roi_mask == 255):
        return np.zeros(roi_mask.shape, dtype=np.float32)
    bg = compute_normalized_background(signal, roi_mask)
    return np.maximum(0.0, signal - bg).astype(np.float32)


def build_eval_mask(roi_mask, delta_excess, inflammation_thr=INFLAMMATION_DELTA_E):
    """ROI anatomica meno pixel con eritema/infiammazione acuta (Δ locale)."""
    return (roi_mask == 255) & (delta_excess <= inflammation_thr)


def extract_texture_energy(img_rgb, eval_mask):
    """
    Bandpass DoG + Laplaciano su log10(G) (Retinex / OD).
    Isola la variazione percentuale del microrilievo, indipendente dalla luce locale.
    """
    rgb_f = img_rgb.astype(np.float32) / 255.0
    G = rgb_f[:, :, 1]
    eval_f = eval_mask.astype(np.float32)

    G_clamped = np.clip(G, OD_EPSILON, 1.0)
    log_G = np.log10(G_clamped).astype(np.float32)

    log_fine = cv2.GaussianBlur(log_G, G_CLEAN_KSIZE, G_CLEAN_SIGMA)
    log_coarse = cv2.GaussianBlur(log_G, G_BASE_KSIZE, G_BASE_SIGMA)
    micro_relief = np.abs(log_fine - log_coarse) * eval_f

    laplacian = np.abs(cv2.Laplacian(log_fine, cv2.CV_32F, ksize=3)) * eval_f
    texture_energy = (
        (MICRO_WEIGHT * micro_relief + LAPLACIAN_WEIGHT * laplacian) * eval_f
    ).astype(np.float32)
    return texture_energy, log_fine, micro_relief, laplacian


def _empty_texture_metrics():
    return {
        "pore_prominence_pct": 0.0,
        "roughness_index": 0.0,
        "smoothness_score": 100.0,
        "valid_pixels": 0,
        "excluded_inflammation_pct": 0.0,
    }


def compute_texture_metrics(texture_energy, eval_mask, roi_mask):
    """
    pore_prominence_pct  — % eval_mask sopra PORE_THRESHOLD
    roughness_index      — energia media × 1000
    smoothness_score     — 100 − 4·roughness, clip [0, 100]
    """
    valid_pixels = int(np.sum(eval_mask))
    roi_pixels = int(np.sum(roi_mask == 255))
    if valid_pixels == 0:
        return _empty_texture_metrics()

    pore_pixels = int(np.sum((texture_energy > PORE_THRESHOLD) & eval_mask))
    roughness_index = float(np.sum(texture_energy[eval_mask]) / valid_pixels * ROUGHNESS_SCALE)
    smoothness_score = float(
        np.clip(100.0 - (roughness_index * SMOOTHNESS_ROUGHNESS_GAIN), 0.0, 100.0)
    )
    excluded_pct = (
        round(float((roi_pixels - valid_pixels) / roi_pixels * 100.0), 2)
        if roi_pixels > 0
        else 0.0
    )
    return {
        "pore_prominence_pct": round(float(pore_pixels / valid_pixels * 100.0), 2),
        "roughness_index": round(roughness_index, 2),
        "smoothness_score": round(smoothness_score, 1),
        "valid_pixels": valid_pixels,
        "excluded_inflammation_pct": excluded_pct,
    }


def analyze_texture_on_mask(img_rgb, roi_mask):
    delta_excess = compute_inflammation_excess(img_rgb, roi_mask)
    eval_mask = build_eval_mask(roi_mask, delta_excess)
    if not np.any(eval_mask):
        h, w = roi_mask.shape
        zeros = np.zeros((h, w), dtype=np.float32)
        metrics = _empty_texture_metrics()
        roi_pixels = int(np.sum(roi_mask == 255))
        if roi_pixels > 0:
            metrics["excluded_inflammation_pct"] = 100.0
        return {
            "metrics": metrics,
            "eval_mask": eval_mask,
            "texture_energy": zeros,
            "delta_excess": delta_excess,
        }

    texture_energy, _, _, _ = extract_texture_energy(img_rgb, eval_mask)
    metrics = compute_texture_metrics(texture_energy, eval_mask, roi_mask)
    return {
        "metrics": metrics,
        "eval_mask": eval_mask,
        "texture_energy": texture_energy,
        "delta_excess": delta_excess,
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


def _normalize_for_vis(energy, eval_mask):
    """Scala texture_energy in [0, 255] sui pixel validi (percentile alto)."""
    out = np.zeros(energy.shape, dtype=np.uint8)
    if not np.any(eval_mask):
        return out
    vals = energy[eval_mask]
    peak = float(np.percentile(vals, HEATMAP_NORM_PERCENTILE))
    if peak <= 1e-8:
        peak = float(vals.max()) if vals.size else 1.0
    if peak <= 1e-8:
        return out
    scaled = np.clip(energy / peak * 255.0, 0, 255).astype(np.uint8)
    out[eval_mask] = scaled[eval_mask]
    return out


def blend_texture_heatmap(img_rgb, texture_energy, eval_mask):
    """COLORMAP su texture_energy, alpha-blend solo dentro eval_mask."""
    gray = _normalize_for_vis(texture_energy, eval_mask)
    heat_bgr = cv2.applyColorMap(gray, HEATMAP_COLORMAP)
    heat_rgb = cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    base = img_rgb.astype(np.float32)
    out = base.copy()
    a = HEATMAP_ALPHA
    out[eval_mask] = base[eval_mask] * (1.0 - a) + heat_rgb[eval_mask] * a
    return np.clip(out, 0, 255).astype(np.uint8)


def green_channel_panel(img_rgb, roi_mask):
    """Pannello diagnostico: solo canale G (scala di grigi), fuori ROI nero."""
    G = img_rgb[:, :, 1]
    panel = np.zeros_like(img_rgb)
    roi = roi_mask == 255
    panel[roi, 0] = G[roi]
    panel[roi, 1] = G[roi]
    panel[roi, 2] = G[roi]
    return panel


def render_texture_overlay(img_rgb, texture_energy, eval_mask):
    """Overlay full-frame per la web: heatmap texture (niente HUD)."""
    return blend_texture_heatmap(img_rgb, texture_energy, eval_mask)


def build_diagnostic_visualization(img_rgb, roi_mask, eval_mask, texture_energy, metrics):
    """Tre pannelli diagnostici (solo CLI)."""
    box1 = img_rgb.copy()
    contours, _ = cv2.findContours(roi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(box1, contours, -1, POLYGON_COLOR, 5, lineType=cv2.LINE_AA)

    box2 = green_channel_panel(img_rgb, roi_mask)
    box3 = render_texture_overlay(img_rgb, texture_energy, eval_mask)

    sm = metrics.get("smoothness_score", 100.0)
    pore = metrics.get("pore_prominence_pct", 0.0)
    rough = metrics.get("roughness_index", 0.0)

    _caption(box1, "1. ROI anatomica")
    _caption(box2, "2. Canale Verde (G)")
    _draw_hud(
        box3,
        [
            f"3. Heatmap  Smoothness: {sm:.1f}/100",
            f"Pore Prominence: {pore:.2f}%  |  Roughness: {rough:.2f}",
        ],
    )
    return np.hstack((box1, box2, box3))


def analyze_texture(img_rgb, rois=None):
    """
    Analisi texture / microrilievo.

    rois: risultato di extract_rois (opzionale). Se None, estrae ROI internamente.
    Ritorna summary, overlay_rgb full-frame, mappe interne e regions.
    """
    if rois is None:
        rois = extract_rois(img_rgb)

    region_results = {}
    combined_roi = np.zeros(img_rgb.shape[:2], dtype=np.uint8)
    combined_eval = np.zeros(img_rgb.shape[:2], dtype=bool)
    combined_energy = np.zeros(img_rgb.shape[:2], dtype=np.float32)
    combined_excess = np.zeros(img_rgb.shape[:2], dtype=np.float32)

    for name, roi_mask in rois["masks"].items():
        analysis = analyze_texture_on_mask(img_rgb, roi_mask)
        region_results[name] = {
            "polygon": rois["polygons"][name],
            "roi_mask": roi_mask,
            "roi_pixels": int(np.count_nonzero(roi_mask == 255)),
            **analysis,
        }
        combined_roi = np.maximum(combined_roi, roi_mask)
        combined_eval |= analysis["eval_mask"]
        sm = analysis["eval_mask"]
        combined_energy[sm] = np.maximum(combined_energy[sm], analysis["texture_energy"][sm])
        combined_excess[roi_mask == 255] = np.maximum(
            combined_excess[roi_mask == 255], analysis["delta_excess"][roi_mask == 255]
        )

    summary = compute_texture_metrics(combined_energy, combined_eval, combined_roi)
    overlay_rgb = render_texture_overlay(img_rgb, combined_energy, combined_eval)
    return {
        "landmarks_px": rois["landmarks_px"],
        "polygons": rois["polygons"],
        "regions": region_results,
        "roi_mask": combined_roi,
        "eval_mask": combined_eval,
        "texture_energy": combined_energy,
        "delta_excess": combined_excess,
        "summary": summary,
        "overlay_rgb": overlay_rgb,
    }


def _print_region_metrics(name, region):
    m = region["metrics"]
    print(f"[*] {name}...")
    print(f"  - Pixel ROI:              {region['roi_pixels']}")
    print(f"  - valid (no flogosi):     {m['valid_pixels']}")
    print(f"  - excluded inflam. %:     {m['excluded_inflammation_pct']:.2f}%")
    print(f"  - Pore prominence:        {m['pore_prominence_pct']:.2f}%")
    print(f"  - Roughness index:        {m['roughness_index']:.2f}")
    print(f"  - Smoothness score:       {m['smoothness_score']:.1f}/100")
    print("-" * 60)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="SkinCare OS - Texture / pore / microrilievo (ROI guance + DoG su G)"
    )
    parser.add_argument("img_path", help="Percorso dell'immagine da analizzare (2048x2048 o da allineare)")
    parser.add_argument("--align", action="store_true", help="Esegue l'allineamento biometrico prima delle ROI")
    parser.add_argument(
        "--output-aligned",
        help="Percorso in cui salvare l'immagine allineata pulita (se usato --align)",
    )
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
        from cv_tools.align_photo import align_image

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
    print(" ANALISI TEXTURE / MICRORILIEVO (fronte + guance)")
    print("=" * 60)
    print("Segnale: DoG(log10 G) + |Laplacian(log G)|  |  gating flogosi Δlocale ≤ "
          f"{INFLAMMATION_DELTA_E:.3f}")
    print(f"Pore thr={PORE_THRESHOLD:.3f}  |  Roughness ×{ROUGHNESS_SCALE:g}")
    print(f"Smoothness = clip(100 − {SMOOTHNESS_ROUGHNESS_GAIN:g}·roughness, 0, 100)")
    print("-" * 60)

    try:
        result = analyze_texture(img_rgb)
    except Exception as e:
        print(f"[ERRORE] Analisi texture fallita: {e}")
        sys.exit(1)

    for name in ("Fronte Neutra", "Guancia Destra", "Guancia Sinistra"):
        _print_region_metrics(name, result["regions"][name])
    s = result["summary"]
    print(" RIEPILOGO TEXTURE (ROI unite)")
    print(f"  - valid (no flogosi):     {s['valid_pixels']}")
    print(f"  - excluded inflam. %:     {s['excluded_inflammation_pct']:.2f}%")
    print(f"  - Pore Prominence:        {s['pore_prominence_pct']:.2f}%")
    print(f"  - Roughness Index:        {s['roughness_index']:.2f}")
    print(f"  - Smoothness Score:       {s['smoothness_score']:.1f}/100")
    print("=" * 60)

    visualization = build_diagnostic_visualization(
        img_rgb,
        result["roi_mask"],
        result["eval_mask"],
        result["texture_energy"],
        result["summary"],
    )
    vis_bgr = cv2.cvtColor(visualization, cv2.COLOR_RGB2BGR)
    target_width = 2400
    h_vis, w_vis = vis_bgr.shape[:2]
    target_height = int(target_width * h_vis / w_vis)
    vis_bgr = cv2.resize(vis_bgr, (target_width, target_height), interpolation=cv2.INTER_AREA)

    combined_path = os.path.join(out_dir, "debug_texture_result.png")
    cv2.imwrite(combined_path, vis_bgr)
    print(f"\n[+] Output salvati in: {out_dir}")
    if args.align and aligned_path:
        print(f"    - allineamento: {aligned_path}")
    print(f"    - ROI + microrilievo + heatmap: {combined_path}")


if __name__ == "__main__":
    main()
