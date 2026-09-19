"""
Orchestrazione analisi foto post-allineamento: ROI una volta → PIH + texture,
salvataggio overlay WebP e metriche globali.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any

import numpy as np
from PIL import Image

from app.config import PHOTOS_DIR
from cv_tools.analyze_pih_spectral import analyze_pih
from cv_tools.analyze_texture import analyze_texture
from cv_tools.extract_rois import extract_rois

ANALYSIS_VERSION = "pih1-tex1"


def overlay_paths_for_date(entry_date_str: str) -> dict[str, str]:
    """Path assoluti degli overlay pre-renderizzati."""
    return {
        "pih": os.path.join(PHOTOS_DIR, f"{entry_date_str}_pih.webp"),
        "texture": os.path.join(PHOTOS_DIR, f"{entry_date_str}_tex.webp"),
    }


def overlay_urls_for_date(entry_date_str: str) -> dict[str, str]:
    return {
        "pih_url": f"/static/photos/{entry_date_str}_pih.webp",
        "texture_url": f"/static/photos/{entry_date_str}_tex.webp",
    }


def _save_webp_atomic(img_rgb: np.ndarray, dest_path: str) -> str:
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    directory = os.path.dirname(dest_path) or PHOTOS_DIR
    with tempfile.NamedTemporaryFile(suffix=".webp", dir=directory, delete=False) as tmp:
        tmp_path = tmp.name
    try:
        Image.fromarray(img_rgb).save(tmp_path, format="WEBP", quality=90, lossless=False)
        os.replace(tmp_path, dest_path)
        return dest_path
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def delete_analysis_overlays(entry_date_str: str) -> None:
    """Rimuove i WebP overlay se presenti (best-effort)."""
    for path in overlay_paths_for_date(entry_date_str).values():
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as e:
            print(f"[WARNING] Impossibile rimuovere {path}: {e}")


def build_analysis_summary(pih_summary: dict, texture_summary: dict) -> dict[str, Any]:
    """Metriche globali persistite in analysis_json."""
    return {
        "analysis_version": ANALYSIS_VERSION,
        "active_spots_count": int(pih_summary.get("active_spots_count", 0)),
        "active_area_pct": float(pih_summary.get("active_area_pct", 0.0)),
        "affected_area_pct": float(pih_summary.get("affected_area_pct", 0.0)),
        "residual_area_pct": float(pih_summary.get("residual_area_pct", 0.0)),
        "discromia_score": float(pih_summary.get("discromia_score", 0.0)),
        "peak_p98": float(pih_summary.get("peak_p98", 0.0)),
        "pore_prominence_pct": float(texture_summary.get("pore_prominence_pct", 0.0)),
        "roughness_index": float(texture_summary.get("roughness_index", 0.0)),
        "smoothness_score": float(texture_summary.get("smoothness_score", 100.0)),
    }


def run_photo_analysis(warped_rgb: np.ndarray, entry_date_str: str) -> dict[str, Any]:
    """
    Estrae ROI una volta, analizza PIH + texture, salva overlay, ritorna metriche + path.
    """
    rois = extract_rois(warped_rgb)
    pih = analyze_pih(warped_rgb, rois=rois)
    tex = analyze_texture(warped_rgb, rois=rois)

    paths = overlay_paths_for_date(entry_date_str)
    pih_path = _save_webp_atomic(pih["overlay_rgb"], paths["pih"])
    tex_path = _save_webp_atomic(tex["overlay_rgb"], paths["texture"])

    analysis = build_analysis_summary(pih["summary"], tex["summary"])
    urls = overlay_urls_for_date(entry_date_str)
    return {
        "pih_path": pih_path,
        "texture_path": tex_path,
        "analysis": analysis,
        "analysis_version": ANALYSIS_VERSION,
        "analysis_json": json.dumps(analysis),
        **urls,
    }
