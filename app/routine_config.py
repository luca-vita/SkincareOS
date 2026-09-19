"""
Carica routine/prodotti/formule da file JSON esterno.

Priorità path:
  1. env ROUTINE_CONFIG_PATH
  2. <repo>/data/routine.json   (personale, sotto data/ gitignored)
  3. <repo>/routine.json        (legacy, root)
  4. <repo>/config.example.json (template condiviso)

I colori UI non stanno qui: restano nel design system del codice.
"""

from __future__ import annotations

import json
import os
import threading
from copy import deepcopy
from typing import Any

from app.config import BASE_DIR, DATA_DIR

_lock = threading.Lock()
_cache: dict[str, Any] | None = None
_cache_path: str | None = None


def _candidate_paths() -> list[str]:
    paths: list[str] = []
    env = os.environ.get("ROUTINE_CONFIG_PATH")
    if env:
        paths.append(env)
    paths.append(os.path.join(DATA_DIR, "routine.json"))
    paths.append(os.path.join(BASE_DIR, "routine.json"))  # legacy
    paths.append(os.path.join(BASE_DIR, "config.example.json"))
    return paths


def resolve_config_path() -> str:
    for path in _candidate_paths():
        if path and os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Nessun file di configurazione routine trovato. "
        "Copia config.example.json → data/routine.json e personalizza."
    )


def _validate(cfg: dict[str, Any]) -> None:
    for key in ("routine", "actives", "saturation_bands", "lags", "windows"):
        if key not in cfg:
            raise ValueError(f"Config routine: manca la sezione '{key}'")
    if not cfg["actives"]:
        raise ValueError("Config routine: 'actives' non può essere vuoto")
    bands = cfg["saturation_bands"]
    if "colors" in bands:
        raise ValueError(
            "Config routine: 'saturation_bands.colors' non è ammesso "
            "(i colori restano nel codice UI)"
        )
    if float(bands["optimal"]) <= float(bands["suboptimal"]):
        raise ValueError("Config routine: optimal deve essere > suboptimal")
    for active_id, active in cfg["actives"].items():
        for req in ("alpha", "target", "dose_col", "label", "log_flags"):
            if req not in active:
                raise ValueError(f"Config active '{active_id}': manca '{req}'")
        # initial_p: default 0 se assente (niente “già a regime” implicito)
        if "initial_p" not in active:
            active["initial_p"] = 0.0
        else:
            active["initial_p"] = float(active["initial_p"])
        if not isinstance(active["log_flags"], list) or not active["log_flags"]:
            raise ValueError(
                f"Config active '{active_id}': log_flags deve essere una lista non vuota"
            )


def load_routine_config(*, force_reload: bool = False) -> dict[str, Any]:
    global _cache, _cache_path
    path = resolve_config_path()
    with _lock:
        if _cache is not None and _cache_path == path and not force_reload:
            return deepcopy(_cache)
        with open(path, "r", encoding="utf-8") as fh:
            cfg = json.load(fh)
        # Compat: vecchie chiavi agents → actives
        if "actives" not in cfg and "agents" in cfg:
            cfg["actives"] = cfg.pop("agents")
        _validate(cfg)
        _cache = cfg
        _cache_path = path
        return deepcopy(cfg)


def reload_routine_config() -> dict[str, Any]:
    return load_routine_config(force_reload=True)


def get_actives() -> dict[str, Any]:
    return load_routine_config()["actives"]


# Alias legacy durante la migrazione
def get_agents() -> dict[str, Any]:
    return get_actives()


def get_routine_steps() -> dict[str, list[dict[str, Any]]]:
    return load_routine_config()["routine"]


def all_slot_ids() -> list[str]:
    steps = get_routine_steps()
    ids: list[str] = []
    for period in ("am", "pm"):
        for step in steps.get(period, []):
            sid = step["id"]
            if sid not in ids:
                ids.append(sid)
    return ids


def get_saturation_bands() -> dict[str, Any]:
    return load_routine_config()["saturation_bands"]


def get_lags() -> dict[str, Any]:
    return load_routine_config()["lags"]


def get_windows() -> dict[str, Any]:
    return load_routine_config()["windows"]


def get_compliance_axes() -> list[dict[str, Any]]:
    """
    Assi del radar di aderenza (therapeutic pressure).

    Derivati dagli actives: per ciascuno, slot = log_flags × finestra terapeutica.
    Non è una sezione separata in JSON — evita la duplicazione che c'era con axes.
    """
    windows = get_windows()
    window_days = int(windows.get("therapeutic", 14))
    axes: list[dict[str, Any]] = []
    for active_id, active in get_actives().items():
        flags = list(active["log_flags"])
        total_slots = window_days * len(flags)
        axes.append(
            {
                "id": active_id,
                "label": active["label"],
                "slot_flags": flags,
                "total_slots": total_slots,
                "target_pct": float(active.get("compliance_target_pct", 80.0)),
            }
        )
    return axes


# Nome precedente usato da metrics.py
def get_therapeutic_axes() -> list[dict[str, Any]]:
    return get_compliance_axes()


def public_config_payload() -> dict[str, Any]:
    """Sottoinsieme da esporre al frontend (niente colori, niente path assoluti)."""
    cfg = load_routine_config()
    return {
        "source_path": os.path.basename(resolve_config_path()),
        "routine": cfg["routine"],
        "actives": {
            aid: {"label": a["label"]}
            for aid, a in cfg["actives"].items()
        },
        "saturation_bands": {
            "optimal": cfg["saturation_bands"]["optimal"],
            "suboptimal": cfg["saturation_bands"]["suboptimal"],
            "labels": cfg["saturation_bands"].get("labels", {}),
        },
        "windows": cfg["windows"],
        "compliance_axes": get_compliance_axes(),
    }
