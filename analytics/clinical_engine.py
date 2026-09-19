"""
Motore analitico clinico SkinCare OS.

EMA farmacocinetica, bande di saturazione, delta lag-dependent su checkpoint
di rasatura, dashboard multi-pannello e matrice di correlazione Spearman.
"""

from __future__ import annotations

from io import BytesIO
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from scipy import stats

from app.routine_config import (
    get_actives,
    get_lags,
    get_saturation_bands,
    get_windows,
)

# ---------------------------------------------------------------------------
# Palette UI (design system — non configurabile in routine.json)
# ---------------------------------------------------------------------------

UI_BG = "#f3f0fc"
UI_CARD = "#e6e0f8"
UI_TEXT = "#1e1b4b"
UI_MUTED = "#5c5685"
UI_BORDER = "#d8d2f1"
UI_PRIMARY = "#00b368"
UI_PRIMARY_HOVER = "#008f52"
UI_SECONDARY = "#3b82f6"
UI_SECONDARY_HOVER = "#1d4ed8"
UI_ACCENT = "#7c3aed"
UI_WARNING = "#f97316"
UI_DANGER = "#ef4444"

COLOR_OPTIMAL = UI_PRIMARY
COLOR_OPTIMAL_DEEP = UI_PRIMARY_HOVER
COLOR_OPTIMAL_SOFT = "#A8E6C8"
COLOR_SUBOPTIMAL = "#F59E0B"
COLOR_SUBOPTIMAL_DEEP = "#D97706"
COLOR_SUBOPTIMAL_SOFT = "#FDE68A"
COLOR_CRITICAL = UI_DANGER
COLOR_CRITICAL_DEEP = UI_DANGER
COLOR_CRITICAL_SOFT = "#FECACA"


def _actives() -> dict[str, Any]:
    return get_actives()


def _agents() -> dict[str, Any]:
    """Alias legacy → actives."""
    return _actives()


def _bands() -> dict[str, Any]:
    return get_saturation_bands()


def _band_optimal() -> float:
    return float(_bands()["optimal"])


def _band_suboptimal() -> float:
    return float(_bands()["suboptimal"])


def _band_labels() -> dict[str, str]:
    return dict(_bands().get("labels") or {})


def _lags() -> dict[str, Any]:
    return get_lags()


def _windows() -> dict[str, Any]:
    return get_windows()


# Snapshot numerici al load (soglie / lag / finestre).
BAND_OPTIMAL = _band_optimal()
BAND_SUBOPTIMAL = _band_suboptimal()
LAG_DAYS_ACTIVE = int(_lags()["active"])
LAG_DAYS_SMOOTHNESS = int(_lags()["smoothness"])
LAG_DAYS_MARKS = int(_lags()["marks"])
LAG_TOLERANCE_DAYS = int(_lags()["tolerance"])
MIN_CORR_SAMPLES = int(_windows().get("min_corr_samples", 3))
PLOT_WINDOW_DAYS = int(_windows()["habits"])
HABITS_WINDOW_DAYS = int(_windows()["habits"])
RESULTS_WINDOW_DAYS = int(_windows()["results"])


def saturation_band_color(s: float) -> str:
    """Colore piatto di fascia (card / legenda)."""
    if s >= _band_optimal():
        return COLOR_OPTIMAL
    if s >= _band_suboptimal():
        return COLOR_SUBOPTIMAL
    return COLOR_CRITICAL


def _hex_to_rgb(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4))


def _lerp_rgb(
    c0: str, c1: str, t: float
) -> tuple[float, float, float]:
    t = float(np.clip(t, 0.0, 1.0))
    r0, g0, b0 = _hex_to_rgb(c0)
    r1, g1, b1 = _hex_to_rgb(c1)
    return (r0 + (r1 - r0) * t, g0 + (g1 - g0) * t, b0 + (b1 - b0) * t)


def saturation_heat_color(s: float) -> tuple[float, float, float]:
    """
    Heatmap bande: verde ottimale, giallo sub-ottimale, rosso critica.
    Soglie da routine.json.
    """
    s = float(np.clip(s, 0.0, 100.0))
    opt = _band_optimal()
    sub = _band_suboptimal()
    if s >= opt:
        t = (s - opt) / (100.0 - opt) if opt < 100 else 1.0
        return _lerp_rgb(COLOR_OPTIMAL_SOFT, COLOR_OPTIMAL_DEEP, t)
    if s >= sub:
        t = (s - sub) / (opt - sub) if opt > sub else 0.0
        return _lerp_rgb(COLOR_SUBOPTIMAL_DEEP, COLOR_SUBOPTIMAL_SOFT, t)
    t = s / sub if sub > 0 else 0.0
    return _lerp_rgb(COLOR_CRITICAL_DEEP, COLOR_CRITICAL_SOFT, t)


def saturation_band_label(s: float) -> str:
    labels = _band_labels()
    if s >= _band_optimal():
        return labels.get("optimal", "Ottimale")
    if s >= _band_suboptimal():
        return labels.get("suboptimal", "Sub-ottimale")
    return labels.get("critical", "Critica")


def compute_ema_saturation(df_routine: pd.DataFrame) -> pd.DataFrame:
    """
    Calcola S_i(t) ∈ [0, 100] per ogni attivo in routine.json.

    Un passo = un giorno solare. Ratio giornaliero clamped a 1.0:
      ratio = min(1, Dose_i(t) / Target_i)
      P_i(t) = α · ratio + (1-α) · P_i(t-1)
      S_i(t) = min(100, P_i(t) · 100)

    P iniziale da actives[*].initial_p.
    """
    df = df_routine.copy()
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
    else:
        df = df.sort_index()
        df.index = pd.to_datetime(df.index)

    out = pd.DataFrame(index=df.index)
    for agent, cfg in _agents().items():
        alpha = cfg["alpha"]
        target = cfg["target"]
        doses = df[cfg["dose_col"]].astype(float).fillna(0.0).to_numpy()
        p = float(cfg.get("initial_p", 0.0))
        sats = []
        for dose in doses:
            ratio = min(1.0, float(dose) / target) if target > 0 else 0.0
            p = alpha * ratio + (1.0 - alpha) * p
            sats.append(min(100.0, p * 100.0))
        out[f"S_{agent}"] = sats
        out[cfg["dose_col"]] = doses
    return out


def mean_saturation_in_window(
    sat_series: pd.Series,
    t_start: pd.Timestamp,
    t_end: pd.Timestamp,
) -> float:
    if pd.isna(t_start) or pd.isna(t_end):
        return float("nan")
    if t_end < t_start:
        t_start, t_end = t_end, t_start
    window = sat_series.loc[(sat_series.index >= t_start) & (sat_series.index <= t_end)]
    if window.empty:
        return float("nan")
    return float(window.mean())


def get_lagged_checkpoint(
    df_checkpoints: pd.DataFrame,
    current_idx: int,
    target_days: int,
    tolerance_days: int = LAG_TOLERANCE_DAYS,
) -> Optional[pd.Series]:
    """
    Checkpoint passato più vicino a `target_days` prima di current_idx
    (tolleranza ±tolerance_days). None se non esiste.
    """
    if current_idx <= 0 or df_checkpoints.empty:
        return None
    current_date = pd.to_datetime(df_checkpoints.loc[current_idx, "date"])
    past = df_checkpoints.loc[: current_idx - 1].copy()
    if past.empty:
        return None
    past = past.copy()
    past["day_diff"] = (current_date - pd.to_datetime(past["date"])).dt.days
    valid = past[np.abs(past["day_diff"] - target_days) <= tolerance_days]
    if valid.empty:
        return None
    best_idx = (valid["day_diff"] - target_days).abs().idxmin()
    return valid.loc[best_idx]


def compute_lag_deltas(
    df_checkpoints: pd.DataFrame,
    df_saturation: pd.DataFrame,
) -> pd.DataFrame:
    """
    Delta clinici + esposizione EMA media sulle finestre biologiche in giorni:
      active ≈ 5gg, smoothness ≈ 11gg, marks ≈ 22gg (± LAG_TOLERANCE_DAYS).
    """
    cp = df_checkpoints.copy()
    cp["date"] = pd.to_datetime(cp["date"])
    cp = cp.sort_values("date").reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    n = len(cp)
    lag_specs = [
        (
            "active",
            LAG_DAYS_ACTIVE,
            "active_area_pct",
            "delta_active",
            "expo_active",
        ),
        (
            "smooth",
            LAG_DAYS_SMOOTHNESS,
            "smoothness_score",
            "delta_smoothness",
            "expo_smooth",
        ),
        (
            "marks",
            LAG_DAYS_MARKS,
            "marks_severity_score",
            "delta_marks",
            "expo_marks",
        ),
    ]

    for k in range(n):
        row: dict[str, Any] = {
            "k": k,
            "date": cp.loc[k, "date"],
            "active_area_pct": float(cp.loc[k, "active_area_pct"]),
            "smoothness_score": float(cp.loc[k, "smoothness_score"]),
            "marks_severity_score": float(cp.loc[k, "marks_severity_score"]),
        }
        t_k = cp.loc[k, "date"]

        for _name, target_days, metric_col, delta_col, expo_prefix in lag_specs:
            lagged = get_lagged_checkpoint(cp, k, target_days)
            if lagged is None:
                row[delta_col] = np.nan
                for agent in _agents():
                    row[f"{expo_prefix}_{agent}"] = np.nan
                continue
            t_prev = pd.to_datetime(lagged["date"])
            row[delta_col] = float(cp.loc[k, metric_col]) - float(lagged[metric_col])
            for agent in _agents():
                row[f"{expo_prefix}_{agent}"] = mean_saturation_in_window(
                    df_saturation[f"S_{agent}"], t_prev, t_k
                )

        rows.append(row)

    return pd.DataFrame(rows)


def compute_spearman_matrix(
    df_deltas: pd.DataFrame,
) -> tuple[Optional[pd.DataFrame], int]:
    """
    Heatmap 3×3 Spearman per-cella.
    Ogni cella usa le coppie disponibili per quella metrica (≥ MIN_CORR_SAMPLES);
    le altre restano NaN («Nessun dato»).
    """
    agents = list(_agents().keys())
    col_specs = [
        ("delta_active", "expo_active"),
        ("delta_smoothness", "expo_smooth"),
        ("delta_marks", "expo_marks"),
    ]
    col_labels = [
        "Δ Area infiammata",
        "Δ Levigatezza",
        "Δ Gravità segni",
    ]
    row_labels = [_agents()[a]["label"] for a in agents]
    n_cols = len(col_specs)

    empty = pd.DataFrame(
        np.full((3, n_cols), np.nan), index=row_labels, columns=col_labels
    )

    if df_deltas is None or df_deltas.empty:
        return empty, 0

    matrix = np.full((3, n_cols), np.nan)
    n_max = 0
    for i, agent in enumerate(agents):
        for j, (delta_col, expo_prefix) in enumerate(col_specs):
            expo_col = f"{expo_prefix}_{agent}"
            if delta_col not in df_deltas.columns or expo_col not in df_deltas.columns:
                continue
            x = df_deltas[expo_col].astype(float)
            y = df_deltas[delta_col].astype(float)
            mask = x.notna() & y.notna()
            n = int(mask.sum())
            n_max = max(n_max, n)
            if n < MIN_CORR_SAMPLES:
                continue
            rho, _ = stats.spearmanr(x[mask], y[mask])
            matrix[i, j] = float(rho)

    return pd.DataFrame(matrix, index=row_labels, columns=col_labels), n_max


def _empty_checkpoint_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "date",
            "active_area_pct",
            "affected_area_pct",
            "active_spots_count",
            "smoothness_score",
            "marks_severity_score",
        ]
    )


def _resolve_plot_window(
    df_saturation: pd.DataFrame,
    df_checkpoints: pd.DataFrame,
    highlight_date: Optional[pd.Timestamp],
    window_days: int,
) -> tuple[
    Optional[pd.Timestamp],
    Optional[pd.Timestamp],
    pd.DataFrame,
    pd.DataFrame,
]:
    """Ritaglia sat/checkpoint agli ultimi `window_days` fino a highlight/fine dati."""
    cp = df_checkpoints.copy()
    if not cp.empty:
        cp["date"] = pd.to_datetime(cp["date"])
        cp = cp.sort_values("date")
    else:
        cp = _empty_checkpoint_frame()

    sat = df_saturation.copy()
    if not sat.empty and not isinstance(sat.index, pd.DatetimeIndex):
        sat.index = pd.to_datetime(sat.index)

    end: Optional[pd.Timestamp] = None
    if highlight_date is not None and not pd.isna(highlight_date):
        end = pd.to_datetime(highlight_date).normalize()
    else:
        ends: list[pd.Timestamp] = []
        if not sat.empty:
            ends.append(pd.Timestamp(sat.index.max()).normalize())
        if not cp.empty:
            ends.append(pd.Timestamp(cp["date"].max()).normalize())
        if ends:
            end = max(ends)

    start: Optional[pd.Timestamp] = None
    if end is not None and window_days > 0:
        start = end - pd.Timedelta(days=window_days - 1)
        if not sat.empty:
            sat = sat.loc[(sat.index >= start) & (sat.index <= end)]
        if not cp.empty:
            cp = cp[(cp["date"] >= start) & (cp["date"] <= end)].reset_index(drop=True)

    return start, end, sat, cp


def _style_metric_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(UI_BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(UI_BORDER)
    ax.spines["bottom"].set_color(UI_BORDER)
    ax.tick_params(colors=UI_MUTED, labelsize=8)
    ax.yaxis.label.set_color(UI_MUTED)
    ax.xaxis.label.set_color(UI_MUTED)
    ax.grid(True, axis="y", alpha=0.4, linestyle=":", linewidth=0.8, color=UI_BORDER)
    ax.grid(False, axis="x")


def _draw_guides(
    ax: plt.Axes,
    rasature: list,
    highlight_date: Optional[pd.Timestamp],
) -> None:
    for d in rasature:
        ax.axvline(d, color=UI_MUTED, linestyle="--", alpha=0.35, linewidth=0.9)
    if highlight_date is not None and not pd.isna(highlight_date):
        ax.axvline(
            pd.to_datetime(highlight_date),
            color=UI_ACCENT,
            linestyle="-",
            alpha=0.85,
            linewidth=1.5,
            zorder=5,
        )


def _finish_figure(fig: plt.Figure) -> None:
    fig.patch.set_facecolor(UI_CARD)
    for ax in fig.axes:
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily("sans-serif")


def _apply_xlim(
    axes: list[plt.Axes],
    start: Optional[pd.Timestamp],
    end: Optional[pd.Timestamp],
) -> None:
    if start is None or end is None:
        return
    x_right = end + pd.Timedelta(days=1)
    for ax in axes:
        ax.set_xlim(start, x_right)


def plot_agent_swimlane(
    df_saturation: pd.DataFrame,
    agent: str,
    df_checkpoints: pd.DataFrame | None = None,
    figsize: tuple[float, float] = (8.0, 0.42),
    highlight_date: Optional[pd.Timestamp] = None,
    window_days: int = HABITS_WINDOW_DAYS,
) -> plt.Figure:
    """Striscia colorata nuda (sfondo trasparente) per la card dell'attivo."""
    if agent not in _agents():
        raise ValueError(f"Unknown active: {agent}")

    cp_in = df_checkpoints if df_checkpoints is not None else _empty_checkpoint_frame()
    start, end, sat, _cp = _resolve_plot_window(
        df_saturation, cp_in, highlight_date, window_days
    )

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(UI_BG)
    ax.set_facecolor(UI_BG)
    ax.patch.set_alpha(1.0)

    if len(sat) >= 1 and f"S_{agent}" in sat.columns:
        dates = sat.index.to_list()
        series = sat[f"S_{agent}"]
        for i in range(len(dates)):
            d0 = dates[i]
            d1 = dates[i + 1] if i + 1 < len(dates) else d0 + pd.Timedelta(days=1)
            color = saturation_heat_color(float(series.iloc[i]))
            left = mdates.date2num(d0)
            width = mdates.date2num(d1) - left
            ax.barh(
                0.0,
                width,
                left=left,
                height=1.0,
                color=color,
                align="edge",
                linewidth=0,
            )

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim(0.0, 1.0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(False)
    ax.set_axis_off()

    if start is not None and end is not None:
        ax.set_xlim(start, end + pd.Timedelta(days=1))

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    return fig


def plot_habits_swimlane(
    df_saturation: pd.DataFrame,
    df_checkpoints: pd.DataFrame | None = None,
    figsize: tuple[float, float] = (11.0, 3.4),
    highlight_date: Optional[pd.Timestamp] = None,
    window_days: int = HABITS_WINDOW_DAYS,
) -> plt.Figure:
    """Compat/test: tre lane stackate (UI usa plot_agent_swimlane per card)."""
    cp_in = df_checkpoints if df_checkpoints is not None else _empty_checkpoint_frame()
    start, end, sat, cp = _resolve_plot_window(
        df_saturation, cp_in, highlight_date, window_days
    )
    rasature = cp["date"].tolist() if not cp.empty else []

    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    _style_metric_ax(ax)
    _draw_guides(ax, rasature, highlight_date)

    agents = list(_agents().keys())
    lane_height = 1.0
    y_positions = {a: (len(agents) - 1 - i) * lane_height for i, a in enumerate(agents)}

    if len(sat) >= 1:
        dates = sat.index.to_list()
        for agent in agents:
            y0 = y_positions[agent]
            series = sat[f"S_{agent}"]
            for i in range(len(dates)):
                d0 = dates[i]
                d1 = dates[i + 1] if i + 1 < len(dates) else d0 + pd.Timedelta(days=1)
                color = saturation_heat_color(float(series.iloc[i]))
                left = mdates.date2num(d0)
                width = mdates.date2num(d1) - left
                ax.barh(
                    y0,
                    width,
                    left=left,
                    height=lane_height * 0.78,
                    color=color,
                    align="edge",
                    linewidth=0,
                )
        ax.xaxis_date()

    ax.set_yticks([y_positions[a] + lane_height * 0.39 for a in agents])
    ax.set_yticklabels([_agents()[a]["label"] for a in agents], fontsize=9, color=UI_TEXT)
    ax.set_ylim(-0.2, len(agents) * lane_height + 0.15)
    ax.set_title(
        f"Saturazione EMA — ultime {window_days} gg",
        fontsize=11,
        fontweight="600",
        color=UI_TEXT,
        pad=8,
    )
    ax.set_xlabel("Data", fontsize=8, color=UI_MUTED)

    opt = _band_optimal()
    sub = _band_suboptimal()
    labels = _band_labels()
    legend_handles = [
        mpatches.Patch(
            color=COLOR_OPTIMAL,
            label=f"{labels.get('optimal', 'Ottimale')} ≥{opt:g}%",
        ),
        mpatches.Patch(
            color=COLOR_SUBOPTIMAL,
            label=f"{labels.get('suboptimal', 'Sub-ottimale')} {sub:g}–{opt - 1:g}%",
        ),
        mpatches.Patch(
            color=COLOR_CRITICAL,
            label=f"{labels.get('critical', 'Critica')} <{sub:g}%",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=7.5,
        frameon=False,
        labelcolor=UI_MUTED,
    )
    _apply_xlim([ax], start, end)
    _finish_figure(fig)
    fig.autofmt_xdate()
    return fig


def _style_results_ax(ax: plt.Axes) -> None:
    """Sfondo chiaro come le card attivi; senza label asse Y."""
    ax.set_facecolor(UI_BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(UI_BORDER)
    ax.spines["bottom"].set_color(UI_BORDER)
    ax.tick_params(colors=UI_MUTED, labelsize=8)
    ax.xaxis.label.set_color(UI_MUTED)
    ax.set_ylabel("")
    ax.grid(True, axis="y", alpha=0.65, linestyle="-", linewidth=1.0, color=UI_BORDER)
    ax.grid(False, axis="x")


def plot_results_metrics(
    df_checkpoints: pd.DataFrame,
    figsize: tuple[float, float] | None = None,
    highlight_date: Optional[pd.Timestamp] = None,
    window_days: int = RESULTS_WINDOW_DAYS,
    df_saturation: pd.DataFrame | None = None,
    page: int = 1,
) -> plt.Figure:
    """
    Sezione Risultati a pagine da 2 plot:
      page=1 → Marks + Active Spots
      page=2 → Active/Affected area + Smoothness
    Sfondo UI_BG (come card saturazione).
    """
    page = 1 if page not in (1, 2) else page
    if figsize is None:
        figsize = (9.5, 11.0)

    sat_in = (
        df_saturation
        if df_saturation is not None
        else pd.DataFrame(columns=["S_adapalene", "S_azelaic", "S_rederma"])
    )
    start, end, _sat, cp = _resolve_plot_window(
        sat_in, df_checkpoints, highlight_date, window_days
    )

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True, constrained_layout=True)
    fig.patch.set_facecolor(UI_BG)
    ax_a, ax_b = axes

    for ax in axes:
        _style_results_ax(ax)
        if highlight_date is not None and not pd.isna(highlight_date):
            ax.axvline(
                pd.to_datetime(highlight_date),
                color=UI_ACCENT,
                linestyle="-",
                alpha=0.85,
                linewidth=1.5,
                zorder=5,
            )

    if page == 1:
        # Marks
        if not cp.empty and "marks_severity_score" in cp.columns:
            ax_a.plot(
                cp["date"],
                cp["marks_severity_score"],
                color=UI_ACCENT,
                marker="o",
                markersize=5,
                linewidth=1.8,
                zorder=3,
            )
            ax_a.fill_between(
                cp["date"], cp["marks_severity_score"], alpha=0.15, color=UI_ACCENT
            )
            ymax = float(cp["marks_severity_score"].max())
            ax_a.set_ylim(0.0, max(ymax * 1.15, 1.0))
        else:
            ax_a.set_ylim(0.0, 1.0)
        ax_a.set_title(
            "Gravità segni (discromia)",
            fontsize=11,
            fontweight="600",
            color=UI_TEXT,
            pad=6,
        )

        # Active spots
        if not cp.empty and "active_spots_count" in cp.columns:
            ax_b.bar(
                cp["date"],
                cp["active_spots_count"],
                width=0.55,
                color=UI_DANGER,
                alpha=0.85,
                align="center",
                zorder=3,
                linewidth=0,
            )
            ymax = max(float(cp["active_spots_count"].max()), 1.0)
            ax_b.set_ylim(0, ymax * 1.25 + 0.5)
        else:
            ax_b.set_ylim(0, 5)
        ax_b.set_title("Spot attivi", fontsize=11, fontweight="600", color=UI_TEXT, pad=6)
        ax_b.set_xlabel("Data", fontsize=8, color=UI_MUTED)
    else:
        # Active + Affected
        if not cp.empty:
            if "active_area_pct" in cp.columns:
                ax_a.plot(
                    cp["date"],
                    cp["active_area_pct"],
                    color=UI_DANGER,
                    marker="o",
                    markersize=5,
                    linewidth=1.8,
                    label="Area attiva / interessata %",
                    zorder=3,
                )
            if "affected_area_pct" in cp.columns:
                ax_a.plot(
                    cp["date"],
                    cp["affected_area_pct"],
                    color=COLOR_SUBOPTIMAL,
                    marker="s",
                    markersize=4.5,
                    linewidth=1.6,
                    linestyle="--",
                    label="Area interessata / ROI %",
                    zorder=3,
                )
            ax_a.legend(loc="upper right", fontsize=7, frameon=False, labelcolor=UI_MUTED)
        ax_a.set_ylim(0, None)
        ax_a.set_title(
            "Area attiva / Area interessata",
            fontsize=11,
            fontweight="600",
            color=UI_TEXT,
            pad=6,
        )

        # Smoothness
        if not cp.empty and "smoothness_score" in cp.columns:
            ax_b.plot(
                cp["date"],
                cp["smoothness_score"],
                color=UI_PRIMARY,
                marker="o",
                markersize=5,
                linewidth=1.8,
                zorder=3,
            )
        ax_b.set_title(
            "Levigatezza", fontsize=11, fontweight="600", color=UI_TEXT, pad=6
        )
        ax_b.set_xlabel("Data", fontsize=8, color=UI_MUTED)

    _apply_xlim(list(axes), start, end)
    fig.autofmt_xdate()
    return fig


def plot_clinical_dashboard(
    df_saturation: pd.DataFrame,
    df_checkpoints: pd.DataFrame,
    figsize: tuple[float, float] = (11.0, 12.0),
    highlight_date: Optional[pd.Timestamp] = None,
    window_days: int = PLOT_WINDOW_DAYS,
) -> plt.Figure:
    """Compat: stack risultati + swimlane (test / export unico)."""
    # Usato dai test legacy: preferisci plot_results + habits separati in produzione.
    fig_res = plot_results_metrics(
        df_checkpoints,
        figsize=(figsize[0], figsize[1] * 0.72),
        highlight_date=highlight_date,
        window_days=window_days,
        df_saturation=df_saturation,
    )
    fig_hab = plot_habits_swimlane(
        df_saturation,
        df_checkpoints,
        figsize=(figsize[0], figsize[1] * 0.28),
        highlight_date=highlight_date,
        window_days=window_days,
    )
    # Ritorna solo results per non rompere assunti sui 4 axes; habits è generato a parte.
    plt.close(fig_hab)
    return fig_res


def plot_correlation_matrix(
    df_deltas: pd.DataFrame,
    figsize: tuple[float, float] = (7.5, 5.5),
) -> plt.Figure:
    matrix, _n_max = compute_spearman_matrix(df_deltas)
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    fig.patch.set_facecolor(UI_BG)
    ax.set_facecolor(UI_BG)

    if matrix is None:
        matrix = pd.DataFrame()

    data = matrix.to_numpy(dtype=float) if not matrix.empty else np.full((3, 3), np.nan)
    from matplotlib.colors import LinearSegmentedColormap

    cmap = LinearSegmentedColormap.from_list(
        "skincare_div",
        [UI_PRIMARY_HOVER, "#EDE9FE", UI_DANGER],
    ).with_extremes(bad="#E8E4F5")
    masked = np.ma.masked_invalid(data)
    im = ax.imshow(masked, cmap=cmap, vmin=-1.0, vmax=1.0, aspect="auto")

    cols = list(matrix.columns) if not matrix.empty else [
        "Δ Area infiammata",
        "Δ Levigatezza",
        "Δ Gravità segni",
    ]
    rows = list(matrix.index) if not matrix.empty else ["Adapalene", "Azelaico", "Rederma"]
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=25, ha="right", color=UI_MUTED)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, color=UI_TEXT)
    ax.tick_params(colors=UI_MUTED)
    for spine in ax.spines.values():
        spine.set_color(UI_BORDER)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                ax.text(
                    j,
                    i,
                    "Nessun\ndato",
                    ha="center",
                    va="center",
                    color=UI_MUTED,
                    fontsize=7.5,
                    linespacing=1.1,
                )
            else:
                ax.text(
                    j, i, f"{val:.2f}", ha="center", va="center", color=UI_TEXT, fontsize=10
                )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors=UI_MUTED, labelsize=8)
    cbar.outline.set_edgecolor(UI_BORDER)
    return fig


def fig_to_png_bytes(
    fig: plt.Figure,
    dpi: int = 120,
    *,
    transparent: bool = False,
) -> bytes:
    buf = BytesIO()
    save_kw: dict[str, Any] = {
        "format": "png",
        "dpi": dpi,
        "edgecolor": "none",
        "transparent": transparent,
    }
    if transparent:
        save_kw["facecolor"] = "none"
        save_kw["bbox_inches"] = "tight"
        save_kw["pad_inches"] = 0
    else:
        save_kw["facecolor"] = fig.get_facecolor()
        save_kw["bbox_inches"] = "tight"
    fig.savefig(buf, **save_kw)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def build_routine_dataframe(daily_logs: list[dict[str, Any]]) -> pd.DataFrame:
    """
    Costruisce dose giornaliere per attivo da log_flags in routine.json.
    Dose = somma dei flag booleani elencati in agents[*].log_flags.
    """
    agents = _agents()
    dose_cols = [cfg["dose_col"] for cfg in agents.values()]
    if not daily_logs:
        return pd.DataFrame(columns=["date", *dose_cols])

    records = []
    for log in daily_logs:
        row: dict[str, Any] = {"date": pd.to_datetime(log["entry_date"])}
        for cfg in agents.values():
            row[cfg["dose_col"]] = float(
                sum(1 for flag in cfg["log_flags"] if log.get(flag))
            )
        records.append(row)
    df = pd.DataFrame(records).drop_duplicates("date").sort_values("date")
    full_idx = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    df = df.set_index("date").reindex(full_idx, fill_value=0.0)
    df.index.name = "date"
    return df.reset_index().rename(columns={"index": "date"})


def build_checkpoint_dataframe(checkpoints: list[dict[str, Any]]) -> pd.DataFrame:
    """marks_severity_score ← discromia_score se assente."""
    rows = []
    for cp in checkpoints:
        analysis = cp.get("analysis") or {}
        marks = analysis.get("marks_severity_score")
        if marks is None:
            marks = analysis.get("discromia_score", 0.0)
        rows.append(
            {
                "date": pd.to_datetime(cp["entry_date"]),
                "active_area_pct": float(analysis.get("active_area_pct", 0.0)),
                "affected_area_pct": float(analysis.get("affected_area_pct", 0.0)),
                "active_spots_count": int(analysis.get("active_spots_count", 0)),
                "smoothness_score": float(analysis.get("smoothness_score", 100.0)),
                "marks_severity_score": float(marks),
            }
        )
    if not rows:
        return _empty_checkpoint_frame()
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


def _rgb_to_hex(rgb: tuple[float, float, float]) -> str:
    r, g, b = [int(round(c * 255)) for c in rgb]
    return f"#{r:02X}{g:02X}{b:02X}"


def run_clinical_pipeline(
    daily_logs: list[dict[str, Any]],
    checkpoints: list[dict[str, Any]],
    as_of: Optional[str] = None,
    window_days: Optional[int] = None,
    results_window_days: Optional[int] = None,
) -> dict[str, Any]:
    """
    EMA + delta + PNG per card/risultati/correlazione.

    as_of: data ISO. EMA su storia ≤ as_of.
    window_days: finestra swimlane abitudini (default HABITS_WINDOW_DAYS).
    results_window_days: finestra grafici risultati (default RESULTS_WINDOW_DAYS).
    """
    w_habits = int(window_days) if window_days is not None else HABITS_WINDOW_DAYS
    w_habits = max(3, min(90, w_habits))
    w_results = (
        int(results_window_days)
        if results_window_days is not None
        else RESULTS_WINDOW_DAYS
    )
    w_results = max(3, min(90, w_results))

    df_routine = build_routine_dataframe(daily_logs)
    df_cp = build_checkpoint_dataframe(checkpoints)

    highlight = pd.to_datetime(as_of) if as_of else None

    if highlight is not None and not df_routine.empty:
        df_routine = df_routine[pd.to_datetime(df_routine["date"]) <= highlight]
    if highlight is not None and not df_cp.empty:
        df_cp = df_cp[df_cp["date"] <= highlight].reset_index(drop=True)

    if df_routine.empty:
        df_sat = pd.DataFrame(columns=[f"S_{a}" for a in _agents()])
    else:
        df_sat = compute_ema_saturation(df_routine)

    if not df_cp.empty and not df_sat.empty:
        df_deltas = compute_lag_deltas(df_cp, df_sat)
    else:
        df_deltas = pd.DataFrame()

    swimlane_pngs: dict[str, bytes] = {}
    for agent in _agents():
        fig_lane = plot_agent_swimlane(
            df_sat,
            agent,
            df_cp,
            highlight_date=highlight,
            window_days=w_habits,
        )
        swimlane_pngs[agent] = fig_to_png_bytes(fig_lane, transparent=True)

    fig_results_p1 = plot_results_metrics(
        df_cp,
        highlight_date=highlight,
        window_days=w_results,
        df_saturation=df_sat,
        page=1,
    )
    fig_results_p2 = plot_results_metrics(
        df_cp,
        highlight_date=highlight,
        window_days=w_results,
        df_saturation=df_sat,
        page=2,
    )
    fig_corr = plot_correlation_matrix(
        df_deltas if not df_deltas.empty else pd.DataFrame()
    )

    latest_sat: dict[str, Any] = {}
    if not df_sat.empty:
        if highlight is not None:
            eligible = df_sat.loc[df_sat.index <= highlight]
            last = eligible.iloc[-1] if not eligible.empty else df_sat.iloc[-1]
        else:
            last = df_sat.iloc[-1]
        for agent in _agents():
            s = float(last[f"S_{agent}"])
            heat = saturation_heat_color(s)
            latest_sat[agent] = {
                "saturation_pct": round(s, 1),
                "band": saturation_band_label(s),
                "color": _rgb_to_hex(heat),
            }

    return {
        "df_saturation": df_sat,
        "df_checkpoints": df_cp,
        "df_deltas": df_deltas,
        "swimlane_pngs": swimlane_pngs,
        "results_page1_png": fig_to_png_bytes(fig_results_p1),
        "results_page2_png": fig_to_png_bytes(fig_results_p2),
        "correlation_png": fig_to_png_bytes(fig_corr),
        "latest_saturation": latest_sat,
        "checkpoint_count": int(len(df_cp)),
        "as_of": as_of,
        "window_days": w_habits,
        "results_window_days": w_results,
    }
