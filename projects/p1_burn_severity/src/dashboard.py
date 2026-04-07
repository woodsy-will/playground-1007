"""Interactive Plotly Dash dashboard for burn severity and recovery.

Provides a web-based viewer with a severity classification map,
recovery time-series charts, and summary statistics panel.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shared.utils.logging import get_logger

logger = get_logger("p1.dashboard")

try:
    import dash  # type: ignore[import-untyped]
    from dash import dcc, html  # type: ignore[import-untyped]
    import plotly.express as px  # type: ignore[import-untyped]
    import plotly.graph_objects as go  # type: ignore[import-untyped]

    _HAS_DASH = True
except ImportError:
    _HAS_DASH = False
    logger.warning("dash / plotly not installed; dashboard disabled.")

_SEVERITY_LABELS: dict[int, str] = {
    0: "Unburned",
    1: "Low",
    2: "Moderate-Low",
    3: "Moderate-High",
    4: "High",
}

_SEVERITY_COLORS: dict[int, str] = {
    0: "#1a9641",
    1: "#a6d96a",
    2: "#ffffbf",
    3: "#fdae61",
    4: "#d7191c",
}


def _load_severity_raster(severity_path: str | Path) -> np.ndarray:
    """Read severity raster via shared I/O."""
    from shared.utils.io import read_raster

    data, _ = read_raster(severity_path)
    if data.ndim == 3:
        data = data[0]
    return data


def _build_severity_figure(severity: np.ndarray) -> go.Figure:
    """Create a heatmap of burn severity classes."""
    # Replace nodata (255) with NaN for display
    display = severity.astype(np.float32)
    display[severity == 255] = np.nan

    fig = px.imshow(
        display,
        color_continuous_scale=[
            [0.0, _SEVERITY_COLORS[0]],
            [0.25, _SEVERITY_COLORS[1]],
            [0.5, _SEVERITY_COLORS[2]],
            [0.75, _SEVERITY_COLORS[3]],
            [1.0, _SEVERITY_COLORS[4]],
        ],
        labels={"color": "Severity Class"},
        title="Burn Severity Classification",
        zmin=0,
        zmax=4,
    )
    fig.update_layout(coloraxis_colorbar=dict(
        tickvals=[0, 1, 2, 3, 4],
        ticktext=list(_SEVERITY_LABELS.values()),
    ))
    return fig


def _build_recovery_figure(recovery_df: pd.DataFrame) -> go.Figure:
    """Create line chart of recovery index by severity class over time."""
    df = recovery_df.copy()
    df["severity_label"] = df["severity_class"].map(_SEVERITY_LABELS)
    fig = px.line(
        df,
        x="year",
        y="mean_index",
        color="severity_label",
        markers=True,
        title="Vegetation Recovery by Severity Class",
        labels={
            "mean_index": "Mean Vegetation Index",
            "year": "Years Post-Fire",
            "severity_label": "Severity",
        },
        color_discrete_map={v: _SEVERITY_COLORS[k] for k, v in _SEVERITY_LABELS.items()},
    )
    return fig


def _build_summary_table(severity: np.ndarray) -> pd.DataFrame:
    """Compute per-class pixel counts and area percentages."""
    valid = severity[severity != 255]
    total = len(valid)
    rows = []
    for code, label in _SEVERITY_LABELS.items():
        count = int(np.count_nonzero(valid == code))
        pct = 100.0 * count / max(total, 1)
        rows.append({"Class": label, "Pixels": count, "Percent": f"{pct:.1f}%"})
    return pd.DataFrame(rows)


def create_app(
    severity_path: str | Path,
    recovery_df: pd.DataFrame,
    config: dict[str, Any],
) -> "dash.Dash":
    """Create a Plotly Dash application for burn severity exploration.

    Parameters
    ----------
    severity_path : str or Path
        Path to severity classification GeoTIFF.
    recovery_df : pd.DataFrame
        Recovery time-series dataframe from
        :func:`recovery.build_recovery_timeseries`.
    config : dict
        Project configuration.

    Returns
    -------
    dash.Dash
        Configured Dash application (call ``.run_server()`` to launch).

    Raises
    ------
    RuntimeError
        If ``dash`` / ``plotly`` are not installed.
    """
    if not _HAS_DASH:
        raise RuntimeError(
            "dash and plotly are required for the dashboard. "
            "Install with: pip install dash plotly"
        )

    severity = _load_severity_raster(severity_path)
    sev_fig = _build_severity_figure(severity)
    rec_fig = _build_recovery_figure(recovery_df)
    summary = _build_summary_table(severity)

    app = dash.Dash(__name__, title="Burn Severity Dashboard")

    app.layout = html.Div(
        [
            html.H1("Post-Wildfire Burn Severity & Recovery Tracker"),
            html.Div(
                [
                    html.Div(
                        [dcc.Graph(figure=sev_fig, id="severity-map")],
                        style={"width": "50%", "display": "inline-block"},
                    ),
                    html.Div(
                        [dcc.Graph(figure=rec_fig, id="recovery-chart")],
                        style={"width": "50%", "display": "inline-block"},
                    ),
                ]
            ),
            html.H2("Summary Statistics"),
            html.Table(
                # Header
                [html.Tr([html.Th(col) for col in summary.columns])]
                # Body
                + [
                    html.Tr([html.Td(summary.iloc[i][col]) for col in summary.columns])
                    for i in range(len(summary))
                ],
                style={"margin": "auto", "borderCollapse": "collapse"},
            ),
        ],
        style={"fontFamily": "Arial, sans-serif", "padding": "20px"},
    )

    return app
