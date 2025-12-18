from __future__ import annotations

import csv
from typing import Sequence

import plotly.graph_objects as go
import numpy as np

from .trajectory import TrajectoryPoint


def save_trajectory_csv(traj: Sequence[TrajectoryPoint], path: str) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "yaw"])
        for p in traj:
            w.writerow([f"{p.t:.6f}", f"{p.x:.6f}", f"{p.y:.6f}", f"{p.yaw:.6f}"])


def save_trajectory_plot(traj: Sequence[TrajectoryPoint], path: str) -> None:
    if not traj:
        print("Warning: Trajectory is empty, skipping plot.")
        return

    ts = np.array([p.t for p in traj])
    xs = np.array([p.x for p in traj])
    ys = np.array([p.y for p in traj])
    yaws = np.array([p.yaw for p in traj])

    hover_texts = [
        f"t={t:.2f}s<br>x={x:.2f}m<br>y={y:.2f}m<br>yaw={yaw:.2f}rad"
        for t, x, y, yaw in zip(ts, xs, ys, yaws)
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode='lines',
        name='Trajectory',
        line=dict(color='royalblue', width=2),
        text=hover_texts,
        hoverinfo="text"
    ))

    fig.add_trace(go.Scatter(
        x=[xs[0]], y=[ys[0]],
        mode='markers',
        name='Start',
        marker=dict(color='green', size=10, symbol='circle'),
        hoverinfo='skip'
    ))

    fig.add_trace(go.Scatter(
        x=[xs[-1]], y=[ys[-1]],
        mode='markers',
        name='End',
        marker=dict(color='red', size=10, symbol='x'),
        hoverinfo='skip'
    ))

    fig.update_layout(
        title="Estimated Trajectory (Interactive)",
        xaxis_title="X (meters)",
        yaxis_title="Y (meters)",
        yaxis=dict(
            scaleanchor="x",
            scaleratio=1,
        ),
        legend=dict(x=0.01, y=0.99),
        template="plotly_white",
        dragmode="pan"
    )

    if path.endswith(".html"):
        fig.write_html(path)
    else:
        try:
            fig.write_image(path, scale=2)
        except ValueError as e:
            print(f"Error saving static image using Plotly: {e}")
            print("To save as PNG/JPG, please install kaleido: pip install kaleido")
            print("Alternatively, use .html extension for interactive plot.")