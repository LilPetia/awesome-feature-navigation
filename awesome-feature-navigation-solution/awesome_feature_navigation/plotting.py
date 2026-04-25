from __future__ import annotations

import csv
from typing import Sequence

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .trajectory import LoopAveragingDebug, TrajectoryPoint

_PLOT_COLORS = (
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf',
)


def save_trajectory_csv(traj: Sequence[TrajectoryPoint], path: str) -> None:
    """Сохранить траекторию в CSV (колонки: t, x, y, yaw)."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['t', 'x', 'y', 'yaw'])
        for p in traj:
            w.writerow([f'{p.t:.6f}', f'{p.x:.6f}', f'{p.y:.6f}', f'{p.yaw:.6f}'])


def save_trajectory_plot(
    traj: Sequence[TrajectoryPoint],
    path: str,
    title: str='Estimated Trajectory (Interactive)',
) -> None:
    """Сохранить интерактивный 2D-график траектории (Plotly HTML или статическая картинка)."""
    if not traj:
        print('Warning: Trajectory is empty, skipping plot.')
        return
    ts = np.array([p.t for p in traj])
    xs = np.array([p.x for p in traj])
    ys = np.array([p.y for p in traj])
    yaws = np.array([p.yaw for p in traj])
    hover_texts = [
        f't={t:.2f}s<br>x={x:.2f}m<br>y={y:.2f}m<br>yaw={yaw:.2f}rad'
        for t, x, y, yaw in zip(ts, xs, ys, yaws)
    ]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=ys,
            mode='lines',
            name='Trajectory',
            line=dict(color='royalblue', width=2),
            text=hover_texts,
            hoverinfo='text',
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[xs[0]],
            y=[ys[0]],
            mode='markers',
            name='Start',
            marker=dict(color='green', size=10, symbol='circle'),
            hoverinfo='skip',
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[xs[-1]],
            y=[ys[-1]],
            mode='markers',
            name='End',
            marker=dict(color='red', size=10, symbol='x'),
            hoverinfo='skip',
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title='X (meters)',
        yaxis_title='Y (meters)',
        yaxis=dict(scaleanchor='x', scaleratio=1),
        legend=dict(x=0.01, y=0.99),
        template='plotly_white',
        dragmode='pan',
    )
    if path.endswith('.html'):
        fig.write_html(path)
    else:
        try:
            fig.write_image(path, scale=2)
        except ValueError as e:
            print(f'Error saving static image using Plotly: {e}')
            print('To save as PNG/JPG, please install kaleido: pip install kaleido')
            print('Alternatively, use .html extension for interactive plot.')


def save_loop_debug_csv(loop_debug: LoopAveragingDebug, path: str) -> None:
    """Сохранить отладку петель (по каждому кругу: raw/aligned/projected XY, фаза, scale, RMSE) в CSV."""
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow([
            'lap_index',
            'phase',
            'raw_x',
            'raw_y',
            'aligned_x',
            'aligned_y',
            'projected_x',
            'projected_y',
            'kept',
            'phase_shift',
            'scale',
            'alignment_rmse',
            'projection_rmse',
            't0',
            't1',
        ])
        denom = max(1, loop_debug.samples_per_lap)
        for lap in loop_debug.laps:
            for idx, (raw_pt, aligned_pt, projected_pt) in enumerate(zip(lap.raw_xy, lap.aligned_xy, lap.projected_xy)):
                w.writerow([
                    lap.lap_index,
                    f'{idx / denom:.6f}',
                    f'{raw_pt[0]:.6f}',
                    f'{raw_pt[1]:.6f}',
                    f'{aligned_pt[0]:.6f}',
                    f'{aligned_pt[1]:.6f}',
                    f'{projected_pt[0]:.6f}',
                    f'{projected_pt[1]:.6f}',
                    int(lap.kept),
                    lap.phase_shift,
                    f'{lap.scale:.6f}',
                    f'{lap.alignment_rmse:.6f}',
                    f'{lap.projection_rmse:.6f}',
                    f'{lap.t0:.6f}',
                    f'{lap.t1:.6f}',
                ])


def save_loop_debug_plot(loop_debug: LoopAveragingDebug, path: str) -> None:
    """Построить интерактивный график отладки петель (raw / aligned / projected + усреднённый сплайн)."""
    if not loop_debug.laps:
        print('Warning: Loop debug is empty, skipping loop plot.')
        return
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=('Raw Laps', 'Aligned Laps', 'Projected Laps + Spline'),
        horizontal_spacing=0.06,
    )
    for idx, lap in enumerate(loop_debug.laps):
        color = _PLOT_COLORS[idx % len(_PLOT_COLORS)]
        if lap.kept:
            line_style = dict(color=color, width=2)
            name = f'Lap {lap.lap_index + 1}'
        else:
            line_style = dict(color='rgba(120, 120, 120, 0.7)', width=1.5, dash='dot')
            name = f'Lap {lap.lap_index + 1} (rejected)'
        hover = (
            f'lap={lap.lap_index + 1}<br>'
            f'keep={lap.kept}<br>'
            f'shift={lap.phase_shift}<br>'
            f'scale={lap.scale:.3f}<br>'
            f'align_rmse={lap.alignment_rmse:.4f}<br>'
            f'proj_rmse={lap.projection_rmse:.4f}'
        )
        fig.add_trace(
            go.Scatter(
                x=lap.raw_xy[:, 0],
                y=lap.raw_xy[:, 1],
                mode='lines',
                name=name,
                line=line_style,
                hovertemplate=hover + '<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>',
                legendgroup=f'lap_{lap.lap_index}',
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=lap.aligned_xy[:, 0],
                y=lap.aligned_xy[:, 1],
                mode='lines',
                name=name,
                line=line_style,
                hovertemplate=hover + '<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>',
                legendgroup=f'lap_{lap.lap_index}',
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=lap.projected_xy[:, 0],
                y=lap.projected_xy[:, 1],
                mode='lines',
                name=name,
                line=line_style,
                hovertemplate=hover + '<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>',
                legendgroup=f'lap_{lap.lap_index}',
                showlegend=False,
            ),
            row=1,
            col=3,
        )
    fig.add_trace(
        go.Scatter(
            x=loop_debug.canonical_xy[:, 0],
            y=loop_debug.canonical_xy[:, 1],
            mode='lines',
            name='Average',
            line=dict(color='black', width=4),
            hovertemplate='average<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>',
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=[loop_debug.canonical_xy[0, 0]],
            y=[loop_debug.canonical_xy[0, 1]],
            mode='markers',
            name='Average Start',
            marker=dict(color='green', size=10, symbol='circle'),
            hoverinfo='skip',
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=loop_debug.spline_control_xy[:, 0],
            y=loop_debug.spline_control_xy[:, 1],
            mode='markers',
            name='Spline Controls',
            marker=dict(color='black', size=5, symbol='diamond'),
            hovertemplate='control<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>',
        ),
        row=1,
        col=3,
    )
    fig.update_xaxes(title_text='X (meters)', row=1, col=1)
    fig.update_yaxes(title_text='Y (meters)', row=1, col=1, scaleanchor='x', scaleratio=1)
    fig.update_xaxes(title_text='X (meters)', row=1, col=2)
    fig.update_yaxes(title_text='Y (meters)', row=1, col=2, scaleanchor='x2', scaleratio=1)
    fig.update_xaxes(title_text='X (meters)', row=1, col=3)
    fig.update_yaxes(title_text='Y (meters)', row=1, col=3, scaleanchor='x3', scaleratio=1)
    fig.update_layout(
        title=(
            'Loop Diagnostics'
            f' (period={loop_debug.period_sec:.2f}s, samples={loop_debug.samples_per_lap})'
        ),
        template='plotly_white',
        dragmode='pan',
        legend=dict(x=0.01, y=0.99),
    )
    if path.endswith('.html'):
        fig.write_html(path)
    else:
        try:
            fig.write_image(path, scale=2)
        except ValueError as e:
            print(f'Error saving static image using Plotly: {e}')
            print('To save as PNG/JPG, please install kaleido: pip install kaleido')
            print('Alternatively, use .html extension for interactive plot.')
