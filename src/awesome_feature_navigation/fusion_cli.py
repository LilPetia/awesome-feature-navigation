from __future__ import annotations

import argparse
from pathlib import Path

from .fusion import fuse_trajectories, load_trajectory_csv
from .plotting import save_trajectory_csv, save_trajectory_plot


def main() -> None:
    ap = argparse.ArgumentParser(description='Fuse left/right camera trajectory CSV files into one 2D trajectory')
    ap.add_argument('--left', required=True, help='Trajectory CSV built from Left_cam.mp4')
    ap.add_argument('--right', required=True, help='Trajectory CSV built from Right_cam.mp4')
    ap.add_argument('--out', default='fused_trajectory', help='Output prefix (without extension)')
    ap.add_argument('--samples', type=int, default=512, help='Number of closed-loop samples used for fusion')
    ap.add_argument(
        '--phase-search-fraction',
        type=float,
        default=0.5,
        help='Fraction of the loop searched when aligning phase between LEFT and RIGHT',
    )
    ap.add_argument(
        '--no-reverse-search',
        action='store_true',
        help='Disable automatic reverse-direction check for the left trajectory',
    )
    ap.add_argument(
        '--save-aligned-debug',
        action='store_true',
        help='Also save resampled RIGHT and aligned LEFT CSV/HTML outputs',
    )
    ap.add_argument('--axis-unit', default='meters', help='Axis unit label for HTML plots')
    args = ap.parse_args()

    left_traj = load_trajectory_csv(args.left)
    right_traj = load_trajectory_csv(args.right)
    result = fuse_trajectories(
        left_traj,
        right_traj,
        sample_count=args.samples,
        phase_search_fraction=args.phase_search_fraction,
        allow_reverse=not args.no_reverse_search,
    )

    out_prefix = Path(args.out)
    csv_path = str(Path(str(out_prefix) + '.csv'))
    html_path = str(Path(str(out_prefix) + '.html'))
    save_trajectory_csv(result.fused_traj, csv_path)
    save_trajectory_plot(
        result.fused_traj,
        html_path,
        title='Fused Left/Right Trajectory',
        axis_unit=args.axis_unit,
    )
    print(f'Saved: {csv_path}')
    print(f'Saved: {html_path}')
    print(f'Samples: {result.diagnostics.sample_count}')
    print(f'LEFT reverse used: {result.diagnostics.reverse_used}')
    print(f'LEFT phase shift: {result.diagnostics.phase_shift}')
    print(f'LEFT -> RIGHT alignment RMSE: {result.diagnostics.alignment_rmse:.6f}')

    if args.save_aligned_debug:
        left_prefix = Path(str(out_prefix) + '_left_aligned')
        right_prefix = Path(str(out_prefix) + '_right_resampled')
        left_csv = str(left_prefix.with_suffix('.csv'))
        left_html = str(left_prefix.with_suffix('.html'))
        right_csv = str(right_prefix.with_suffix('.csv'))
        right_html = str(right_prefix.with_suffix('.html'))
        save_trajectory_csv(result.left_aligned_traj, left_csv)
        save_trajectory_plot(
            result.left_aligned_traj,
            left_html,
            title='LEFT Aligned To RIGHT',
            axis_unit=args.axis_unit,
        )
        save_trajectory_csv(result.right_resampled_traj, right_csv)
        save_trajectory_plot(
            result.right_resampled_traj,
            right_html,
            title='RIGHT Resampled Reference',
            axis_unit=args.axis_unit,
        )
        print(f'Saved: {left_csv}')
        print(f'Saved: {left_html}')
        print(f'Saved: {right_csv}')
        print(f'Saved: {right_html}')


if __name__ == '__main__':
    main()
