from __future__ import annotations

from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import numpy as np

from .trajectory import TrajectoryPoint


def save_trajectory_csv(traj: Sequence[TrajectoryPoint], path: str) -> None:
    import csv
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["t", "x", "y", "yaw"])
        for p in traj:
            w.writerow([f"{p.t:.6f}", f"{p.x:.6f}", f"{p.y:.6f}", f"{p.yaw:.6f}"])


def save_trajectory_plot(traj: Sequence[TrajectoryPoint], path: str) -> None:
    xs = np.array([p.x for p in traj], dtype=float)
    ys = np.array([p.y for p in traj], dtype=float)

    plt.figure()
    plt.plot(xs, ys)
    plt.axis("equal")
    plt.xlabel("x, m")
    plt.ylabel("y, m")
    plt.title("Estimated trajectory")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
